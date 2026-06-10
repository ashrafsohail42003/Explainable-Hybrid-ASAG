"""Train one cross-encoder, predict per item — the inner loop the protocol calls.

Kept separate from the evaluation protocol (``evaluate_neural.py``) so this file
only knows "fit on these rows, predict those rows" and stays task-aware via
``tasks.TaskSpec``. Predictions come back in the **metric space** that
``metrics.compute_metrics`` expects: class codes for classification, the integer
score for ordinal (rank → training-score value), the raw float for regression —
so neural numbers drop straight into the same metric functions as the GBM.
"""

from __future__ import annotations

import copy

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from asag.config import NeuralCfg
from asag.models.metrics import compute_metrics
from asag.models.tasks import TaskSpec
from asag.neural.dataset import PairDataset
from asag.neural.heads import ordinal_loss, ordinal_predict
from asag.neural.model import CrossEncoderGrader
from asag.utils.logging import get_logger
from asag.utils.seed import set_global_seed

log = get_logger()


class LabelSpace:
    """Map a task's raw targets to model space and back.

    * classification — label strings → contiguous codes (vocab from train).
    * ordinal        — sorted unique training scores → ranks ``0..K-1`` (and back).
    * regression     — identity on the float score.
    """

    def __init__(self, spec: TaskSpec, train_df: pd.DataFrame):
        self.spec = spec
        self.kind = spec.task_type
        if self.kind == "classification":
            labs = sorted(s for s in train_df["label"].astype(str).unique() if s != "")
            self.vocab = {lab: i for i, lab in enumerate(labs)}
            self.inv = {i: lab for lab, i in self.vocab.items()}
            self.num_classes = len(labs)
        elif self.kind == "ordinal":
            scores = sorted({int(round(float(s)))
                             for s in pd.to_numeric(train_df["score"], errors="coerce").dropna()})
            self.ranks = {s: i for i, s in enumerate(scores)}
            self.score_of_rank = {i: s for s, i in self.ranks.items()}
            self.num_classes = len(scores)
        else:
            self.num_classes = 0

    def targets(self, df: pd.DataFrame) -> np.ndarray:
        if self.kind == "classification":
            return df["label"].astype(str).map(self.vocab).to_numpy(dtype="float32")
        s = pd.to_numeric(df["score"], errors="coerce")
        if self.kind == "ordinal":
            return s.round().map(self.ranks).to_numpy(dtype="float32")
        return s.to_numpy(dtype="float32")

    def to_metric_space(self, raw_pred: np.ndarray) -> np.ndarray:
        """Discrete model output → the metric space (score / code / float)."""
        if self.kind == "ordinal":
            top = max(self.score_of_rank)
            return np.array([self.score_of_rank.get(int(r), top) for r in raw_pred], dtype=float)
        return np.asarray(raw_pred, dtype=float)


def _device(cfg: NeuralCfg) -> torch.device:
    if cfg.device == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _subsample(df: pd.DataFrame, cap: int | None, seed: int) -> pd.DataFrame:
    if cap is None or len(df) <= cap:
        return df
    return df.sample(n=cap, random_state=seed).reset_index(drop=True)


@torch.no_grad()
def _predict(model, loader, spec: TaskSpec, ls: LabelSpace, device) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(discrete_pred, continuous)`` — continuous is the regression value
    or the ordinal expected-rank / classification max-prob (used by the hybrid)."""
    model.eval()
    disc, cont = [], []
    for batch in loader:
        ids = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        tt = batch.get("token_type_ids")
        kw = {"token_type_ids": tt.to(device)} if tt is not None else {}
        z = model(ids, mask, **kw)
        if spec.task_type == "regression":
            disc.append(z.cpu().numpy()); cont.append(z.cpu().numpy())
        elif spec.task_type == "classification":
            disc.append(z.argmax(1).cpu().numpy())
            cont.append(torch.softmax(z, 1).max(1).values.cpu().numpy())
        else:  # ordinal
            disc.append(ordinal_predict(model.ordinal_head, z).cpu().numpy())
            cont.append(torch.sigmoid(z).sum(1).cpu().numpy())  # expected rank
    return np.concatenate(disc), np.concatenate(cont)


def fit_predict(train_df: pd.DataFrame, test_df: pd.DataFrame, spec: TaskSpec,
                cfg: NeuralCfg, seed: int, *, max_len: int,
                dev_df: pd.DataFrame | None = None,
                tokenizer=None) -> dict:
    """Fine-tune one cross-encoder and predict ``test_df``.

    Returns ``{y_true, y_pred, y_cont, num_classes}`` — ``y_pred`` in metric space.
    """
    set_global_seed(seed)
    torch.manual_seed(seed)
    device = _device(cfg)
    tok = tokenizer or AutoTokenizer.from_pretrained(cfg.backbone)

    train_df = _subsample(train_df, cfg.max_train_rows, seed)
    ls = LabelSpace(spec, train_df)
    if spec.task_type in ("ordinal", "classification") and ls.num_classes < 2:
        # degenerate fold (one class) — predict the constant, skip training
        const = ls.targets(train_df)
        y_true = ls.to_metric_space(ls.targets(test_df)) if spec.task_type == "ordinal" \
            else ls.targets(test_df)
        fill = float(np.nanmedian(const)) if const.size else 0.0
        pred = np.full(len(test_df), fill)
        return {"y_true": y_true, "y_pred": ls.to_metric_space(pred) if spec.task_type == "ordinal" else pred,
                "y_cont": pred, "num_classes": ls.num_classes}

    model = CrossEncoderGrader(
        cfg.backbone, spec.task_type, num_classes=ls.num_classes,
        dropout=cfg.dropout, pooling=cfg.pooling, ordinal_head=cfg.ordinal_head,
        freeze_backbone=cfg.freeze_backbone, lora_enabled=cfg.lora_enabled,
        lora_r=cfg.lora_r, lora_alpha=cfg.lora_alpha, lora_dropout=cfg.lora_dropout,
    ).to(device)

    tr_ds = PairDataset(train_df, ls.targets(train_df), tok, max_len)
    tr_loader = DataLoader(tr_ds, batch_size=cfg.batch_size, shuffle=True,
                           num_workers=cfg.num_workers)
    te_ds = PairDataset(test_df, np.zeros(len(test_df), "float32"), tok, max_len)
    te_loader = DataLoader(te_ds, batch_size=cfg.batch_size, num_workers=cfg.num_workers)

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    steps = max(1, len(tr_loader) // cfg.grad_accum) * cfg.epochs
    sched = get_linear_schedule_with_warmup(opt, int(cfg.warmup_ratio * steps), steps)
    mse = torch.nn.MSELoss()
    ce = torch.nn.CrossEntropyLoss()

    dev_loader = None
    if cfg.select_on_dev and dev_df is not None and len(dev_df):
        dev_ds = PairDataset(dev_df, np.zeros(len(dev_df), "float32"), tok, max_len)
        dev_loader = DataLoader(dev_ds, batch_size=cfg.batch_size, num_workers=cfg.num_workers)
        dev_true = ls.to_metric_space(ls.targets(dev_df)) if spec.task_type == "ordinal" \
            else ls.targets(dev_df)

    best_state, best_metric = None, -np.inf
    for epoch in range(cfg.epochs):
        model.train()
        opt.zero_grad()
        for step, batch in enumerate(tr_loader):
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            tt = batch.get("token_type_ids")
            kw = {"token_type_ids": tt.to(device)} if tt is not None else {}
            y = batch["target"].to(device)
            z = model(ids, mask, **kw)
            if spec.task_type == "regression":
                loss = mse(z, y)
            elif spec.task_type == "classification":
                loss = ce(z, y.long())
            else:
                loss = ordinal_loss(cfg.ordinal_head, z, y.long(), ls.num_classes)
            (loss / cfg.grad_accum).backward()
            if (step + 1) % cfg.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step(); sched.step(); opt.zero_grad()

        if dev_loader is not None:
            disc, _ = _predict(model, dev_loader, spec, ls, device)
            metric = compute_metrics(dev_true, ls.to_metric_space(disc) if spec.task_type == "ordinal"
                                     else disc, (spec.headline,))[spec.headline]
            metric = -1e9 if not np.isfinite(metric) else metric
            log.info(f"    epoch {epoch+1}/{cfg.epochs} dev {spec.headline}={metric:.4f}")
            if metric > best_metric:
                best_metric = metric
                best_state = copy.deepcopy({k: v.cpu() for k, v in model.state_dict().items()})

    if best_state is not None:
        model.load_state_dict(best_state)

    disc, cont = _predict(model, te_loader, spec, ls, device)
    y_pred = ls.to_metric_space(disc)
    y_true = ls.to_metric_space(ls.targets(test_df)) if spec.task_type == "ordinal" \
        else ls.targets(test_df)
    del model
    return {"y_true": np.asarray(y_true, dtype=float), "y_pred": np.asarray(y_pred, dtype=float),
            "y_cont": np.asarray(cont, dtype=float), "num_classes": ls.num_classes}

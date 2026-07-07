"""Zero-shot LLM grading baseline under the paper's own metrics/protocol.

Grades each dataset's **headline split** with an instruction-tuned LLM (temperature
0, prompts logged), then scores it with the *same* metric the GBM/DeBERTa arms use
(``asag.models.metrics`` via ``asag.models.tasks.REGISTRY``). This is the LLM row
reviewers expect in 2026; the project's own review simulation flags its absence as a
near-reject at Q2 (Reviewer C).

Design choices (all deliberate, all documented so a reviewer can check):

* **Zero-shot, no training, no folds.** ``official_split`` datasets are graded on the
  last / hardest test split (SemEval ``test_ud``, SAF ``test_uq``, ASAP-SAS
  ``test_ua``); ``kfold`` datasets (Mohler, Powergrading, MIND-CA) are graded once on
  all CV rows — there is nothing to fit, so the fold structure is irrelevant.
* **Task-aware prompt + metric.** classification → pick one label (macro-F1 on shared
  codes); ordinal → an integer in the observed range (QWK); regression → a number in
  range (Pearson). Per-prompt datasets (ASAP-SAS) average the metric across prompts,
  exactly like the GBM headline.
* **Cost cap.** ``LLM_MAX_ROWS`` (default 400) subsamples large test splits with a
  fixed seed; ``n_eval`` is reported so the estimate's basis is explicit.

Swap ``LLM_MODEL`` for a stronger open model (7B/8B) or wire an API inside
``generate()`` for the headline row; the metric plumbing is unchanged.

    python experiments/llm_zeroshot_baseline.py [dataset ...]
    LLM_MODEL=Qwen/Qwen2.5-3B-Instruct LLM_MAX_ROWS=400 python experiments/llm_zeroshot_baseline.py

Outputs: reports/phase_llm/{llm_baseline.json, examples.json}.
"""
from __future__ import annotations

import json
import os
import re
import sys

import numpy as np
import pandas as pd

from asag.config import load_data_config
from asag.models.metrics import compute_metrics
from asag.models.tasks import REGISTRY, get_spec

MODEL = os.environ.get("LLM_MODEL", "Qwen/Qwen2.5-3B-Instruct")
MAX_ROWS = int(os.environ.get("LLM_MAX_ROWS", "400"))          # cap eval rows/dataset (0 = all)
MAX_PER_PROMPT = int(os.environ.get("LLM_MAX_PER_PROMPT", "150"))
BATCH = int(os.environ.get("LLM_BATCH", "16"))
SEED = 42


def _eval_rows(df: pd.DataFrame, spec) -> pd.DataFrame:
    """Headline-split rows for official_split; all CV rows for kfold. Cost-capped."""
    if spec.protocol == "kfold":
        mask = pd.to_numeric(df["fold"], errors="coerce").fillna(-1) >= 0
    else:
        mask = df["split"].astype(str) == spec.test_splits[-1]
    d = df[mask].reset_index(drop=True)
    if spec.per_prompt and MAX_PER_PROMPT:
        d = (d.groupby("question_id", group_keys=False)
               .apply(lambda g: g.sample(n=min(len(g), MAX_PER_PROMPT), random_state=SEED))
               .reset_index(drop=True))
    elif MAX_ROWS and len(d) > MAX_ROWS:
        d = d.sample(n=MAX_ROWS, random_state=SEED).reset_index(drop=True)
    return d


def _label_space(df: pd.DataFrame, spec) -> dict:
    """Allowed labels (classification) or numeric range (ordinal/regression)."""
    if spec.task_type == "classification":
        labs = sorted(s for s in df["label"].astype(str).unique() if s and s.lower() != "nan")
        return {"labels": labs, "vocab": {lab: i for i, lab in enumerate(labs)}}
    s = pd.to_numeric(df["score"], errors="coerce").dropna()
    return {"lo": float(s.min()), "hi": float(s.max())}


def _prompt(row: pd.Series, spec, space: dict) -> str:
    q = str(row.get("question_enc") or "").strip()
    ref = str(row.get("reference_answer_enc") or "").strip()
    ans = str(row.get("student_answer_enc") or "").strip()
    ctx = ""
    if q and q.lower() != "nan":
        ctx += f"Question: {q}\n"
    if ref and ref.lower() != "nan":
        ctx += f"Reference answer: {ref}\n"
    if spec.task_type == "classification":
        task = (f"Classify the student answer into exactly one of these labels: "
                f"{', '.join(space['labels'])}.\nReply with ONLY the label, nothing else.")
    elif spec.task_type == "ordinal":
        task = (f"Grade the student answer with an INTEGER score from {int(space['lo'])} to "
                f"{int(space['hi'])} (higher = better). Reply with ONLY the integer.")
    else:
        task = (f"Grade the student answer with a score from {space['lo']:.2f} to "
                f"{space['hi']:.2f} (higher = better). Reply with ONLY the number.")
    return f"{ctx}Student answer: {ans}\n\n{task}"


def _parse(text: str, spec, space: dict) -> float:
    t = (text or "").strip()
    if spec.task_type == "classification":
        low = t.lower()
        for lab in space["labels"]:                       # first label mentioned wins
            if lab.lower() in low:
                return float(space["vocab"][lab])
        return float(space["vocab"][space["labels"][0]])  # fallback: first label
    m = re.search(r"-?\d+(?:\.\d+)?", t)
    mid = (space["lo"] + space["hi"]) / 2.0
    if not m:
        return round(mid) if spec.task_type == "ordinal" else mid
    v = max(space["lo"], min(space["hi"], float(m.group())))
    return float(round(v)) if spec.task_type == "ordinal" else v


def load_llm():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"                              # correct batched decoder-only gen
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=torch.float16,
        device_map="cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    return tok, model


def generate(tok, model, prompts: list[str]) -> list[str]:
    import torch

    sys_msg = "You are a strict grading assistant. Follow the requested output format exactly."
    texts = [tok.apply_chat_template(
        [{"role": "system", "content": sys_msg}, {"role": "user", "content": p}],
        tokenize=False, add_generation_prompt=True) for p in prompts]
    out: list[str] = []
    for i in range(0, len(texts), BATCH):
        chunk = texts[i:i + BATCH]
        enc = tok(chunk, return_tensors="pt", padding=True, truncation=True,
                  max_length=1024).to(model.device)
        with torch.no_grad():
            gen = model.generate(**enc, max_new_tokens=12, do_sample=False,   # temp 0 (greedy)
                                 pad_token_id=tok.pad_token_id)
        for j in range(len(chunk)):
            out.append(tok.decode(gen[j][enc["input_ids"].shape[1]:], skip_special_tokens=True))
        print(f"    generated {min(i + BATCH, len(texts))}/{len(texts)}", flush=True)
    return out


def score_dataset(name: str, cfg, tok, model, examples: dict) -> dict:
    spec = get_spec(name)
    path = cfg.paths.processed / name / "encoder.parquet"
    if not path.exists():
        return {"status": "no_encoder_parquet"}
    df = pd.read_parquet(path).reset_index(drop=True)
    space = _label_space(df, spec)
    d = _eval_rows(df, spec)
    if d.empty:
        return {"status": "no_eval_rows"}

    prompts = [_prompt(r, spec, space) for _, r in d.iterrows()]
    raw = generate(tok, model, prompts)
    y_pred = np.array([_parse(t, spec, space) for t in raw], dtype=float)
    if spec.task_type == "classification":
        y_true = d["label"].astype(str).map(space["vocab"]).to_numpy(dtype=float)
    else:
        y_true = pd.to_numeric(d["score"], errors="coerce").to_numpy(dtype=float)
    fin = np.isfinite(y_true) & np.isfinite(y_pred)
    examples[name] = [{"prompt": prompts[k], "llm_raw": raw[k], "pred": float(y_pred[k]),
                       "true": float(y_true[k])} for k in range(min(3, len(prompts)))]

    def metric(mask: np.ndarray) -> float:
        return compute_metrics(y_true[mask], y_pred[mask], (spec.headline,)).get(
            spec.headline, float("nan"))

    if spec.per_prompt:
        qid = d["question_id"].astype(str).to_numpy()
        vals = [v for p in np.unique(qid[fin]) if np.isfinite(v := metric(fin & (qid == p)))]
        headline = float(np.mean(vals)) if vals else float("nan")
    else:
        headline = metric(fin)
    return {"status": "ok", "metric": spec.headline,
            "headline_split": spec.test_splits[-1] if spec.protocol == "official_split" else "cv",
            "n_eval": int(fin.sum()),
            "llm_headline": None if not np.isfinite(headline) else round(float(headline), 4),
            "model": MODEL}


def main(names: list[str]) -> None:
    cfg = load_data_config()
    names = names or [n for n in REGISTRY if (cfg.paths.processed / n / "encoder.parquet").exists()]
    tok, model = load_llm()
    out, examples = {}, {}
    for n in names:
        print(f"=== LLM zero-shot: {n} ({MODEL}) ===", flush=True)
        out[n] = score_dataset(n, cfg, tok, model, examples)
        print(f"    {n}: {out[n]}", flush=True)
    d = cfg.paths.reports / "phase_llm"
    d.mkdir(parents=True, exist_ok=True)
    (d / "llm_baseline.json").write_text(json.dumps(
        {"model": MODEL, "max_rows": MAX_ROWS, "temperature": 0.0, "datasets": out},
        indent=2, default=str), encoding="utf-8")
    (d / "examples.json").write_text(json.dumps(examples, indent=2, default=str), encoding="utf-8")
    print(f"wrote {d / 'llm_baseline.json'}")


if __name__ == "__main__":
    main([a for a in sys.argv[1:] if not a.startswith("-")])

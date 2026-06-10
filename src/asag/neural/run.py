"""Phase 2G CLI — fine-tune the DeBERTa cross-encoder across datasets.

    python -m asag.neural.run [<name>...]     # no args = all datasets with encoder.parquet

Writes ``reports/phase2g/{results.json,results.csv}`` (+ per-item prediction caches
under ``preds/`` and a neural-vs-GBM headline figure). CPU-bound — start with the
small datasets (``mohler``) and expand. Reads the GBM headline from
``reports/phase2c/results.json`` when present to draw the side-by-side bar.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from asag.config import DataConfig, ensure_dirs, load_data_config
from asag.models.tasks import REGISTRY
from asag.neural import (NEURAL_SCHEMA_VERSION, TORCH_AVAILABLE,
                         TRANSFORMERS_AVAILABLE)
from asag.neural.evaluate_neural import evaluate_neural
from asag.utils.logging import get_logger
from asag.utils.seed import set_global_seed

log = get_logger()


def _save_preds(cfg: DataConfig, name: str, cache: dict) -> None:
    if not cfg.neural.save_predictions or not cache:
        return
    out = cfg.paths.reports / "phase2g" / "preds"
    out.mkdir(parents=True, exist_ok=True)
    payload = {}
    for split, items in cache.items():
        payload[split] = {k: np.asarray(v).tolist() for k, v in items.items()}
    (out / f"{name}.json").write_text(json.dumps(payload, default=str), encoding="utf-8")


def _flatten(name: str, res: dict) -> list[dict]:
    rows = []
    for split, ev in res["evaluations"].items():
        for model in ("neural", "baseline"):
            for metric, stats in ev.get(model, {}).items():
                rows.append({"dataset": name, "split": split, "model": model,
                             "metric": metric, "mean": round(stats.get("mean", float("nan")), 4),
                             "std": round(stats.get("std", float("nan")), 4)})
    return rows


def _gbm_headline(cfg: DataConfig) -> dict:
    p = cfg.paths.reports / "phase2c" / "results.json"
    if not p.exists():
        return {}
    doc = json.loads(p.read_text(encoding="utf-8"))
    return {n: r.get("headline", {}) for n, r in doc.get("datasets", {}).items()}


def _write_figure(cfg: DataConfig, results: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    gbm = _gbm_headline(cfg)
    names = list(results.keys())
    neural = [results[n]["headline"]["neural"].get("mean", np.nan) for n in names]
    nstd = [results[n]["headline"]["neural"].get("std", 0.0) or 0.0 for n in names]
    gbm_m = [gbm.get(n, {}).get("gbm", {}).get("mean", np.nan) for n in names]
    base = [results[n]["headline"]["baseline"].get("mean", np.nan) for n in names]
    labels = [f"{n}\n({results[n]['headline']['metric']}@{results[n]['headline']['split']})" for n in names]

    x = np.arange(len(names)); w = 0.27
    fig, ax = plt.subplots(figsize=(max(8, 1.9 * len(names)), 5))
    ax.bar(x - w, neural, w, yerr=nstd, capsize=3, label="DeBERTa cross-encoder", color="#6a51a3")
    ax.bar(x, gbm_m, w, label="GBM fusion head", color="#2c7fb8")
    ax.bar(x + w, base, w, label="naive baseline", color="#bdbdbd")
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("headline metric")
    ax.set_title("Phase 2G — DeBERTa cross-encoder vs GBM fusion head")
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(cfg.paths.figures / "phase2g_neural_vs_gbm.png", dpi=120); plt.close(fig)
    log.info(f"wrote {cfg.paths.figures / 'phase2g_neural_vs_gbm.png'}")


def run_neural(cfg: DataConfig | None = None, only: list[str] | None = None) -> dict:
    cfg = cfg or load_data_config()
    ensure_dirs(cfg)
    set_global_seed(cfg.seed)
    if not (TORCH_AVAILABLE and TRANSFORMERS_AVAILABLE):
        raise RuntimeError("torch + transformers are required for the neural slice")
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(cfg.neural.backbone)
    names = only or [n for n in REGISTRY if (cfg.paths.processed / n / "encoder.parquet").exists()]
    results: dict = {}
    for name in names:
        log.info(f"=== neural: {name} ===")
        rc = evaluate_neural(name, cfg, tok)
        if rc is None:
            continue
        res, cache = rc
        results[name] = res
        _save_preds(cfg, name, cache)

    if results:
        out_dir = cfg.paths.reports / "phase2g"
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "results.json").write_text(json.dumps(
            {"schema_version": NEURAL_SCHEMA_VERSION, "backbone": cfg.neural.backbone,
             "seeds": list(cfg.neural.seeds), "datasets": results}, indent=2, default=str),
            encoding="utf-8")
        rows = [r for n, res in results.items() for r in _flatten(n, res)]
        pd.DataFrame(rows).to_csv(out_dir / "results.csv", index=False)
        _write_figure(cfg, results)
        log.info(f"wrote {out_dir}/results.json, results.csv")
    return results


if __name__ == "__main__":
    import sys

    run_neural(only=sys.argv[1:] or None)

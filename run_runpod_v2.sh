#!/usr/bin/env bash
# =============================================================================
# RunPod one-shot: full neural + hybrid run -> all paper artifacts.
# Target: RTX 4090 (24GB), On-Demand, template "RunPod PyTorch 2.x".
#
# USAGE (in the pod's JupyterLab terminal / Web Terminal):
#   1) upload Explainable-Hybrid-ASAG.zip AND this file to /workspace
#   2) run inside tmux so a terminal drop never kills the job:
#        tmux new -s asag
#        bash /workspace/run_runpod.sh 2>&1 | tee /workspace/run.log
#      reattach anytime with:  tmux attach -t asag
#
# Robust by design: each dataset is a SEPARATE process (GPU freed between datasets,
# like the project's notebooks), resumable (finished datasets are skipped), and one
# dataset failing never kills the others. The smoke (mohler) is a hard gate.
# =============================================================================
set -uo pipefail          # NOT -e: we handle failures per-stage so the run always reaches the bundle

# ---- 0) locate + unzip the project -----------------------------------------
command -v unzip >/dev/null 2>&1 || (apt-get update -qq && apt-get install -y -qq unzip)
cd /workspace
ZIP=$(ls /workspace/Explainable-Hybrid-ASAG*.zip 2>/dev/null | head -1)
[ -n "$ZIP" ] || { echo "ERROR: upload Explainable-Hybrid-ASAG.zip to /workspace first"; exit 1; }
rm -rf /workspace/proj && unzip -q -o "$ZIP" -d /workspace/proj
REPO=$(dirname "$(find /workspace/proj -name pyproject.toml | head -1)")
[ -f "$REPO/configs/data.yaml" ] || { echo "ERROR: repo root (configs/data.yaml) not found"; exit 1; }
cd "$REPO"
export ASAG_PROJECT_ROOT="$REPO" PYTHONPATH="$REPO/src" PYTHONUTF8=1 PYTHONIOENCODING=utf-8 TOKENIZERS_PARALLELISM=false
echo "REPO = $REPO"

# ---- 1) deps (torch ships with the template - we do NOT touch it) ----------
pip install -q transformers peft sentence-transformers sentencepiece protobuf accelerate spacy \
    "lightgbm==4.5.0" "optuna==4.1.0" loguru pandas numpy pyarrow scikit-learn scipy matplotlib \
    pydantic pyyaml tqdm huggingface-hub || { echo "ERROR: pip install failed - fix the environment first"; exit 1; }
python -m spacy download en_core_web_sm
pip uninstall -y torchao 2>/dev/null || true   # stale torchao hard-breaks fresh peft's LoRA

# ---- 2) config: GPU, 1 seed (time), lr 1e-4 + 8 epochs + LoRA r16 (credible neural), batch 32 --
python - <<'PY'
import yaml, os
p = os.path.join(os.environ["ASAG_PROJECT_ROOT"], "configs", "data.yaml")
d = yaml.safe_load(open(p))
d["neural"].update({"device": "cuda", "seeds": [42], "batch_size": 32, "num_workers": 0, "lr": 1.0e-4, "epochs": 8, "lora_r": 16, "lora_alpha": 32})
yaml.safe_dump(d, open(p, "w"), sort_keys=False, allow_unicode=True)
print("neural:", {k: d["neural"][k] for k in ("device", "seeds", "batch_size", "num_workers", "epochs")})
PY

DATASETS="mohler powergrading mindreading saf asap_sas semeval"   # small -> large
mkdir -p reports/phase2g/_partial

cache_block () {   # $1=dataset : copy that dataset's block out of the (overwritten) results.json
  python - "$1" <<'PY' || true
import json, sys, os
ds = sys.argv[1]; rj = "reports/phase2g/results.json"
if os.path.exists(rj):
    b = json.load(open(rj)).get("datasets", {}).get(ds)
    if b:
        json.dump(b, open(f"reports/phase2g/_partial/{ds}.json", "w"), default=str)
        print("  cached", ds)
PY
}

# ---- 3) SMOKE = mohler is a hard gate (proves torch+transformers+peft+deberta+CORN) ----
echo "=== SMOKE / Phase 2G: mohler ==="
python -m asag.neural.run mohler || { echo "SMOKE FAILED - fix the error above before continuing"; exit 1; }
cache_block mohler

# ---- 4) Phase 2G for the rest: one process per dataset (GPU freed between), resumable ----
for ds in $DATASETS; do
  [ -f "reports/phase2g/_partial/$ds.json" ] && { echo "skip (done): $ds"; continue; }
  if python -m asag.neural.run "$ds"; then cache_block "$ds"; else echo "WARN: Phase 2G $ds failed (continuing)"; fi
done
# merge per-dataset caches -> one complete reports/phase2g/results.json (+ figure)
python - <<'PY' || true
import json, glob, os
from asag.config import load_data_config
from asag.neural.run import _write_figure
c = load_data_config()
parts = {os.path.basename(x)[:-5]: json.load(open(x)) for x in glob.glob("reports/phase2g/_partial/*.json")}
if parts:
    json.dump({"backbone": c.neural.backbone, "seeds": list(c.neural.seeds), "datasets": parts},
              open("reports/phase2g/results.json", "w"), indent=2, default=str)
    try:
        _write_figure(c, parts)
    except Exception as e:
        print("figure skipped:", e)
    print("Phase 2G merged:", sorted(parts))
PY

# ---- 5) Phase 4 hybrid OOF: one process per dataset, resumable ----
for ds in $DATASETS; do
  [ -f "data/processed/$ds/neural_oof.parquet" ] && { echo "skip (done): $ds"; continue; }
  echo "=== OOF: $ds ==="
  python -m asag.neural.extract_features "$ds" || echo "WARN: OOF $ds failed (continuing)"
done

# ---- 6) downstream GBM-with-neural (each guarded so we always reach the bundle) ----
for stage in asag.models.ablations asag.models.neural_compare asag.models.train2d asag.xai.run; do
  echo "=== $stage ==="
  python -m "$stage" || echo "WARN: $stage failed (continuing)"
done

# ---- 6b) fix a pandas-2.2 groupby bug in the LLM script before running it ----
python - <<'PYX' || true
p="experiments/llm_zeroshot_baseline.py"; s=open(p).read()
old="""        d = (d.groupby("question_id", group_keys=False)
               .apply(lambda g: g.sample(n=min(len(g), MAX_PER_PROMPT), random_state=SEED))
               .reset_index(drop=True))"""
new="""        d = pd.concat([g.sample(n=min(len(g), MAX_PER_PROMPT), random_state=SEED)
                       for _, g in d.groupby("question_id")]).reset_index(drop=True)"""
if old in s:
    open(p,"w").write(s.replace(old,new)); print("LLM script patched (pandas fix)")
else:
    print("LLM anchor not found (already patched or changed)")
PYX

# ---- 7) LLM zero-shot baseline (non-fatal) ----
echo "=== LLM baseline (Qwen-3B) ==="
LLM_MODEL=Qwen/Qwen2.5-3B-Instruct LLM_MAX_ROWS=400 python experiments/llm_zeroshot_baseline.py \
  || echo "WARN: LLM baseline failed (non-fatal)"

# ---- 8) verify + bundle everything for download ----
python paper/verify_numbers.py || true
cd "$REPO" && zip -q -r /workspace/results_bundle.zip \
    data/processed/*/neural_oof.parquet reports/phase2g reports/phase_hybrid reports/phase_llm \
    reports/phase2c reports/phase3 reports/phase2d reports/phase2f \
    reports/phase4_audit reports/phase4_robust reports/phase4_lodo \
    reports/figures/*.png paper 2>/dev/null || true

# ---- 9) readable results summary (read the numbers immediately) ----
echo ""; echo "================= RESULTS SUMMARY ================="
python - <<'PY' || true
import json
def load(p):
    try: return json.load(open(p))
    except Exception: return {}
p2g = load("reports/phase2g/results.json").get("datasets", {})
tw  = load("reports/phase_hybrid/three_way.json").get("datasets", {})
llm = load("reports/phase_llm/llm_baseline.json").get("datasets", {})
d2d = load("reports/phase2d/results.json").get("datasets", {})
names = sorted(set(p2g) | set(tw) | set(d2d))
f = lambda x: "  -  " if x is None else ("%.4f" % x)
g = lambda x: "  -  " if x is None else ("%+.3f" % x)
print("%-13s%-9s%9s%9s%9s%8s%9s%8s" % ("dataset","metric","neural","feature","hybrid","gain","tuned","LLM"))
for n in names:
    h = p2g.get(n, {}).get("headline", {})
    t = tw.get(n, {})
    print("%-13s%-9s%9s%9s%9s%8s%9s%8s" % (
        n, str(h.get("metric","")),
        f(h.get("neural",{}).get("mean")),
        f((t.get("feature_only") or {}).get("mean")),
        f((t.get("hybrid") or {}).get("mean")),
        g(t.get("fusion_gain")),
        f(d2d.get(n,{}).get("headline",{}).get("gbm",{}).get("mean")),
        f(llm.get(n,{}).get("llm_headline"))))
PY
echo "==================================================="
echo "======================================================================"
echo "DONE. Download /workspace/results_bundle.zip, then STOP the pod."
echo "Completed neural datasets:"; ls reports/phase2g/_partial/ 2>/dev/null | sed 's/.json//' | sed 's/^/  - /'
echo "======================================================================"

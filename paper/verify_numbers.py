#!/usr/bin/env python3
"""Cross-check every number quoted in paper/main.tex against reports/**.

Run from the repo root:  python paper/verify_numbers.py
Exits non-zero if any check fails. Re-run after regenerating reports (e.g.,
after the hybrid Colab run) to catch silent drift between reports and paper.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "reports"

FAIL: list[str] = []
N = 0


def load(rel: str):
    return json.loads((R / rel).read_text(encoding="utf-8"))


def check(name: str, actual: float | None, expected: float, tol: float = 5e-4) -> None:
    global N
    N += 1
    if actual is None or abs(actual - expected) > tol:
        FAIL.append(f"{name}: paper says {expected}, report says {actual}")


# ---------------- Table: leakage audit (phase4_audit/leakage_audit.json) ----
a = load("phase4_audit/leakage_audit.json")["datasets"]
check("mohler strat", a["mohler"]["gbm_stratified"], 0.5698, 5e-4)
check("mohler grouped", a["mohler"]["gbm_grouped"], 0.4297)
check("mohler qshort strat", a["mohler"]["qshortcut_stratified"], 0.3646)
check("mohler qshort grouped", a["mohler"]["qshortcut_grouped"], -0.1663)
check("mohler eta2", a["mohler"]["between_question_eta2"], 0.1857, 5e-3)
check("pg strat", a["powergrading"]["gbm_stratified"], 0.8295)
check("pg grouped", a["powergrading"]["gbm_grouped"], 0.45)
check("pg qshort strat", a["powergrading"]["qshortcut_stratified"], 0.8733)
check("pg qshort grouped", a["powergrading"]["qshortcut_grouped"], 0.3923)
check("pg eta2", a["powergrading"]["between_question_eta2"], 0.9151, 5e-3)
check("mind strat", a["mindreading"]["gbm_stratified"], 0.1206)
check("mind grouped", a["mindreading"]["gbm_grouped"], 0.0594)
check("mind qshort strat", a["mindreading"]["qshortcut_stratified"], 0.1636)
check("mind qshort grouped", a["mindreading"]["qshortcut_grouped"], 0.0)
check("mind eta2", a["mindreading"]["between_question_eta2"], 0.1413, 5e-3)
check("saf eta2", a["saf"]["between_question_eta2"], 0.702, 5e-3)
check("asap len-shortcut r", a["asap_sas"]["length_vs_score_pearson"], 0.5887, 5e-3)

g = load("phase4_audit/grouped_cv_audit.json")["datasets"]
check("regularized-head inflation mohler", g["mohler"]["gbm_inflation"], 0.1739, 5e-3)
check("regularized-head inflation pg", g["powergrading"]["gbm_inflation"], 0.3506, 5e-3)
check("regularized-head inflation mind", g["mindreading"]["gbm_inflation"], 0.0778, 5e-3)

# ---------------- Table: main results (phase2d/results.json) ----------------
d = load("phase2d/results.json")["datasets"]


def hl(ds, arm):
    return d[ds]["headline"][arm]["mean"]


def hlstd(ds, arm):
    return d[ds]["headline"][arm]["std"]


check("semeval tuned", hl("semeval", "gbm"), 0.4269, 1e-3)
check("semeval tuned std", hlstd("semeval", "gbm"), 0.005, 1e-3)
check("semeval base", hl("semeval", "baseline"), 0.1184, 1e-3)
check("asap tuned", hl("asap_sas", "gbm"), 0.363, 1e-3)
check("mohler tuned", hl("mohler", "gbm"), 0.4904, 1e-3)
check("mohler qshort", hl("mohler", "question_shortcut"), -0.1663, 1e-3)
check("saf tuned (test_uq)", hl("saf", "gbm"), 0.0193, 1e-3)
check("saf tuned std", hlstd("saf", "gbm"), 0.0328, 1e-3)
check("pg tuned", hl("powergrading", "gbm"), 0.4862, 1e-3)
check("pg base", hl("powergrading", "baseline"), 0.3923, 1e-3)
check("mind tuned", hl("mindreading", "gbm"), 0.077, 1e-3)

# official-split corroboration numbers quoted in Sec 7.1
ev = d["semeval"]["evaluations"]
check("semeval UA gbm", ev["test_ua"]["gbm"]["macro_f1"]["mean"], 0.5077, 1e-3)
check("semeval UQ gbm", ev["test_uq"]["gbm"]["macro_f1"]["mean"], 0.4527, 1e-3)
check("semeval UA qshort", ev["test_ua"]["question_shortcut"]["macro_f1"]["mean"], 0.3233, 1e-3)
sev = d["saf"]["evaluations"]
check("saf UA pearson", sev["test_ua"]["gbm"]["pearson"]["mean"], 0.9039, 1e-3)
check("saf UA qshort pearson", sev["test_ua"]["question_shortcut"]["pearson"]["mean"], 0.8913, 1e-3)

# 2C default-head numbers quoted in Table main
c = load("phase2c/results.json")["datasets"]
check("semeval 2C", c["semeval"]["headline"]["gbm"]["mean"], 0.4111, 1e-3)
check("asap 2C", c["asap_sas"]["headline"]["gbm"]["mean"], 0.3497, 1e-3)
check("mohler 2C", c["mohler"]["headline"]["gbm"]["mean"], 0.4297, 1e-3)
check("saf 2C", c["saf"]["headline"]["gbm"]["mean"], 0.0261, 1e-3)
check("pg 2C", c["powergrading"]["headline"]["gbm"]["mean"], 0.45, 1e-3)
check("mind 2C", c["mindreading"]["headline"]["gbm"]["mean"], 0.0594, 1e-3)

# ---------------- Table: significance (phase2d/significance.json) -----------
s = load("phase2d/significance.json")["datasets"]
for ds, delta, lo, hi, ph in [
    ("semeval", 0.3148, 0.2565, 0.3605, 0.0),
    ("asap_sas", 0.3598, 0.3176, 0.3976, 0.0),
    ("mohler", 0.6575, 0.4626, 0.8638, 0.0),
    ("mindreading", 0.0729, 0.017, 0.1355, 0.0111),
    ("powergrading", 0.0787, -0.0693, 0.2016, 0.2912),
    ("saf", -0.0184, -0.0724, 0.1305, 0.5815),
]:
    check(f"sig {ds} delta", s[ds]["delta_observed"], delta, 1e-3)
    check(f"sig {ds} ci_lo", s[ds]["ci_lo"], lo, 1e-3)
    check(f"sig {ds} ci_hi", s[ds]["ci_hi"], hi, 1e-3)
    check(f"sig {ds} p_holm", s[ds]["p_holm"], ph, 1e-3)

# ---------------- Table: ablations (phase3/ablations.json) ------------------
ab = load("phase3/ablations.json")["datasets"]


def delta(ds, var):
    return ab[ds]["variants"][var]["delta_vs_full"]


def full(ds):
    return ab[ds]["variants"]["full"]["headline"]["mean"]


check("abl semeval full", full("semeval"), 0.4151, 1e-3)
check("abl semeval -A", delta("semeval", "-A"), -0.0139, 1e-3)
check("abl semeval -B", delta("semeval", "-B"), -0.1087, 1e-3)
check("abl semeval -C", delta("semeval", "-C"), -0.0028, 1e-3)
check("abl semeval -neg", delta("semeval", "-neg"), -0.0288, 1e-3)
check("abl asap full", full("asap_sas"), 0.34, 1e-3)
check("abl asap -B", delta("asap_sas", "-B"), -0.34, 1e-3)
check("abl asap -neg", delta("asap_sas", "-neg"), -0.0205, 1e-3)
check("abl mohler full", full("mohler"), 0.4392, 1e-3)
check("abl mohler -B", delta("mohler", "-B"), -0.068, 1e-3)
check("abl mohler -C", delta("mohler", "-C"), 0.0076, 1e-3)
check("abl pg full", full("powergrading"), 0.4728, 1e-3)
check("abl pg -B", delta("powergrading", "-B"), 0.0087, 1e-3)
check("abl pg only-C", delta("powergrading", "only-C"), 0.0139, 1e-3)
check("abl mind full", full("mindreading"), 0.0626, 1e-3)
check("abl mind -neg", delta("mindreading", "-neg"), 0.0061, 1e-3)
check("abl saf full", full("saf"), 0.0238, 1e-3)
check("abl saf only-C", delta("saf", "only-C"), -0.1921, 1e-3)

# ---------------- SAF validation (phase2f/saf_validation.json) --------------
sv = load("phase2f/saf_validation.json")["result"]
assert sv["n"] == 854, "SAF n mismatch"
sig4 = sv["signals"]
check("saf cov Incorrect", sig4["rub_coverage_at_tau"]["by_class"]["Incorrect"], 0.2242, 1e-3)
check("saf cov Correct", sig4["rub_coverage_at_tau"]["by_class"]["Correct"], 0.547, 1e-3)
check("saf sem rho", sig4["sem_cosine"]["spearman_vs_verdict"], 0.255, 1e-3)
check("saf sem auc", sig4["sem_cosine"]["auc_correct_vs_rest"], 0.621, 1e-3)
for k in sig4:
    assert sig4[k]["monotonic"], f"SAF signal {k} not monotonic"

# ---------------- Calibration / robustness ----------------------------------
r = load("phase4_robust/robustness.json")["datasets"]
check("semeval ECE pre", r["semeval"]["calibration"]["ece_pre"], 0.1224, 1e-3)
check("semeval ECE post", r["semeval"]["calibration"]["ece_post"], 0.0592, 1e-3)
check("semeval T", r["semeval"]["calibration"]["temperature"], 1.35, 1e-2)
check("pg ECE pre", r["powergrading"]["calibration"]["ece_pre"], 0.6134, 1e-3)
check("pg ECE post", r["powergrading"]["calibration"]["ece_post"], 0.3337, 1e-3)
check("pg T", r["powergrading"]["calibration"]["temperature"], 5.95, 1e-2)
rc = {p["coverage"]: p["accuracy"] for p in r["semeval"]["calibration"]["risk_coverage"]}
check("risk-cov 1.0", rc[1.0], 0.4671, 1e-3)
check("risk-cov 0.8", rc[0.8], 0.5062, 1e-3)
check("risk-cov 0.5", rc[0.5], 0.5747, 1e-3)
pert = r["semeval"]["perturbations"]["perturbations"]
check("semeval negflip frac", pert["negation_flip"]["frac_changed"], 0.4867, 1e-3)
check("semeval negflip absdelta", pert["negation_flip"]["mean_abs_delta"], 0.8, 1e-3)
check("semeval parap frac", pert["paraphrase_drop"]["frac_changed"], 0.1833, 1e-3)
check("semeval lenpad frac", pert["length_pad"]["frac_changed"], 0.3167, 1e-3)
check("mohler parap signed", r["mohler"]["perturbations"]["perturbations"]["paraphrase_drop"]["mean_signed_delta"], -0.2426, 1e-3)

# ---------------- LODO -------------------------------------------------------
lo = load("phase4_lodo/lodo.json")["results"]
check("lodo semeval f1", lo["semeval"]["macro_f1_mean"], 0.6051, 1e-3)
check("lodo semeval auc", lo["semeval"]["auc_mean"], 0.676, 1e-3)
check("lodo pg f1", lo["powergrading"]["macro_f1_mean"], 0.6311, 1e-3)
check("lodo pg auc", lo["powergrading"]["auc_mean"], 0.7112, 1e-3)
check("lodo saf auc", lo["saf"]["auc_mean"], 0.5042, 1e-3)
check("lodo asap auc", lo["asap_sas"]["auc_mean"], 0.2694, 1e-3)

# ---------------- Ceiling ----------------------------------------------------
ce = load("phase2d/ceiling.json")["datasets"]["asap_sas"]
check("asap IAA macro QWK", ce["macro_qwk"], 0.9419, 1e-3)
check("asap IAA min prompt", ce["per_prompt"]["set_2"]["qwk"], 0.9143, 1e-3)
check("asap IAA max prompt", ce["per_prompt"]["set_6"]["qwk"], 0.9562, 1e-3)

# ---------------- Error analysis (SemEval, appendix) -------------------------
e = load("phase3/error_analysis.json")["datasets"]["semeval"]["gbm"]
check("semeval exact acc", e["exact"], 0.4649, 1e-3)
pc = e["per_class"]
check("F1 correct", pc["correct"]["f1"], 0.6326, 1e-3)
check("F1 partial", pc["partially_correct_incomplete"]["f1"], 0.2667, 1e-3)
check("F1 irrelevant", pc["irrelevant"]["f1"], 0.35, 1e-3)
check("F1 contradictory", pc["contradictory"]["f1"], 0.2333, 1e-3)
check("F1 non_domain", pc["non_domain"]["f1"], 0.625, 1e-3)
check("contradictory recall", pc["contradictory"]["recall"], 0.2134, 1e-3)
tc = {(t["true"], t["pred"]): t["count"] for t in e["top_confused"]}
assert tc.get(("partially_correct_incomplete", "correct")) == 524, "top confusion 1"
assert tc.get(("irrelevant", "correct")) == 490, "top confusion 2"

# ---------------- SHAP-vs-gain Spearman (phase2f/shap.json) ------------------
sh = load("phase2f/shap.json")
expected_rho = {"semeval": 0.87, "saf": 0.653, "asap_sas": 0.998,
                "mohler": 0.959, "powergrading": 0.88, "mindreading": 0.999}


def find_rho(node, out):
    if isinstance(node, dict):
        for k, v in node.items():
            if "spearman" in k.lower() and isinstance(v, (int, float)):
                out.append((k, v))
            else:
                find_rho(v, out)
    elif isinstance(node, list):
        for v in node:
            find_rho(v, out)


for ds, exp in expected_rho.items():
    found: list = []
    find_rho(sh.get("datasets", sh).get(ds, {}), found)
    vals = [v for _, v in found]
    if not any(abs(v - exp) <= 5e-3 for v in vals):
        FAIL.append(f"shap-gain rho {ds}: paper says {exp}, found {vals}")
    N += 1

# ---------------- Tuned hyperparameters (appendix) ---------------------------
tuned = {k: v["lightgbm_tuned"] for k, v in d.items()}
for ds, lr in [("semeval", 0.088), ("saf", 0.096), ("asap_sas", 0.023),
               ("mohler", 0.015), ("powergrading", 0.229), ("mindreading", 0.146)]:
    check(f"tuned lr {ds}", tuned[ds]["learning_rate"], lr, 1e-3)

# ---------------- Faithfulness (mohler branch summary) -----------------------
f = load("phase2f/faithfulness.json")["datasets"]["mohler"]["branches"]
check("faith rubric validity", f["rubric"]["mean_predictive_validity_rho"], 0.3352, 1e-3)
check("faith rubric share", f["rubric"]["share_global_abs_shap"], 0.2139, 1e-3)
check("faith linguistic share", f["linguistic"]["share_global_abs_shap"], 0.717, 1e-3)
check("faith linguistic validity", f["linguistic"]["mean_predictive_validity_rho"], 0.2, 1e-3)
check("faith semantic validity", f["semantic"]["mean_predictive_validity_rho"], 0.1421, 1e-3)

# ---------------- verdict ----------------------------------------------------
print(f"checks run: {N}")
if FAIL:
    print(f"FAILED: {len(FAIL)}")
    for x in FAIL:
        print("  -", x)
    sys.exit(1)
print("ALL PASS — every quoted number matches reports/**")

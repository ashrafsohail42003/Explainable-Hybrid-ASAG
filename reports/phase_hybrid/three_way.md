# Phase B — three-way comparison: neural-only / feature-only / hybrid

> One fixed regularized head for feature-only and hybrid (the Δ is the
> neural features). Tuned headline numbers live in `reports/phase2d/`.

| Dataset | Metric | Neural-only | Feature-only | Hybrid | Fusion gain | Interp. cost |
|---|---|---|---|---|---|---|
| semeval | macro_f1 | 0.1184 | 0.4151 | 0.4269 | 0.0118 | 0.3085 |
| saf | pearson | -0.0221 | 0.0238 | 0.0258 | 0.0020 | 0.0479 |
| asap_sas | qwk | 0.1957 | 0.3400 | 0.3835 | 0.0435 | 0.1878 |
| mohler | pearson | 0.0475 | 0.4392 | 0.4245 | -0.0147 | 0.3770 |
| powergrading | macro_f1 | 0.3923 | 0.4728 | 0.7040 | 0.2312 | 0.3117 |
| mindreading | qwk | 0.0000 | 0.0626 | 0.0173 | -0.0453 | 0.0173 |

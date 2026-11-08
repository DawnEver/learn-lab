---
created: 2026-11-08
---
# Bagging and hyper-parameter tuning moved in from optimi-lab's `pr/sensitivity-bagging-hpo`

- Source: optimi-lab `origin/pr/sensitivity-bagging-hpo` added `model/bagging.py`, `model/tuning.py` after A1
  moved every model here. The MODEL half lands here; the SEARCH half (`analysis/ranking.py` sensitivity ranking,
  `metric/statistics.py` Pearson) stays optimi-lab's and is not ported.
- `models/response/bagging.py`: `Bagging(member=RegressionTree(), n_members=10, max_samples=1.0, seed=None)`.
  Every member is fitted in the bag's ONE envelope (`domain=`, else the rows' span), so a resample missing an
  extreme row never narrows where the bag answers; `FittedBagging.predict` holds queries to it. Default member
  added (optimi-lab had none) so the bag is a registry row (`'bagging'`) constructible with no settings, and
  `tune` can grid its `member`; bagged trees is Breiman's default.
- `tuning.py` beside `validation.py`: `tune(name, candidates, x, y, *, metric, k, seed, domain=None)`; every
  fold of every trial fitted in one envelope.
- Artifact: `FAMILIES['bagging']` + schema from its first member; round-trip bit-for-bit with a GP member
  (member variance path). Envelope table row added.
- Tests ported with bars unchanged (1e-9 mean, 1e-12 variance, 1e-8 gradient; r2 1.0 at 1e-9); new: one-envelope
  members (control: 2.1 refused), tuning a bag's member in a domain, a domain not holding the rows refused.

---
created: 2026-11-07
---
# optimi-lab `model/` moved in (A1): response models under `learn_lab.models.response`

- Layout: the 13 families + kernel/likelihood in `models/response/`; `protocol`, `registry`, `validation`,
  `generations` at the top (cross-family). `protocol.py` is now the one copy (optimi-lab's text was the behaviour
  truth: `query_points` joined it). Refusals are `learn_lab.errors.Refused`.
- `bench` did NOT move: it scores by Pareto front, IGD+ and a feasibility acquisition -- search domain. It stays in
  optimi-lab (`optimi_lab.surrogate.bench`) and imports models from here.
- Runtime gained numerics-lab and scipy (special only); scikit-learn is a dev-group test oracle.
- Not moved yet: optimi-lab `pr/sensitivity-bagging-hpo` adds `model/bagging.py` and `model/tuning.py`; whoever
  integrates it lands them here.

---
created: 2026-11-08
---
# Tuning states WHAT is searched; optimi-lab owns the search

- User ruling 2026-11-08: learn-lab declares what is tuned and how a point scores; the search is optimi-lab's.
  `tune()` (a private grid search) is deleted, no shim; `tuning import tune` and `Trial(` are retired rows.
- `space.py`: `Real(lo, hi, log=False)`, `Integer(lo, hi)`, `Choice(values)` checked at construction
  (`lo < hi`, log needs `lo > 0`, a choice is non-empty); `Space` a frozen mapping (`problems`, `holds`,
  `lowest`); `Family(build, space, shaped=False)`.
- `registry.SURROGATES` rows are `Family`s; `registry.family(name)` resolves response or deep, the deep rows
  from `deep/registry.py::DEEP` imported on first use (torch extra). Deep families: deep_ensemble, unet,
  deeponet, fno, gnn, neural_ode, monotone_law, play_hysteresis -- each tunes `steps` and `learning_rate` beside
  its own widths. A shaped build folds sample-indexed arrays (flattened to rows for `cross_validate`); data
  beyond `(x, y)` is `fixed` (`points`, `edges`, `t`, `input_is`). The physics-informed field fits a residual,
  not data: no data objective.
- `tuning.objective(name, x, y, *, metric='r2', k=5, seed=None, domain=None, fixed=None, checkpoint=None)`:
  one data/fold/envelope set for every call; returns a value to MINIMISE (r2 negated); refuses a setting
  outside the space and a fixed setting that is also tuned; a numerically failing fit (Refused, LinAlgError,
  FloatingPointError) is `inf` with its reason in `Objective.evaluations`. A deep checkpoint lands under
  `<dir>/<settings digest>/member-<fold>`, so a rerun resumes in any search order.
- Planted controls: degree 1 vs 2 on an exact quadratic; 300 vs 1 Adam steps on a linear map; out-of-space,
  unknown and fixed-and-tuned settings refused; nine ill-formed spaces refused; a singular RBF tail is `inf`.

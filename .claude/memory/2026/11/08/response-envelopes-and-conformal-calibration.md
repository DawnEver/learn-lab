---
created: 2026-11-08
---
# Response models carry an envelope; the GP's x1.4 sd patch became a measured conformal widening

Closes surrogate gaps 2-3 of motronics' `memory/2026/11/07/HANDOFF-surrogate-models-state-and-gaps-for-the-other-workstation.md`.

## Envelope (every response family)

- ONE rule in `protocol.py`: `training_data(x, y, domain=None) -> (x, y, envelope)`; `query_points(x, envelope)`
  refuses OUT_OF_DOMAIN and non-finite rows. Every `fit` takes `domain: BoxEnvelope | None`; every fitted model
  exposes `.envelope` (a field, or for ensemble / forest / co-Kriging a property read from what holds it once).
- Decision: the envelope is the DECLARED domain when given -- refused unless it holds every training row --
  else the training span. Evidence: optimi-lab predicts on candidates drawn from the whole design space
  (`solver/model_based.py::maximise` Sobol over the unit cube; `policy/proposals.py::surrogate_candidates`
  draws from `space.natural_sample`; trust region scales by its radius). A span-only envelope would refuse
  most of them; an `extrapolate=` switch would be a dual path. Declaring the space is the first-principles
  answer: the model is asked about that space, so that space is its envelope.
- `ChaosExpansion` defaults to the unit cube (its basis lives there). Cross-validation / `scored_fit` /
  `Ensemble` / `Forest` fit every fold, member and bootstrap tree in ONE envelope (the whole sample's), else an
  extreme held-out row would be refused for being outside its own fold's span. `GP.condition` keeps the
  envelope and refuses fantasy rows outside it. `Network` gained `envelope` as its first field;
  `MultilayerPerceptron.initial(envelope, k)`; `FittedTree.width` became `.envelope`.
- `artifact.py`: a response family's `schema.inputs` carries `envelope: {lower, upper}`; the state tree already
  round-trips the `BoxEnvelope` dataclass. `DeepEnsemble.fit` takes `domain=` too (else span widened by margin).

## Calibration (`learn_lab/calibration.py`)

- `conformal(measured, prediction, coverage)` (normalised scores `|y - mean| / sd`, rank `ceil((n+1)c)`),
  `split_conformal(surrogate, x, y, coverage, share, seed, domain)` (makes the split itself),
  `jackknife(fitted_gp, coverage)` (Dubrule LOO residuals over the held-out OBSERVATION sd; apply to
  `sqrt(latent + noise)`), `ConformalInterval.report` (empirical coverage, mean width), `.widening = q / z_c`.
- Measured, 200 seeds, GP on 24 rows of a 20x-heteroscedastic 2-D problem, 99 calibration rows, 2000 test:
  Gaussian 90 % obs interval 0.843; split-conformal 0.898 (bar 0.88-0.92, sd of the mean ~0.002); jackknife
  0.884 (bar 0.86-0.92, approximate guarantee); planted in-sample calibrator 0.780 (must be < 0.85).
- optimi-lab `pr/model-moves-to-learn-lab`: `GP_SD_INFLATION` / `GP_CALIBRATED_ROWS` / `sd_inflation` removed;
  `predictive` widens by `jackknife(fitted, coverage=min(0.95, n/(n+1))).widening` per column; every fit site
  declares its domain (unit cube; trust-region scaled cube; the space's natural box; the bench's train+test box).

## Left

- Acceptance in motronics still does not READ a calibrated interval (gap 3's second half): the API exists,
  the consumer does not.

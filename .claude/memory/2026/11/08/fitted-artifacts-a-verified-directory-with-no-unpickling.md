---
created: 2026-11-08
---
# Fitted artifacts (G4 / review H7): a verified directory, nothing unpickled

- `learn_lab/artifact.py`: `save(fitted, path, *, declaration, data_identity, metrics=None)`, `load(path, *,
  device=None)`, `manifest(path)`. ONE new directory: `manifest.json` (format, `version` 1, learn-lab version +
  git sha/dirty, family, declaration settings, schema, seed, runtime, data identity, metrics, the fitted STATE
  as a tagged JSON tree, per-file SHA-256, `content_hash`), `arrays.npz` (`allow_pickle=False`), `weights.pt`
  (deep only; flat `{name: tensor}`, `torch.load(weights_only=True)`, written on CPU).
- Why a directory and not a zip: each part is readable by its own standard tool for an audit; a zip adds a
  layer and buys nothing the per-file hashes do not.
- Why a generic state tree and not per-family codecs: every fitted family is a frozen dataclass of arrays,
  scalars, tuples, enums and nested dataclasses, so ONE structural encoder covers all 22. Decoding resolves only
  dataclasses/enums inside `learn_lab.` and rebuilds them THROUGH their constructors (their own checks run;
  e.g. `Runtime` refuses `cuda` on a box without it). Torch modules come only from `deep.artifact.MODULES`.
- `declaration=` is required (not in the original brief's signature): a fitted object does not carry its
  constructor settings, and H7 asks for them; save checks it is the family's own declaration type.
- Three closures blocked persistence and were the cause, fixed at source: `FittedField.field_at` and
  `FittedDynamics.rate_net` were closures over the net -> the fit now holds `net` (+ `time_scale`);
  equation-library terms were lambdas -> `Monomial(factors)` data terms (a caller's own callable term stays
  legal, and save refuses it naming `library.terms[i]`). `mlp()` became `MLP` recording `arguments`; every
  persisted module records its constructor arguments.
- Loading builds modules under `torch.random.fork_rng`: loading never moves the caller's seed (control:
  building an MLP does).
- Measured: bit-for-bit (`np.array_equal`) round trip for all 12 response families + sparse regression and all
  9 deep families on CPU; CUDA->CUDA exact; CUDA->CPU float32 within 1e-5 of max|output| (+1e-6 floor, output
  unit), and a one-step-different fit is outside that bar (control). Tamper of one payload byte, one manifest
  value, an unknown version, a class outside learn_lab, a non-dataclass learn_lab name, an object array in the
  npz (consistent hashes), a module outside the table: each refused.
- FORMATS-UPGRADE-ON-READ stays absent with a new reason: the rule has a subject now, but version 1 is the
  only schema issued; the second schema must bring an upgrade step as data beside `VERSION` and its test.
- Not covered: units (learn-lab is unit-free; the caller's data identity names them), response models carry no
  validity envelope (the GP/polynomial families only check width), so their manifest schema has no bounds.

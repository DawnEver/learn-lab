# learn-lab

Learned models: fit and evaluate surrogates, physics-informed fields, learned laws, inverse problems.

- Runtime dependency: numpy. The deep models (`learn_lab.deep`) need the `[torch]` extra, which resolves to the
  CUDA 12.8 wheels on Windows and Linux and to PyPI's CPU wheel on macOS.
- Install for development: `<any family venv python> -m dev_lab.dev.dep --bootstrap <this checkout>` (the extras every
  environment carries are `[tool.dev_lab.dep]`'s).
- Measure a change: `<venv python> -m dev_lab.dev.verify -- <paths>`.

Contributing: see `AGENTS.md`.

# learn-lab

Learned models for the family, domain-free: surrogates, physics-informed fields, learned constitutive laws, inverse
problems and hybrids. The public boundary speaks numpy only; torch is the optional `[torch]` extra, imported only
under `learn_lab.deep` (`tests/architecture/test_the_layers_point_one_way.py`).

## How a change lands

The family flow, stated once in [the integration branch](https://github.com/DawnEver/lab-commons/blob/integrate/main/docs-src/dev/integration.md):

- Cut `pr/<topic>` from `origin/integrate/main`, measure what you touched, commit, push, open a pull request.
- A producer claims the pushed head and posts `tests/<part>`; the FORGE merges the PR once every part
  stands PASS, the head merges cleanly and the merge passes the audit.
- `main` moves only by `integrate promote <sha>` on a full PASS.
- `ephemeral = ['pr/*']` and `pr_merges_since` are declared in `pyproject.toml`, and
  `tests/architecture/test_this_repo_declares_the_one_pr_flow.py` refuses this repo without them.

The mechanisms are lab-commons's (`docs-src/dev/`); this repo declares the names and runs the assertions.

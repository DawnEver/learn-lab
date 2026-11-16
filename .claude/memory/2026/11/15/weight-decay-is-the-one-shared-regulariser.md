---
name: weight-decay-is-the-one-shared-regulariser
description: Schedule.weight_decay (decoupled, AdamW) is the one regulariser every learned family shares; dropout stays out
created: 2026-11-15
---

# `Schedule.weight_decay` -- why decoupled decay is the one shared knob

Motivation: a downstream consumer's Transolver point-field fit (1262 designs) ends with held worst 0.0563 vs
train worst 0.0446, and doubling the data barely moved the gap. learn-lab had no regulariser.

## Which regulariser, from first principles

- The one thing every family shares is the optimiser: `minimise` (`src/learn_lab/deep/fit.py`). A knob on
  `Schedule` reaches every family with no per-architecture code, the way `batch` did.
- Decoupled weight decay (AdamW) shrinks each parameter by `rate(step) * weight_decay` per step, outside Adam's
  per-coordinate scaling. An L2 term added to the loss would instead be rescaled by Adam's second moment, so its
  strength would differ per coordinate and per family -- not one knob with one meaning.
- It keeps the closure contract (`loss(index)` is unchanged, deterministic), and it is resumable: the decay
  lives in the optimizer param group, saved with the Adam state; `repr(schedule)` puts it in the checkpoint
  identity, so a resume under another decay is refused.
- `0.0` is plain Adam: AdamW multiplies by `1 - lr * 0 == 1.0` exactly, so the history and weights are
  bit-identical (test pinned against a hand-written `torch.optim.Adam` loop).
- The L-BFGS polish takes no decay: it minimises the objective as given. Decoupled decay is a property of an
  optimiser step, not of the objective, and L-BFGS's line search needs a fixed objective to be meaningful.

## Dropout -- not added

- Dropout is per-architecture (where the masks sit is a design choice per family), and makes the loss
  stochastic: under L-BFGS the strong-Wolfe line search would compare evaluations of different functions. It
  would need train/eval mode switching in every family and a seeded mask stream to stay resumable.
- The measured gap is not yet shown to need more than decay. If a decay sweep leaves the Transolver gap
  open, dropout belongs on that family's own config (attention/slice-weight dropout) with a planted control,
  Adam-only, disabled in the polish -- not on `Schedule`.

## Where it lives

- `Schedule.weight_decay` (validated finite `>= 0`), `minimise` uses `torch.optim.AdamW`.
- Every registry builder takes `weight_decay=0.0` (`src/learn_lab/deep/registry.py::_schedule`).
- Tests: `tests/unit/deep/test_weight_decay.py`.

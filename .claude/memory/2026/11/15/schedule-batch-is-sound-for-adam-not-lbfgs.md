---
name: schedule-batch-is-sound-for-adam-not-lbfgs
description: Schedule.batch mini-batches Adam only; the batch mean is unbiased, draws are a function of (seed, step), L-BFGS sees every sample
created: 2026-11-15
---

# `Schedule.batch` -- why mini-batching is sound here, and where it stops

User ruling 2026-11-15: one `batch` field on the shared `Schedule` (`src/learn_lab/deep/fit.py`), used by every
learned family. Motivation: a full-batch closure holds every sample's activations at once, so fit memory is
O(N x model); a downstream consumer had to cut a graph-operator fit to 10 designs (2 GB) and fits a Transolver on
1262 point clouds.

## Soundness

- Every family's loss is a MEAN over its sample axis (rows, graphs, point clouds, trajectory windows, histories,
  snapshot pairs, matrices). A mean over a uniformly drawn subset is an unbiased estimate of the full mean, and so
  is its gradient: Adam on it is stochastic gradient descent on the same objective, and a learning rate means
  what it meant full-batch. A sum would scale the gradient by N/batch -- hence means, never sums. Normalisers that
  are data statistics stay FULL-data constants (PDE-Net divides by the full `mean(rate**2)`), or the estimate
  would be a ratio of two noisy terms.
- Where samples differ in size (graphs of different node counts, masked histories, matrices of different order)
  the per-element mean over a batch is a ratio estimator: consistent, biased O(1/batch) toward equal weighting.
  Accepted; equal-size data (the common case) is exactly unbiased.
- L-BFGS is excluded: its strong-Wolfe line search compares losses at trial points, and its curvature pairs
  difference gradients across steps -- both assume ONE deterministic objective. A fresh batch per evaluation
  makes both meaningless, so every polish evaluation and every recorded polish loss is `loss(None)`.
- Resumability: indices are drawn in ONE place (`minimise`) from a private generator seeded by (seed, step), not
  from the global RNG, so a fit resumed from a checkpoint draws the batches the uninterrupted fit drew, and a
  batched fit perturbs no other random stream. The checkpoint identity hashes `repr(schedule)`, so the batch is
  part of it (`test_the_batch_is_part_of_the_checkpoint_identity`). Adding the field changes that repr even at
  `batch=None`: a checkpoint written before this change is refused as another config, never silently resumed.
- `batch=None` takes the old code path op for op (`take(None, ...)` returns the tensors as they are), so existing
  results are bit-identical (`test_batch_none_is_the_plain_full_batch_adam_loop_bit_for_bit`).

## Memory and small N

- Peak activation memory per Adam step is O(batch x model) instead of O(N x model); the data tensors themselves
  stay resident (O(N x sample size), small next to activations). `chunk` (checkpointed recompute) remains the
  orthogonal tool for a single batch too large to hold.
- Small-N fits should keep `batch=None`: a batch near N buys no memory and adds gradient noise; with very small
  batches a constant rate leaves the weights jittering -- use `decay='cosine'` (or the L-BFGS polish, which sees
  the full objective) to settle them. `batch >= N` is every sample in a drawn order and matches full batch to
  rounding (the planted control).

## Families

Honoured (each closure slices its own sample axis via `fit.take`, or `graph.subgraph` for graph batches): deep
ensemble, U-Net field, DeepONet (sensors; shared query points), FNO, transformer operator, Transolver (own point
clouds sliced too), graph operator (shared graph), GNN (graphs of a batch), neural ODE (trajectory windows),
monotone law, play hysteresis (histories and their mask), diffusion field (rows; the noise draw is per batch),
PDE-Net (snapshot pairs; both phases), learned preconditioner (matrices). Refused by name: the physics-informed
field -- its objective is a quadrature over per-condition points with per-step resampling, balancing and
continuation state, not a mean over samples (`UNSUPPORTED-RAISES`).

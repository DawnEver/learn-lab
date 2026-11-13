---
created: 2026-11-12
corrects: 2026/11/12/transolver-physics-attention-over-learned-slices.md
---
# A slice token is a quadrature: the point measure closed the Transolver gap

- Corrects the proof line of `transolver-physics-attention-over-learned-slices.md`: the planted test now fits on
  192 jittered-grid clouds of 125 points (11 x 11 cells + corners) with Voronoi-area `measure`, Schedule 4500
  steps lr 3e-3 `decay='cosine'`, and judges 24 held-out sources on uniformly random 512-point clouds. Held-out
  mean relative L2: 0.0361 (seed 0), 0.0324 (seed 1) against the unchanged bar 0.05; pointwise MLP control 0.601.
- Cause 1, measured: Physics-Attention's token `sum_i w_ij x_i / sum_i w_ij` is a point MEAN, so N-invariant
  (not the suspected sum), but on a random cloud it is Monte-Carlo quadrature. Projecting f on the modes by an
  unweighted mean leaves 0.33 / 0.17 / 0.064 relative error at 128 / 512 / 2048 random points; area-weighted
  0.11 / 0.015 / 0.0017. The unweighted Transolver sat at 0.16-0.20 for every setting tried.
  Fix: `Transolver.fit(..., measure=)` / `predict(..., measure=)`, `(p,)` or `(n, p)` weights >= 0 scaled to
  mean 1; token = `sum_i a_i w_ij x_i / sum_i a_i w_ij`; deslice unchanged; a fit with a measure requires one
  at predict and vice versa (refused). 1500 steps: 0.074 / 0.075 vs unweighted 0.183 / 0.161.
- Cause 2, measured: the remaining ~0.07 was flat in query size (0.074 at 512, 0.075 at 2048) and in random
  training-set size (576 clouds 0.068 / 0.069; 256-point clouds 0.063 / 0.073), Ada-Temp + Gumbel (0.071 /
  0.067), reference=8 (0.107), a 4x larger net (0.073), 6000 steps (train 0.034, held-out 0.073): training
  inputs carried 128-point random-cloud quadrature noise (errors in the variables). Jittered training clouds:
  1500 steps 0.062 / 0.073 (train equal: under-trained), 4500 steps cosine 0.036 / 0.032.
- Slice usage after a 1500-step fit: entropy 1.1-1.4 of log 16 = 2.77, 5-8 of 16 slices carry their share per
  head (partial collapse; Ada-Temp did not change accuracy here).
- `_PhysicsAttention.last` is now (weights, scores, tokens). Planted test: the first layer's tokens of one source
  on 512 vs 2048 random points agree with the area measure better than half as far apart as with a uniform one.
- TransformerOperator on the earlier random-cloud task: 0.329 / 0.325 (one shared training cloud; not comparable).

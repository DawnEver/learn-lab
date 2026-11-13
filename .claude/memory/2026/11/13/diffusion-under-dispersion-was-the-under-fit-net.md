---
created: 2026-11-13
---
# Diffusion under-dispersion was the under-fit noise net, not the sampler

- Gap: `test_the_sample_variance_matches_the_closed_form` was a strict xfail; the relative variance error was 0.40..0.45
  against the 0.3 bar, with too little spread in the samples.
- Candidates from first principles: the reverse-step variance (posterior beta-tilde_t is the lower bound, beta_t the
  upper), too few noise steps, the end of the linear schedule, or the noise net under-fitting. Planted field, CPU
  float64, query c=0.8, 128 samples, seeds 0 / 1 (data and init). Relative variance error, posterior / beta:
  - width 8, 1000 Adam, 50 steps: -0.409 / -0.358 and -0.330 / -0.276
  - 200 noise steps: -0.399 / -0.386 and -0.361 / -0.348 (more steps do not help)
  - width 16: -0.344 / -0.305 and -0.319 / -0.284
  - 3000 Adam: -0.284 / -0.237 and -0.348 / -0.319
  - width 16 + 3000 Adam: -0.168 / -0.125 and -0.187 / -0.156 (mean 0.003 / 0.040 T, bar 0.06)
- The noise net's capacity times its training is the cause. Switching to 'beta' always helps by about 0.04, and in one
  case doubles the mean error (0.016 -> 0.032 T), so `sample(variance=)` / `moments(variance=)` keep 'posterior' as the
  default and offer 'beta'. The test now uses width 16 x 3000 Adam; the xfail is gone and the bar is unchanged.
- The sweep script sits under `output/logs/26/11/13/diffusion_variance/sweep.py` (not committed).

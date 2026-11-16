---
name: the-local-baseline-cpu-gpu-npu-table-for-the-a100-comparison
description: The first full baseline of python -m learn_lab.deep.bench on a Core Ultra 9 285K / RTX PRO 4000 Blackwell / Intel AI Boost NPU host, fit and inference per family, kept for the later A100 comparison.
created: 2026-11-15
---

# Local baseline: CPU / GPU / NPU (2026-11-14 18:01 UTC)

Reproduce (same command on the A100 host): `python -m learn_lab.deep.bench --size baseline --devices cpu,cuda,npu --steps 50 --repeats 5 --out output/logs/<yy>/<mm>/<dd>/bench`
Full record: `output/logs/26/11/14/bench/bench-180129.{json,md}` in the hw-bench worktree (regenerable).

Host: Intel Core Ultra 9 285K (24 threads), NVIDIA RTX PRO 4000 Blackwell (driver 621.14), torch 2.14.1+cu130,
cuDNN 92400, OpenVINO 2026.4.1, NPU "Intel(R) AI Boost" (int8 13.1 TOPS, fp16 6.6 TFLOPS nominal).

## Fit, median ms per Adam step (float32; baseline sizes)

| family | CPU | CUDA | CUDA speed-up |
|---|---|---|---|
| mlp (deep ensemble, 1 member) | 3.28 | 1.29 | 2.5x |
| fno | 425 | 25.3 | 16.8x |
| field (U-Net) | 245 | 15.6 | 15.7x |
| transolver | 327 | 24.2 | 13.5x |
| transformer operator | 35.8 | 6.46 | 5.5x |
| graph operator | 57.9 | 4.83 | 12.0x |

float64 on this GPU is 10-30x slower than its float32 (a workstation card's FP64 rate), e.g. transolver 751 ms/step
vs 24.2; on CPU float64 costs about 2x. The CUDA float64 rows carry a CONTENDED flag (13-76 % utilisation at their
start, the tail of the run's own preceding rows), so they are upper bounds.

## Inference, median ms (batch 1 / batch 64; mlp batch 8192)

| family | CPU fp32 | CUDA fp32 | NPU fp32 | NPU int8 |
|---|---|---|---|---|
| field | 3.54 / 62.4 | 2.94 / 26.6 | 1.30 / 27.9 | 1.21 / 31.9 |
| fno | 10.6 / 130 | 2.88 / 34.1 | 23.9 / 122 | 18.7 / 110 |
| transolver | 5.37 / 204 | 3.20 / 27.9 | refused | refused |
| mlp | 0.185 / 3.98 | 0.576 / 1.25 | refused | refused |

- The NPU wins single-sample latency for the U-Net field model (1.2-1.3 ms vs 2.9 on CUDA, 3.5 on CPU) and loses
  for the FNO (its FFTs): 19-24 ms vs 2.9 on CUDA. int8 helps the FNO by about 20 %, the field model not at all.
- NPU compile time is negligible at batch 1 (0.1-0.2 s) but 53-66 s for the FNO at batch 64.
- Only `field_model` and `fno` export to ONNX today; every other family is `refused` on the NPU (user ruling
  2026-11-15: measure only what exports).
- At batch 1, CUDA is launch-bound (~2-3 ms floor); the MLP is faster on CPU (0.19 ms) than on CUDA (0.58 ms).

## For the A100 comparison

Compare the float32 fit rows (the A100 has full-rate FP64, so its float64 rows will close most of the 10-30x gap
seen here) and the batch-64 inference rows. A training cost estimate from this table: the 1262-design Transolver
fit downstream (width 128, 20000 steps) took 7.2 h on this GPU.

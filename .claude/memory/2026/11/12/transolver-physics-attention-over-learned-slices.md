---
created: 2026-11-12
---
# Transolver: Physics-Attention over learned slices, linear in the points

- Source: Wu, Luo, Wang, Wang, Long, "Transolver: A Fast Transformer Solver for PDEs on General Geometries",
  ICML 2024 (arXiv 2402.02366, github thuml/Transolver); Transolver++ (arXiv 2502.02414).
- Physics-Attention, per head, N points, M slices: `w_i = softmax(Linear(x_i) / tau)` over the M slices
  (each row sums to 1); token `z_j = sum_i w_ij x_i / sum_i w_ij`; `q, k, v = Linear(z)`,
  `z' = softmax(q k^T / sqrt(C)) v` among the M tokens; deslice `x'_i = sum_j w_ij z'_j`. Cost
  O(N M C + M^2 C). The reference code projects separate "fx" (values) and "x_mid" (assignment) branches,
  `tau` a per-head learnable parameter initialised 0.5, orthogonal init of the slice projection, 1e-5 in the
  token normaliser.
- Block (paper Eq. 6, pre-norm): `x^ = PA(LN(x)) + x; x = FFN(LN(x^)) + x^`; input `x0 = Linear(concat(g, u))`
  (an MLP in code); last block LN + linear head. Optional "unified position": distances to a reference grid.
- Standard benchmarks: 8 layers, C 128 or 256, M in {32, 64} (64 for Elasticity, Darcy), AdamW lr 1e-3,
  OneCycle, 500 epochs (code). Relative L2: Elasticity 0.0064, Plasticity 0.0012, Airfoil 0.0053, Pipe
  0.0033, Navier-Stokes 0.0900, Darcy 0.0057. ShapeNet Car: volume 0.0207, surface 0.0745, C_D 0.0103,
  rho_D 0.9935. AirfRANS: volume 0.0037, surface 0.0142, C_L 0.1030, rho_L 0.9978.
- Transolver++: Ada-Temp `tau_i = tau0 + Linear(x_i)` (per point); Rep-Slice
  `softmax((Linear(x) - log(-log eps)) / tau)`, eps ~ U(0, 1) (Gumbel); multi-GPU by all-reducing the
  O(M C) token numerators and O(M) normalisers only. Standard results: Elasticity 0.0052, Plasticity 0.0011,
  Airfoil 0.0048, Pipe 0.0027, NS 0.0719, Darcy 0.0049; million-point DrivAerNet++ / AirCraft gains > 20%.
- Against learn-lab's other operators: `TransformerOperator` attends over all p points (softmax O(p^2), or the
  elu+1 linear kernel) with one shared point set per call; `GraphOperator` mixes only along graph edges
  (local kernel integral); `FNO` needs a regular grid. Transolver mixes globally through M learned soft groups.
- Implemented: `deep/operator.py::Transolver` / `FittedTransolver` (`_PhysicsAttention`, `_TransolverNet`);
  points shared `(p, d)` or per sample `(n, p, d)`, any count at predict; switches `adaptive_temperature`,
  `gumbel` (noise only in training mode: prediction deterministic), `reference` (unified position),
  `mlp_ratio`, `temperature`. Family `transolver` in `DEEP` (tunes width, heads, layers, slices), FAMILIES,
  MODULES (`TransolverNet`), schema. `_PhysicsAttention.last` keeps (weights, scores) for inspection.
- Proof (`tests/unit/deep/test_transolver.py`): `-lap u = f` on the unit square, f = 9 sine modes, fit on 192
  different 128-point clouds, judged on 24 different 512-point clouds; numbers in that test's docstring.

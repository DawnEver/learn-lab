---
created: 2026-11-08
---
# Play-bank hysteresis, the torch/numpy gradient bridge, PDE-FIND

- `deep/hysteresis.py`: generalized Prandtl-Ishlinskii (plays r>=0, w>=0, slope a>0, `MonotoneScalar` F).
  Rate independence, monotone branches and passivity (`oint x dy >= 0`) hold by construction; `dissipation`
  measures it. The play is written as a threshold TEST, not a clamp, so a negative threshold LEADS (active) --
  the planted control; a clamp with crossed bounds would give no loop and the control could not say no.
- BPTT through 180 sequential steps cost 0.37 s per Adam step (GPU launch-bound). The play recursion unrolls
  exactly: each output is 0 or `x[tau] - sigma r`; tau/sigma in numpy, one differentiable gather -- same
  gradient (stepwise and unrolled agree to 0.0), fit 5.5 s. Held-out FORC/nested-reversal error 0.53 %/0.62 %
  of the major-loop span (bar 2 %).
- `deep/bridge.py`: `classical_op(forward, vjp)` with `vjp(x, y, cotangent)` -- y passed so an adjoint never
  re-solves; `check_vjp` refuses above 1e-6 relative (floor 1e-12 in gradient unit). Controls: sign error and
  missing transpose (A made unsymmetric so the latter is detectable).
- `equation.py`: `SparseRegression.fit(..., targets=)` -- one STLSQ for ODE and PDE. Repeated `np.gradient`
  for u_xx (2dx stencil) gave nu 1.2 % off; compact 3-point stencils on interior points: 0.22 %.
  Noise is NOT refused (smoothness is a data property, not the library's); the support check is the arbiter,
  and 1 % noise fails it (4 terms, 2 spurious).

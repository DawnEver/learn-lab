---
created: 2026-11-08
---
# The play-bank hysteresis declares its causality; "passive by construction" held only for an effort input

- DEFECT (measured downstream, motronics `lane/material-seams` a61d3b24c5): `y = F(a x + w.p(x))`, lagging plays,
  used with x = B, y = H gave oint H dB < 0 (rotational -132.6 J/m^3, alternating loops negative too).
  `dissipation` measured oint x dy, which is the work only when x is the EFFORT; for a displacement input it is
  minus the work, so the planted control said "passive" about an active material. Both the model and the
  instrument were wrong for the B-driven use.
- FIX: `PlayHysteresis(input_is='effort'|'displacement')`, required (no default -- the orientation decides which
  loop is passive). `dissipation(..., input_is=)` measures oint x dy (effort) or oint y dx (displacement).
  - effort: unchanged, `F(a x + w.p(x))`, plays lag.
  - displacement: `u = G(x)`, `y = a u + w.(u - p(u))` -- Prandtl STOP operators, which lead. Exact inverse class
    of the effort form: inverting `v = a x + w.p_r(x)` gives a stop bank in v with thresholds
    `a r_i + sum_{j<i} w_j (r_i - r_j)` and non-negative weights, and the outer F becomes the inner G = F^-1.
    Same state (play outputs), same r/w/a parameters; slope `G'(a + sum_held w) >= a G' > 0`; rate independent.
  - The monotone net is now `PlayBank.monotone` (it is inner for displacement): weights key renamed.
- MEASURED (CPU float64, defaults): held-out max error / major-loop span 5.3e-3, 6.2e-3 (effort), 4.6e-3, 5.9e-3
  (displacement), bar 2e-2. Major-loop energy vs the planted law's 0.9621: 0.11 % (effort), 0.69 % (displacement),
  bar 5 %. PLANTED: the effort construction fitted B -> H loses -9.8e-3 measured as displacement-driven (the
  defect, caught), and looks passive (+) measured in its own declared orientation -- why the declaration matters.
- ARTIFACT: format VERSION stays 1 -- the container schema is unchanged; the family's declaration gained a field
  and its module arguments `input_is`. A play_hysteresis artifact written before is refused at load by the
  existing field check (`... has fields [...], the artifact holds [...]`), never read under a guessed orientation.

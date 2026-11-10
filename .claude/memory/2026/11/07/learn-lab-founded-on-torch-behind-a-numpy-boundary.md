---
created: 2026-11-07
---
# learn-lab founded: torch families behind a numpy boundary

- Rulings (user, 2026-11-07, recorded in motronics `RULING-learn-lab-on-torch-behind-the-shannon-door.md`): PyTorch,
  Windows + Linux via the pytorch cu128 index; motronics reaches this library through `motronics.shannon`; full
  lab-commons/dev-lab adoption.
- Layout: `protocol.py` (L6, owner here; optimi-lab's copy is deleted at A1), `validity.py` (refuse out-of-domain),
  `deep/` (the only torch importer): `surrogate`, `physics` (ResidualProvider; PINN and inverse are one fit),
  `constitutive` (monotone law; value/derivative callables are the hybrid seam).
- `MonotoneScalar` needs both tanh (concave) and softplus (convex) odd pairs: a tanh-only basis missed the B-H knee
  by 37 % of range (measured 1308 A/m of 3558 A/m).
- `deep.fit.minimise` must not record the final loss under `no_grad`: a residual loss differentiates the field wrt
  its inputs.

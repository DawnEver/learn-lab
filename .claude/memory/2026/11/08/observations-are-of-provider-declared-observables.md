---
created: 2026-11-08
---
# Observations are of provider-declared observables, not only of the field

Follows `observations-pin-only-the-column-the-provider-owns.md`. Driver: motronics' current-free core hides a 2 %
`mu_r` below the forward fit's accuracy in `A` (gain 0.2 / 0.11); `B = curl A` carries it at gain ~1.

## API (one path, no shim)
- Provider declares `observables: {name: k}`, `observe(name, field, points) -> (n, k)` (differentiable; the field
  itself is declared explicitly like any other), `owns(name, points) -> (n, k)`.
- `fit(observations={name: (x, values)})`: one data term per observable, scaled by the RMS of its owned values.
- Refused: undeclared name, values not `(n, k)`, `owns` of the wrong shape, no owned value, non-finite owned
  value, all-zero owned values (no scale), empty mapping.

## Decisions (decision -- principle -- evidence)
| # | decision | principle | evidence |
|---|---|---|---|
| O1 | Mapping name -> (x, values) over declared observables, not a single `observe` stacking all. | keep-valuable-options | several sensors at different points (B in a core, A in a coil) |
| O2 | `owns` takes the observable name. | single-source-of-truth | the provider's region map decides ownership of derived components too |
| O3 | Data scale = RMS of owned values (was std). | TOLERANCE-CARRIES-A-UNIT | std of a uniform slope is 0 -> fell back to a bare 1.0 |
| O4 | Planted case = stiff slab (k_right 20) with a 0.5 % sensor gain error, sensors in the stiff slab. | PLANTED-CONTROL | exact data: u identified 19.99 (control could not say no) |

## Measured (CPU, FAST schedule)
- u' observed: 19.89 (bar 5 %), 31 s. u observed: 22.37 (+11.8 %), 25 s. Closed-form LS: 19.90 / 23.8.
- `verify` physics + artifact + architecture: 81 passed.

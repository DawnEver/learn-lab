# Observations pin only the column the provider owns

**Defect** (measured in motronics `lane/em-residual-provider`): `PhysicsInformedField.fit` fitted EVERY output
column at every observation point, so a provider with one column per region had its observations near an
interface pin the NEIGHBOUR's column, pulling the identified coefficient toward the neighbour (two-slab mu_r,
truths 4 and 8: identified 2.30 / 2.65).

**Fix**: the `ResidualProvider` protocol gains `owns(points) -> (n, outputs) bool`; `fit` fits only owned entries
(unowned may be NaN; a non-finite owned value, a wrong-shape mask or no owned entry is refused).

**Why the provider, not a caller mask**: the provider defines the regions, so it is the one source of which column
is the field where (`single-source-of-truth`); a caller mask duplicates the region map and can disagree with it.
A boolean mask rather than one column index per point keeps vector fields (several columns observed at one point,
`keep-valuable-options`). No dual path: every provider implements `owns`.

**Measured** (`tests/unit/deep/_two_slab.py`, k_left = 1, k_right = 4, 12 observations incl. x = 0.48, 0.52):
owned columns only -> k_right = 4.0000 (bar 5 % relative); planted control `EveryColumnEverywhere` (the old
behaviour) -> 2.675 (-33 %).

**Downstream**: motronics' hamilton magnetostatics provider must implement `owns` (one-hot of the region column
at each point) and drop any caller-side workaround.

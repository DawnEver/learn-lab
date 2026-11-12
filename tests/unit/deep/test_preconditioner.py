"""A learned preconditioner cuts CG iterations on held-out matrices of its family and never moves the answer.

Family: 2-D diffusion ``-div(k grad u)`` on an ``m x m`` interior grid (5-point finite volume, zero Dirichlet),
``k = exp(1.5 * smooth random field)`` -- SPD, a different coefficient per matrix. The fit sees 12 matrices; the
held-out ones are 6 others of the training size and 3 of a larger grid. A small in-test CG runs to a relative
residual of :data:`RTOL`. PLANTED CONTROL: the untrained network is Jacobi exactly -- the same iteration
count, so a reduction is the training's, not the comparison's.
"""

import numpy as np
import pytest
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from learn_lab.deep.fit import Schedule
from learn_lab.deep.preconditioner import LearnedPreconditioner
from learn_lab.deep.runtime import Runtime
from learn_lab.errors import Refused

RTOL = 1e-8
#: Mean learned iterations over mean Jacobi iterations on the held-out matrices must be below this.
REDUCTION = 0.7
CPU64 = Runtime(device='cpu', dtype='float64')


def _diffusion(m: int, seed: int) -> sp.csr_matrix:
    """Cell coefficients on an ``(m + 2)^2`` grid (a ghost ring for the boundary faces); harmonic-mean faces."""
    rng = np.random.default_rng(seed)
    x = np.linspace(0, 1, m + 2)
    waves = [(i, j) for i in range(3) for j in range(3)]
    field = sum(rng.standard_normal() * np.cos(np.pi * (i * x[:, None] + j * x[None, :])) for i, j in waves)
    k = np.exp(1.5 * field / 3)
    index = np.arange(m * m).reshape(m, m)
    rows, cols, vals, diagonal = [], [], [], np.zeros(m * m)
    for a in range(m):
        for b in range(m):
            for da, db in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                here, there = k[a + 1, b + 1], k[a + 1 + da, b + 1 + db]
                face = 2 * here * there / (here + there)
                diagonal[index[a, b]] += face
                if 0 <= a + da < m and 0 <= b + db < m:
                    rows.append(index[a, b])
                    cols.append(index[a + da, b + db])
                    vals.append(-face)
    return sp.csr_matrix((vals, (rows, cols)), shape=(m * m, m * m)) + sp.diags(diagonal, format='csr')


def _cg(a, b, precondition):
    """Preconditioned CG to ``||r|| <= RTOL ||b||``: ``(x, iterations)``."""
    x = np.zeros_like(b)
    r = b.copy()
    z = precondition(r)
    p, rz = z.copy(), r @ z
    for iteration in range(1, 10 * len(b)):
        q = a @ p
        alpha = rz / (p @ q)
        x, r = x + alpha * p, r - alpha * q
        if np.linalg.norm(r) <= RTOL * np.linalg.norm(b):
            return x, iteration
        z = precondition(r)
        rz, previous = r @ z, rz
        p = z + (rz / previous) * p
    msg = 'CG did not converge'
    raise AssertionError(msg)


def _jacobi(a):
    inverse = 1.0 / a.diagonal()
    return lambda r: inverse * r


TRAIN = [_diffusion(10, seed) for seed in range(12)]
HELD_OUT = [_diffusion(10, seed) for seed in range(100, 106)] + [_diffusion(14, seed) for seed in range(200, 203)]
B = {id(a): np.random.default_rng(7).standard_normal(a.shape[0]) for a in HELD_OUT}


def _declaration(steps: int) -> LearnedPreconditioner:
    return LearnedPreconditioner(
        hidden=16, rounds=3, schedule=Schedule(adam_steps=steps, learning_rate=3e-3), runtime=CPU64, margin=0.5
    )


@pytest.fixture(scope='module')
def fitted():
    return _declaration(250).fit(TRAIN)


def _iterations(preconditioner_of):
    return [_cg(a, B[id(a)], preconditioner_of(a))[1] for a in HELD_OUT]


def test_the_learned_preconditioner_cuts_cg_iterations_on_held_out_matrices(fitted) -> None:
    jacobi, learned = _iterations(_jacobi), _iterations(fitted.for_matrix)
    ratio = np.mean(learned) / np.mean(jacobi)
    assert ratio < REDUCTION, (jacobi, learned, ratio)


def test_the_preconditioned_solution_is_the_unpreconditioned_one(fitted) -> None:
    """A preconditioner changes speed, never the answer: both CG solutions meet the direct solve."""
    for a in HELD_OUT:
        b = B[id(a)]
        exact = spla.spsolve(a.tocsc(), b)
        for precondition in (_jacobi(a), fitted.for_matrix(a)):
            x, _ = _cg(a, b, precondition)
            assert np.linalg.norm(a @ x - b) <= RTOL * np.linalg.norm(b)
            assert np.linalg.norm(x - exact) <= 1e-5 * np.linalg.norm(exact)


def test_an_untrained_network_is_jacobi_and_gives_no_reduction() -> None:
    """PLANTED CONTROL: the zero-initialised heads make M = D^-1; one step at a vanishing rate stays there."""
    untrained = LearnedPreconditioner(
        hidden=16, rounds=3, schedule=Schedule(adam_steps=1, learning_rate=1e-300), runtime=CPU64, margin=0.5
    ).fit(TRAIN[:2])
    a = HELD_OUT[0]
    r = np.random.default_rng(1).standard_normal(a.shape[0])
    assert np.allclose(untrained.for_matrix(a)(r), r / a.diagonal(), rtol=1e-14, atol=0.0)
    assert _iterations(untrained.for_matrix) == _iterations(_jacobi)


def test_the_preconditioner_is_symmetric_positive_definite(fitted) -> None:
    a = HELD_OUT[-1]
    m = fitted.for_matrix(a)(np.eye(a.shape[0]))
    assert np.allclose(m, m.T, rtol=0.0, atol=1e-12 * np.abs(m).max())
    assert np.linalg.eigvalsh(m).min() > 0


def test_a_non_symmetric_or_indefinite_matrix_is_refused(fitted) -> None:
    a = TRAIN[0].toarray()
    skew = a.copy()
    skew[0, 1] += 0.5
    with pytest.raises(Refused, match='symmetric'):
        fitted.for_matrix(skew)
    with pytest.raises(Refused, match='diagonal'):
        fitted.for_matrix(-a)

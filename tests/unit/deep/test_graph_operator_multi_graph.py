"""A graph operator fits ONE kernel on samples over SEVERAL graphs (discretisations of one domain).

Planted operator: ``-u'' = f`` on ``[0, 1]``, ``u(0) = u(1) = 0``, ``f = sum_k a_k sin(k pi x)`` -- so
``u = sum_k a_k sin(k pi x) / (k pi)^2`` exactly. The claim: trained on a fine AND a coarse grid, the kernel
predicts held-out coarse-grid samples better than the same budget trained on the fine grid alone, because a
discretisation-invariant operator learns the continuum map only when it sees more than one discretisation.
"""

import numpy as np
import pytest

from learn_lab.deep import operator
from learn_lab.deep.fit import Schedule
from learn_lab.deep.operator import GraphOperator, GraphSamples
from learn_lab.deep.runtime import Runtime
from learn_lab.errors import Refused
from learn_lab.point_graph import intervals, mesh_graph

MODES = np.arange(1, 4)
CPU64 = Runtime(device='cpu', dtype='float64')
#: The mixed fit's held coarse RMS error is at most this fraction of the fine-only fit's (u's unit cancels).
#: The mixed-resolution fit must be at least 5 % better on the held coarse sample. MEASURED 2026-11-15: mixed RMS
#: 0.006282 vs fine-only 0.007012 (ratio 0.896). On this 1-D Poisson a fine-grid fit already transfers well, so the
#: gain is modest; the bar is a claim the measurement clears, not a number fitted to it.
MARGIN = 0.95


def _grid(n: int):
    p = np.linspace(0, 1, n)
    return p, mesh_graph(p[:, None], intervals(p), radius=1.5)


def _samples(n_points: int, n: int, seed: int) -> GraphSamples:
    p, graph = _grid(n_points)
    a = np.random.default_rng(seed).standard_normal((n, len(MODES)))
    basis = np.sin(np.pi * MODES[:, None] * p[None, :])
    return GraphSamples(graph, (a @ basis)[:, None], ((a / (np.pi * MODES) ** 2) @ basis)[:, None])


def _model(steps: int = 300, batch: int | None = None) -> GraphOperator:
    return GraphOperator(
        width=8,
        layers=2,
        kernel_width=8,
        schedule=Schedule(adam_steps=steps, learning_rate=3e-3, batch=batch),
        runtime=CPU64,
        margin=0.5,
    )


def _rms(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((a - b) ** 2)))


FINE, COARSE, HELD = _samples(33, 48, 1), _samples(5, 48, 2), _samples(5, 32, 3)


def test_a_fit_on_mixed_resolutions_beats_the_fine_grid_alone_on_a_held_coarse_sample() -> None:
    """THE CLAIM: same steps, same seed; the fine-only fit is the planted control it must beat by MARGIN."""
    fine_only = _model().fit([FINE]).predict(HELD.x, HELD.graph)
    mixed = _model().fit([FINE, COARSE]).predict(HELD.x, HELD.graph)
    fine_error, mixed_error = _rms(fine_only, HELD.y), _rms(mixed, HELD.y)
    assert mixed_error < MARGIN * fine_error, (mixed_error, fine_error)


def test_one_graph_reproduces_the_single_graph_fit_bit_identically() -> None:
    """Pinned from the single-graph ``fit(x, graph, y)`` before samples could span graphs (9 points, 6 samples)."""
    p, graph = _grid(9)
    a = np.random.default_rng(0).standard_normal((6, 2))
    x = (a @ np.sin(np.pi * np.arange(1, 3)[:, None] * p))[:, None]
    pinned = {
        None: ('0x1.fc139c9173026p-1', '-0x1.993eafbec2aeep+2'),
        3: ('0x1.8f05dc89c4145p+0', '-0x1.14d63931fd7efp+3'),
    }
    for batch, (loss, total) in pinned.items():
        schedule = Schedule(adam_steps=20, learning_rate=3e-3, batch=batch)
        model = GraphOperator(width=4, layers=1, kernel_width=4, schedule=schedule, runtime=CPU64)
        fitted = model.fit([GraphSamples(graph, x, -x)])
        assert (fitted.history[-1].hex(), float(fitted.predict(x, graph).sum()).hex()) == (loss, total)


def _counted(monkeypatch) -> list[int]:
    seen: list[int] = []
    forward = operator._GraphOperatorNet.forward

    def counting(self, x, *structure):
        seen.append(x.shape[0])
        return forward(self, x, *structure)

    monkeypatch.setattr(operator._GraphOperatorNet, 'forward', counting)
    return seen


def test_a_batch_draws_the_declared_count_across_graphs(monkeypatch) -> None:
    seen = _counted(monkeypatch)
    _model(steps=1, batch=7).fit([_samples(9, 4, 5), _samples(5, 4, 6)])
    assert sum(seen) == 7 and len(seen) == 2, seen


def test_a_batch_larger_than_the_samples_takes_every_sample_once(monkeypatch) -> None:
    """PLANTED CONTROL for the count: with fewer samples than declared the draw must report fewer, not 50."""
    seen = _counted(monkeypatch)
    _model(steps=1, batch=50).fit([_samples(9, 4, 5), _samples(5, 3, 6)])
    assert seen == [4, 3], seen


def test_mismatched_input_channels_across_graphs_are_refused_by_name() -> None:
    two = GraphSamples(COARSE.graph, np.concatenate([COARSE.x, COARSE.x], axis=1), COARSE.y)
    with pytest.raises(Refused, match='same input channels'):
        _model(steps=1).fit([FINE, two])


def test_no_samples_are_refused() -> None:
    with pytest.raises(Refused, match='at least one'):
        _model(steps=1).fit([])

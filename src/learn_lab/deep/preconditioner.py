"""A learned preconditioner for SPD systems of one parametric family: a learned correction to Jacobi.

The symmetric scaling ``A_hat = D^-1/2 A D^-1/2`` (unit diagonal) is factored approximately inverse:
``G A_hat G^T ~ I`` with ``G = diag(exp(g)) (I + C)``, ``C`` strictly lower on ``A``'s own sparsity pattern
-- a factorised sparse approximate inverse (FSAI). A message-passing network over the matrix graph (nodes =
unknowns, edges = off-diagonal entries ``a_hat_ij``) predicts ``g`` and ``C``; it is fitted on matrices of the
family by minimising ``||G A_hat G^T - I||_F^2`` (exactly, by block-identity probes; or on ``probes`` random
columns). The preconditioner is

    ``M(r) = D^-1/2 G^T G D^-1/2 r``  (~ ``A^-1``),

SPD for ANY network output (``G`` is triangular with a positive diagonal), so a Krylov solver converges to the
same solution with or without it: the preconditioner changes speed, never the answer. Both output heads start
at zero, so the untrained network IS Jacobi (``M = D^-1``).

:meth:`FittedPreconditioner.for_matrix` returns a numpy callable ``M(r) -> z`` (two sparse products) for any
Krylov solver that accepts one; learn-lab never imports a solver. The envelope is the box of the node and edge
features the fit saw; matrix size and pattern are free.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp
import torch
from torch import nn

from learn_lab.deep.fit import Checkpoint, Schedule, minimise
from learn_lab.deep.graph import MessagePassing
from learn_lab.deep.networks import MLP
from learn_lab.deep.runtime import Runtime, numpy
from learn_lab.errors import refuse, require_finite
from learn_lab.validity import ChannelEnvelope

__all__ = ['FittedPreconditioner', 'LearnedPreconditioner', 'Preconditioner']

#: Symmetry is judged relative to the largest entry.
_SYMMETRY = 1e-12


@dataclass(frozen=True, slots=True, eq=False)
class _System:
    """One matrix as the network sees it: scaled off-diagonal entries on directed edges ``col -> row``."""

    n: int
    scale: np.ndarray  # D^-1/2, (n,)
    edges: np.ndarray  # (2, E): source col j -> target row i, every off-diagonal entry
    values: np.ndarray  # (E,): a_hat_ij
    nodes: np.ndarray  # (n, 2): sum_j |a_hat_ij| and the off-diagonal count of the row

    @property
    def lower(self) -> np.ndarray:
        return np.flatnonzero(self.edges[1] > self.edges[0])


def _system(matrix: object, what: str) -> _System:
    a = sp.csr_matrix(matrix, dtype=float)
    if a.ndim != 2 or a.shape[0] != a.shape[1] or a.shape[0] < 1:
        refuse(f'{what} is a square matrix, got {a.shape}')
    require_finite(a.data, what)
    biggest = float(np.abs(a.data).max()) if a.nnz else 0.0
    if a.nnz and abs(a - a.T).max() > _SYMMETRY * biggest:
        refuse(f'{what} is not symmetric; a learned preconditioner is for SPD systems')
    diagonal = a.diagonal()
    if np.any(diagonal <= 0):
        refuse(f'{what} has a diagonal entry <= 0; an SPD matrix has none')
    scale = 1.0 / np.sqrt(diagonal)
    coo = a.tocoo()
    off = (coo.row != coo.col) & (coo.data != 0)
    rows, cols = coo.row[off].astype(np.int64), coo.col[off].astype(np.int64)
    values = coo.data[off] * scale[rows] * scale[cols]
    n = a.shape[0]
    nodes = np.column_stack([
        np.bincount(rows, np.abs(values), minlength=n),
        np.bincount(rows, minlength=n).astype(float),
    ])
    return _System(n, scale, np.stack([cols, rows]), values, nodes)


class _PreconditionerNet(nn.Module):
    def __init__(self, *, hidden: int, rounds: int) -> None:
        super().__init__()
        self.arguments = {'hidden': hidden, 'rounds': rounds}
        self.body = MessagePassing(2, 1, hidden, hidden=hidden, rounds=rounds)
        self.node_head = MLP([hidden, hidden, 1], activation='gelu')
        self.edge_head = MLP([2 * hidden + 1, hidden, 1], activation='gelu')
        for head in (self.node_head, self.edge_head):  # untrained IS Jacobi
            nn.init.zeros_(head[-1].weight)
            nn.init.zeros_(head[-1].bias)

    def forward(
        self, nodes: torch.Tensor, edges: torch.Tensor, values: torch.Tensor, lower: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """``(g (n,), c (L,))``: the log diagonal of ``G`` and its strictly lower entries on the *lower* edges."""
        h = self.body(nodes, edges, values[:, None])
        source, target = edges[:, lower]
        c = self.edge_head(torch.cat([h[target], h[source], values[lower, None]], dim=1))[:, 0]
        return self.node_head(h)[:, 0], c


def _g_times(g, c, source, target, v):
    """``G v`` for ``G = diag(exp(g)) (I + C)``, ``C`` on ``target > source`` edges; *v* ``(n, k)``."""
    return torch.exp(g)[:, None] * v.index_add(0, target, c[:, None] * v[source])


def _gt_times(g, c, source, target, v):
    """``G^T v``."""
    w = torch.exp(g)[:, None] * v
    return w.index_add(0, source, c[:, None] * w[target])


@dataclass(frozen=True, slots=True)
class LearnedPreconditioner:
    """``hidden`` x ``rounds`` message passing; ``probes`` random columns per fit, or ``None`` for the exact norm."""

    hidden: int = 32
    rounds: int = 4
    probes: int | None = None
    schedule: Schedule = field(default_factory=lambda: Schedule(adam_steps=1000, learning_rate=3e-3))
    runtime: Runtime = field(default_factory=Runtime)
    margin: float = 0.0

    def __post_init__(self) -> None:
        if self.hidden < 1 or self.rounds < 1:
            refuse(f'a learned preconditioner has hidden and rounds >= 1, got {self.hidden}, {self.rounds}')
        if self.probes is not None and (isinstance(self.probes, bool) or self.probes < 1):
            refuse(f'probes is a column count >= 1, or None for the exact norm; got {self.probes!r}')

    def fit(self, matrices: Sequence[object], *, checkpoint: Checkpoint | None = None) -> FittedPreconditioner:
        """Fit on SPD *matrices* (dense or scipy sparse) of one family; they may differ in size and pattern."""
        if not matrices:
            refuse('a learned preconditioner is fitted on at least one matrix')
        systems = [_system(m, f'training matrix {i}') for i, m in enumerate(matrices)]
        offsets = np.concatenate([[0], np.cumsum([s.n for s in systems])])
        nodes = np.concatenate([s.nodes for s in systems])
        values = np.concatenate([s.values for s in systems])
        edges = np.concatenate([s.edges + o for s, o in zip(systems, offsets, strict=False)], axis=1)
        rt = self.runtime
        rt.seeded()
        total = int(offsets[-1])
        if self.probes is None:  # block identity: column i is node i of every block -> the exact Frobenius norm
            w = np.zeros((total, max(s.n for s in systems)))
            for s, o in zip(systems, offsets, strict=False):
                w[o + np.arange(s.n), np.arange(s.n)] = 1.0
            wt = rt.tensor(w)
        else:
            wt = torch.randn(total, self.probes, generator=rt.generator(), device=rt.torch_device, dtype=rt.torch_dtype)
        net = _PreconditionerNet(hidden=self.hidden, rounds=self.rounds).to(rt.torch_device, rt.torch_dtype)
        nt, et, vt = rt.tensor(nodes), torch.as_tensor(edges, device=rt.torch_device), rt.tensor(values)
        lower = torch.as_tensor(np.flatnonzero(edges[1] > edges[0]), device=rt.torch_device)
        source, target = et[:, lower]

        def a_times(v: torch.Tensor) -> torch.Tensor:
            return v.index_add(0, et[1], vt[:, None] * v[et[0]])

        def loss() -> torch.Tensor:
            g, c = net(nt, et, vt, lower)
            product = _g_times(g, c, source, target, a_times(_gt_times(g, c, source, target, wt)))
            return ((product - wt) ** 2).sum() / wt.shape[0]

        history = minimise(
            net.parameters(), loss, self.schedule, checkpoint=checkpoint, config=repr(self), data=[nt, et, vt, wt]
        )
        return FittedPreconditioner(
            net=net.eval(),
            runtime=rt,
            node_envelope=ChannelEnvelope.of(nodes, margin=self.margin),
            edge_envelope=ChannelEnvelope.of(values[:, None], margin=self.margin),
            history=history,
        )


@dataclass(frozen=True, slots=True, eq=False)
class Preconditioner:
    """``M(r) = s * (G^T (G (s * r)))``: SPD, ``~ A^-1``; numpy in, numpy out. Accepts ``(n,)`` or ``(n, k)``."""

    scale: np.ndarray
    factor: sp.csr_matrix

    def __call__(self, r: np.ndarray) -> np.ndarray:
        r = np.asarray(r, float)
        s = self.scale if r.ndim == 1 else self.scale[:, None]
        return s * (self.factor.T @ (self.factor @ (s * r)))


@dataclass(frozen=True, slots=True, eq=False)
class FittedPreconditioner:
    net: _PreconditionerNet
    runtime: Runtime
    node_envelope: ChannelEnvelope
    edge_envelope: ChannelEnvelope
    history: list[float]

    def for_matrix(self, matrix: object) -> Preconditioner:
        """The preconditioner ``M(r) -> z`` of the SPD *matrix* (dense or scipy sparse)."""
        s = _system(matrix, 'a preconditioned matrix')
        self.node_envelope.require(s.nodes, 'a preconditioned matrix (node features)')
        if s.values.size:
            self.edge_envelope.require(s.values[:, None], 'a preconditioned matrix (scaled entries)')
        lower = s.lower
        rt = self.runtime
        with torch.no_grad():
            g, c = self.net(
                rt.tensor(s.nodes),
                torch.as_tensor(s.edges, device=rt.torch_device),
                rt.tensor(s.values),
                torch.as_tensor(lower, device=rt.torch_device),
            )
        g, c = numpy(g), numpy(c)
        source, target = s.edges[:, lower]
        unit = sp.identity(s.n, format='csr') + sp.csr_matrix((c, (target, source)), shape=(s.n, s.n))
        return Preconditioner(s.scale, sp.csr_matrix(sp.diags(np.exp(g)) @ unit))

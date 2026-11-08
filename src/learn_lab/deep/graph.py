"""Message-passing GNN predicting node fields on graphs given as numpy arrays; no graph library.

:class:`Graphs` is a batch of graphs in one array set: ``nodes (N, f)`` features, ``edges (2, E)`` directed
``source -> target`` indices into the batch (an undirected edge is two rows), optional ``edge_features
(E, g)``, and ``offsets (G + 1,)`` -- graph ``i`` owns nodes ``offsets[i]:offsets[i+1]``. Every edge stays
inside one graph; one that crosses is INVALID.

The network encodes nodes and edges, runs ``rounds`` residual message-passing steps (message
``phi(h_src, h_dst, e)`` summed at the target, update ``h += psi(h, sum)``), and decodes per node. Sum
aggregation and per-node weights make it permutation-equivariant: relabelling the nodes relabels the output.
The envelope is the box of node (and edge) feature values; graph size and topology are free.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
import torch
from torch import nn

from learn_lab.deep.fit import Schedule, minimise
from learn_lab.deep.networks import MLP
from learn_lab.deep.runtime import Runtime, numpy
from learn_lab.deep.standardise import Standardiser
from learn_lab.errors import refuse, require_finite
from learn_lab.validity import ChannelEnvelope

__all__ = ['FittedGraphModel', 'GraphModel', 'Graphs', 'MessagePassing']


@dataclass(frozen=True, slots=True, eq=False)
class Graphs:
    nodes: np.ndarray
    edges: np.ndarray
    edge_features: np.ndarray | None
    offsets: np.ndarray

    @classmethod
    def batch(cls, graphs: Sequence[tuple[np.ndarray, np.ndarray, np.ndarray | None]]) -> Graphs:
        """``[(nodes, edges, edge_features | None), ...]`` with per-graph edge indices -> one batch."""
        if not graphs:
            refuse('a graph batch holds at least one graph')
        sizes = [np.asarray(nodes).shape[0] for nodes, _, _ in graphs]
        offsets = np.concatenate([[0], np.cumsum(sizes)])
        featured = {features is not None for _, _, features in graphs}
        if len(featured) > 1:
            refuse('either every graph of a batch has edge features or none has')
        return cls(
            nodes=np.concatenate([np.asarray(nodes, float) for nodes, _, _ in graphs]),
            edges=np.concatenate(
                [np.asarray(e, int).reshape(2, -1) + o for (_, e, _), o in zip(graphs, offsets, strict=False)], axis=1
            ),
            edge_features=np.concatenate([np.asarray(f, float) for _, _, f in graphs]) if featured == {True} else None,
            offsets=offsets,
        )

    def require_valid(self, what: str) -> Graphs:
        nodes, edges, offsets = np.asarray(self.nodes, float), np.asarray(self.edges), np.asarray(self.offsets)
        n = nodes.shape[0] if nodes.ndim == 2 else -1
        problems = []
        if nodes.ndim != 2:
            problems.append(f'nodes are (N, f), got {nodes.shape}')
        if edges.ndim != 2 or edges.shape[0] != 2 or not np.issubdtype(edges.dtype, np.integer):
            problems.append(f'edges are integer (2, E), got {edges.dtype} {edges.shape}')
        elif edges.size and (edges.min() < 0 or edges.max() >= n):
            problems.append(f'an edge index leaves [0, {n})')
        if offsets.ndim != 1 or offsets[0] != 0 or offsets[-1] != n or np.any(np.diff(offsets) < 0):
            problems.append(f'offsets run 0 .. {n} non-decreasing, got {offsets}')
        elif not problems:
            graph_of = np.searchsorted(offsets, np.arange(n), side='right') - 1
            if np.any(graph_of[edges[0]] != graph_of[edges[1]]):
                problems.append('an edge joins two different graphs of the batch')
        if self.edge_features is not None and not problems:
            ef = np.asarray(self.edge_features)
            if ef.ndim != 2 or ef.shape[0] != edges.shape[1]:
                problems.append(f'edge features are (E, g) with E = {edges.shape[1]}, got {ef.shape}')
        if problems:
            refuse(f'{what} is invalid: ' + '; '.join(problems))
        require_finite(nodes, f'{what} node features')
        return self


class MessagePassing(nn.Module):
    def __init__(self, node_in: int, edge_in: int, out: int, *, hidden: int, rounds: int) -> None:
        super().__init__()
        self.arguments = {'node_in': node_in, 'edge_in': edge_in, 'out': out, 'hidden': hidden, 'rounds': rounds}
        self.encode = MLP([node_in, hidden, hidden], activation='gelu')
        self.encode_edge = MLP([max(edge_in, 1), hidden, hidden], activation='gelu')
        self.edge_in = edge_in
        self.message = nn.ModuleList(MLP([3 * hidden, hidden, hidden], activation='gelu') for _ in range(rounds))
        self.update = nn.ModuleList(MLP([2 * hidden, hidden, hidden], activation='gelu') for _ in range(rounds))
        self.decode = MLP([hidden, hidden, out], activation='gelu')

    def forward(self, x: torch.Tensor, edges: torch.Tensor, edge_features: torch.Tensor | None) -> torch.Tensor:
        source, target = edges
        e = edge_features if self.edge_in else x.new_ones((edges.shape[1], 1))
        h, e = self.encode(x), self.encode_edge(e)
        for message, update in zip(self.message, self.update, strict=True):
            m = message(torch.cat([h[source], h[target], e], dim=1))
            summed = torch.zeros_like(h).index_add_(0, target, m)
            h = h + update(torch.cat([h, summed], dim=1))
        return self.decode(h)


@dataclass(frozen=True, slots=True)
class GraphModel:
    hidden: int = 64
    rounds: int = 6
    schedule: Schedule = field(default_factory=lambda: Schedule(adam_steps=3000, learning_rate=2e-3))
    runtime: Runtime = field(default_factory=lambda: Runtime(dtype='float32'))
    margin: float = 0.0

    def fit(self, graphs: Graphs, targets: np.ndarray) -> FittedGraphModel:
        """``targets (N, k)``: one row per node of the batch."""
        graphs = graphs.require_valid('the training graphs')
        y = require_finite(np.asarray(targets, float), 'the node targets')
        if y.ndim != 2 or y.shape[0] != graphs.nodes.shape[0]:
            refuse(f'node targets are (N, k) with N = {graphs.nodes.shape[0]}, got {y.shape}')
        rt = self.runtime
        rt.seeded()
        sx, sy = Standardiser.of(graphs.nodes), Standardiser.of(y)
        se = None if graphs.edge_features is None else Standardiser.of(graphs.edge_features)
        edge_in = 0 if se is None else graphs.edge_features.shape[1]
        net = MessagePassing(graphs.nodes.shape[1], edge_in, y.shape[1], hidden=self.hidden, rounds=self.rounds)
        net = net.to(rt.torch_device, rt.torch_dtype)
        inputs = _tensors(rt, graphs, sx, se)
        yt = rt.tensor(sy.forward(y))

        def loss() -> torch.Tensor:
            return ((net(*inputs) - yt) ** 2).mean()

        history = minimise(net.parameters(), loss, self.schedule)
        edges = None if se is None else ChannelEnvelope.of(graphs.edge_features, margin=self.margin)
        return FittedGraphModel(
            net=net.eval(),
            runtime=rt,
            node_envelope=ChannelEnvelope.of(graphs.nodes, margin=self.margin),
            edge_envelope=edges,
            nodes=sx,
            edges=se,
            outputs=sy,
            history=history,
        )


def _tensors(rt: Runtime, graphs: Graphs, sx: Standardiser, se: Standardiser | None):
    edges = torch.as_tensor(np.asarray(graphs.edges, np.int64), device=rt.torch_device)
    ef = None if se is None else rt.tensor(se.forward(graphs.edge_features))
    return rt.tensor(sx.forward(graphs.nodes)), edges, ef


@dataclass(frozen=True, slots=True, eq=False)
class FittedGraphModel:
    net: MessagePassing
    runtime: Runtime
    node_envelope: ChannelEnvelope
    edge_envelope: ChannelEnvelope | None
    nodes: Standardiser
    edges: Standardiser | None
    outputs: Standardiser
    history: list[float]

    def predict(self, graphs: Graphs) -> np.ndarray:
        """``(N, k)``: the node field of every graph of the batch."""
        graphs = graphs.require_valid('a graph-model query')
        if (graphs.edge_features is None) != (self.edge_envelope is None):
            refuse('a graph-model query is invalid: it must carry edge features exactly when the training graphs did')
        self.node_envelope.require(graphs.nodes, 'a graph-model query (node features)')
        if self.edge_envelope is not None:
            self.edge_envelope.require(graphs.edge_features, 'a graph-model query (edge features)')
        with torch.no_grad():
            return self.outputs.back(numpy(self.net(*_tensors(self.runtime, graphs, self.nodes, self.edges))))

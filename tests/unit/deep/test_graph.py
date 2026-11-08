"""A message-passing GNN learns the planted solution of (L + I) u = b on random graphs it never saw."""

import numpy as np
import pytest

from learn_lab.deep.fit import Schedule
from learn_lab.deep.graph import GraphModel, Graphs
from learn_lab.errors import Refused

BAR = 0.05  # relative L2 over the held-out graphs' nodes, dimensionless


def _graph(rng):
    """A random geometric graph: 30-50 points in the unit square, each joined to its 4 nearest (both ways)."""
    n = int(rng.integers(30, 51))
    p = rng.uniform(0, 1, (n, 2))
    d = np.linalg.norm(p[:, None] - p[None], axis=2)
    near = np.argsort(d, axis=1)[:, 1:5]
    pairs = {(i, int(j)) for i in range(n) for j in near[i]}
    pairs |= {(j, i) for i, j in pairs}
    edges = np.array(sorted(pairs)).T
    laplacian = np.zeros((n, n))
    np.add.at(laplacian, (edges[0], edges[1]), -1.0)
    laplacian[np.diag_indices(n)] = -laplacian.sum(1)
    b = rng.uniform(-1, 1, n)
    return b[:, None], edges, np.linalg.solve(laplacian + np.eye(n), b)[:, None]


def _set(count, seed):
    rng = np.random.default_rng(seed)
    triples = [_graph(rng) for _ in range(count)]
    graphs = Graphs.batch([(nodes, edges, None) for nodes, edges, _ in triples])
    return graphs, np.concatenate([u for _, _, u in triples])


def _relative_l2(p, y):
    return float(np.linalg.norm(p - y) / np.linalg.norm(y))


@pytest.fixture(scope='module')
def fitted():
    graphs, u = _set(200, 0)
    return GraphModel(rounds=4, schedule=Schedule(adam_steps=1500, learning_rate=2e-3)).fit(graphs, u)


def test_the_gnn_solves_the_graph_system_on_held_out_graphs(fitted):
    graphs, u = _set(40, 1)
    error = _relative_l2(fitted.predict(graphs), u)
    assert error < BAR
    # PLANTED CONTROL: one training step cannot pass the bar.
    train, u_train = _set(200, 0)
    untrained = GraphModel(schedule=Schedule(adam_steps=1)).fit(train, u_train)
    assert _relative_l2(untrained.predict(graphs), u) > BAR


def _permuted(graphs, order, *, remap_edges):
    inverse = np.argsort(order)
    edges = inverse[graphs.edges] if remap_edges else graphs.edges
    return Graphs(graphs.nodes[order], edges, graphs.edge_features, graphs.offsets)


def test_relabelling_the_nodes_relabels_the_prediction(fitted):
    graphs, _ = _set(1, 2)
    order = np.random.default_rng(3).permutation(graphs.nodes.shape[0])
    base = fitted.predict(graphs)
    np.testing.assert_allclose(
        fitted.predict(_permuted(graphs, order, remap_edges=True)), base[order], rtol=0, atol=1e-5
    )  # unit of u; float32 summation order
    # PLANTED CONTROL: shuffling the features without moving the edges is a different graph.
    assert np.max(np.abs(fitted.predict(_permuted(graphs, order, remap_edges=False)) - base[order])) > 1e-2


def test_an_edge_between_two_graphs_is_invalid_and_a_strong_source_out_of_domain(fitted):
    graphs, _ = _set(2, 4)
    crossing = graphs.edges.copy()
    crossing[1, 0] = graphs.nodes.shape[0] - 1
    with pytest.raises(Refused, match='invalid'):
        fitted.predict(Graphs(graphs.nodes, crossing, None, graphs.offsets))
    with pytest.raises(Refused, match='out_of_domain'):
        fitted.predict(Graphs(5 * graphs.nodes, graphs.edges, None, graphs.offsets))

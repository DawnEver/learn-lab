"""The mesh-to-graph encoder: edges by radius, neighbours or cells, and weights that integrate on any point set.

The integral check is the encoder's whole purpose: ``sum_j weight_ij v(x_j)`` over a full-radius graph is the
integral of ``v`` over the domain at EVERY node, on a coarse and on a fine point set alike. PLANTED CONTROL:
plain counting weights (1 per edge) grow with the point count and miss it.
"""

import numpy as np
import pytest

from learn_lab.errors import Refused
from learn_lab.representation import intervals, mesh_graph


def _line(n: int, seed: int) -> np.ndarray:
    return np.concatenate([[0.0], np.sort(np.random.default_rng(seed).uniform(0, 1, n - 2)), [1.0]])


def _integral_at_every_node(graph, v: np.ndarray) -> np.ndarray:
    return np.bincount(graph.edges[1], graph.weights * v[graph.edges[0]], minlength=len(graph.points))


@pytest.mark.parametrize('n', [20, 200])
def test_cell_weights_integrate_on_any_point_set(n) -> None:
    x = _line(n, n)
    graph = mesh_graph(x[:, None], intervals(x), radius=2.0)
    v = x**2  # integral 1/3 (unit: v's unit times length)
    assert np.allclose(_integral_at_every_node(graph, v), 1 / 3, rtol=0, atol=2.0 / n)
    counting = np.bincount(graph.edges[1], v[graph.edges[0]])
    assert not np.allclose(counting, 1 / 3, rtol=0, atol=2.0 / n)  # PLANTED CONTROL


def test_triangle_weights_sum_to_the_area() -> None:
    points = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], float)
    graph = mesh_graph(points, [[0, 1, 2], [0, 2, 3]], radius=2.0)
    assert np.allclose(_integral_at_every_node(graph, np.ones(4)), 1.0, rtol=0, atol=1e-12)


def test_cell_edges_join_exactly_the_vertices_that_share_a_cell() -> None:
    x = np.array([0.0, 0.5, 1.0])
    graph = mesh_graph(x[:, None], intervals(x))
    pairs = set(map(tuple, graph.edges.T))
    assert pairs == {(0, 0), (1, 1), (2, 2), (0, 1), (1, 0), (1, 2), (2, 1)}


def test_radius_and_neighbour_edges_with_mean_weights() -> None:
    points = np.random.default_rng(0).uniform(0, 1, (30, 2))
    graph = mesh_graph(points, radius=0.3)
    lengths = graph.edge_features[:, -1]
    assert lengths.max() <= 0.3
    assert np.allclose(np.bincount(graph.edges[1], graph.weights), 1.0)  # the neighbourhood mean
    knn = mesh_graph(points, neighbours=5)
    assert np.array_equal(np.bincount(knn.edges[1]), np.full(30, 5))
    assert graph.edge_features.shape == (graph.edges.shape[1], 3)  # offset (2) and its length


def test_an_encoder_without_a_rule_for_edges_is_refused() -> None:
    with pytest.raises(Refused, match='radius'):
        mesh_graph(np.zeros((3, 1)))
    with pytest.raises(Refused, match='not both'):
        mesh_graph(np.zeros((3, 1)), radius=1.0, neighbours=2)

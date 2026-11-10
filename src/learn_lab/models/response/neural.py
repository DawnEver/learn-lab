"""D54 -- the multilayer perceptron: dense hidden layers, a linear output, full-batch L-BFGS training.

The loss is the mean squared error over every point and output, halved, plus the l2 penalty
``alpha / (2 n) sum |W|^2`` on the weights (not the biases). The weights start Glorot-uniform from
``seed`` and are trained by numerics-lab's ``minimize_lbfgs`` on the exact
back-propagated gradient. The mean's gradient in the inputs is exact (forward-mode through the layers).
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Final

import numpy as np
from numerics_lab.minimize import minimize_lbfgs

from learn_lab.errors import refuse
from learn_lab.protocol import Give, Prediction, query_points, require_gives, training_data
from learn_lab.validity import BoxEnvelope

__all__ = ['ACTIVATIONS', 'MultilayerPerceptron', 'Network']

#: The declared hidden-layer activations.
ACTIVATIONS: Final = ('identity', 'logistic', 'tanh', 'relu')


def _activate(name: str, z: np.ndarray) -> np.ndarray:
    if name == 'tanh':
        return np.tanh(z)
    if name == 'logistic':
        return 1 / (1 + np.exp(-z))
    if name == 'relu':
        return np.maximum(z, 0.0)
    return z


def _slope(name: str, a: np.ndarray) -> np.ndarray:
    """The activation's derivative, written in its OUTPUT ``a``."""
    if name == 'tanh':
        return 1 - a**2
    if name == 'logistic':
        return a * (1 - a)
    if name == 'relu':
        return (a > 0).astype(float)
    return np.ones_like(a)


@dataclass(frozen=True, slots=True, eq=False)
class Network:
    """The *envelope* it answers in, layer weights ``W_l (in, out)`` and biases ``b_l (out,)``; hidden layers use
    *activation*, the output none.
    """

    envelope: BoxEnvelope
    weights: tuple[np.ndarray, ...]
    biases: tuple[np.ndarray, ...]
    activation: str = 'tanh'

    def __post_init__(self) -> None:
        if self.activation not in ACTIVATIONS:
            refuse(f'an activation is one of {list(ACTIVATIONS)}, got {self.activation!r}')

    @property
    def gives(self) -> frozenset[Give]:
        return frozenset({Give.GRADIENT})

    @property
    def flat(self) -> np.ndarray:
        """Every weight then every bias, layer by layer, as one vector."""
        return np.concatenate([p.ravel() for pair in zip(self.weights, self.biases, strict=True) for p in pair])

    def with_flat(self, flat: np.ndarray) -> Network:
        """The same architecture holding the parameters *flat* (the order of :attr:`flat`)."""
        weights, biases, at = [], [], 0
        for w, b in zip(self.weights, self.biases, strict=True):
            weights.append(flat[at : at + w.size].reshape(w.shape))
            at += w.size
            biases.append(flat[at : at + b.size])
            at += b.size
        return Network(self.envelope, tuple(weights), tuple(biases), self.activation)

    def _forward(self, x: np.ndarray) -> list[np.ndarray]:
        layers = [x]
        for i, (w, b) in enumerate(zip(self.weights, self.biases, strict=True)):
            z = layers[-1] @ w + b
            layers.append(z if i == len(self.weights) - 1 else _activate(self.activation, z))
        return layers

    def loss(self, x: np.ndarray, y: np.ndarray, *, alpha: float) -> tuple[float, np.ndarray]:
        """The training loss (see the module docstring) and its gradient in :attr:`flat` order."""
        n = len(x)
        layers = self._forward(x)
        error = layers[-1] - y
        value = 0.5 * np.mean(error**2) + 0.5 * alpha * sum(np.sum(w**2) for w in self.weights) / n
        delta = error / error.size
        grads: list[np.ndarray] = []
        for i in range(len(self.weights) - 1, -1, -1):
            grads[:0] = [layers[i].T @ delta + alpha * self.weights[i] / n, delta.sum(axis=0)]
            if i:
                delta = (delta @ self.weights[i].T) * _slope(self.activation, layers[i])
        return float(value), np.concatenate([g.ravel() for g in grads])

    def predict(self, x, *, variance: bool = False, gradient: bool = False) -> Prediction:
        require_gives(self.gives, variance=variance, gradient=gradient, what='a multilayer perceptron')
        q = query_points(x, self.envelope)
        layers = self._forward(q)
        if not gradient:
            return Prediction(layers[-1])
        jacobian = np.broadcast_to(np.eye(q.shape[1]), (len(q), q.shape[1], q.shape[1]))
        for i, w in enumerate(self.weights):
            jacobian = np.einsum('hj,mhd->mjd', w, jacobian)
            if i < len(self.weights) - 1:
                jacobian = _slope(self.activation, layers[i + 1])[:, :, None] * jacobian
        return Prediction(layers[-1], gradient=jacobian)


@dataclass(frozen=True, slots=True)
class MultilayerPerceptron:
    """*hidden* layer widths, their *activation*, l2 penalty *alpha*, L-BFGS *max_iterations*, start *seed*."""

    hidden: tuple[int, ...] = (32,)
    activation: str = 'tanh'
    alpha: float = 1e-4
    max_iterations: int = 2000
    seed: int = 0

    def __post_init__(self) -> None:
        if not self.hidden or any(not isinstance(h, int) or h < 1 for h in self.hidden):
            refuse(f'hidden is a non-empty tuple of widths >= 1, got {self.hidden!r}')
        if self.activation not in ACTIVATIONS:
            refuse(f'an activation is one of {list(ACTIVATIONS)}, got {self.activation!r}')
        if self.alpha < 0:
            refuse(f'alpha is >= 0, got {self.alpha!r}')

    @property
    def gives(self) -> frozenset[Give]:
        return frozenset({Give.GRADIENT})

    def initial(self, envelope: BoxEnvelope, k: int) -> Network:
        """The Glorot-uniform starting network for the inputs of *envelope* and *k* outputs."""
        rng = np.random.default_rng(self.seed)
        sizes = (envelope.width, *self.hidden, k)
        factor = 2.0 if self.activation == 'logistic' else 6.0
        weights, biases = [], []
        for fan_in, fan_out in itertools.pairwise(sizes):
            bound = np.sqrt(factor / (fan_in + fan_out))
            weights.append(rng.uniform(-bound, bound, (fan_in, fan_out)))
            biases.append(rng.uniform(-bound, bound, fan_out))
        return Network(envelope, tuple(weights), tuple(biases), self.activation)

    def fit(self, x, y, *, domain: BoxEnvelope | None = None) -> Network:
        x, y, envelope = training_data(x, y, domain)
        start = self.initial(envelope, y.shape[1])
        trained = minimize_lbfgs(
            lambda flat: start.with_flat(flat).loss(x, y, alpha=self.alpha),
            start.flat,
            rtol=0.0,
            atol=1e-10,
            max_iter=self.max_iterations,
        )
        return start.with_flat(trained.x)

"""The gradient seam: a planted numpy Newton solve inside a torch loss, its adjoint as the VJP."""

import numpy as np
import pytest
import torch

from learn_lab.deep.bridge import check_vjp, classical_op
from learn_lab.deep.fit import Schedule, minimise
from learn_lab.deep.networks import mlp
from learn_lab.deep.runtime import Runtime
from learn_lab.errors import Refused

N = 4
A = np.array([[4.0, -1.0, 0.0, 0.0], [-1.0, 4.0, -1.0, 0.0], [0.0, -1.0, 4.0, -1.0], [0.0, 0.0, -1.0, 4.0]])
A[0, 3] = 0.5  # unsymmetric, so a VJP that forgot the transpose is wrong too


def _jacobian(y):
    return A + np.diag(3.0 * y**2)


def solve(x):
    """``y`` with ``A y + y^3 = x`` by Newton: the classical solver the bridge wraps."""
    y = np.zeros_like(x)
    for _ in range(50):
        step = np.linalg.solve(_jacobian(y), x - A @ y - y**3)
        y = y + step
        if np.max(np.abs(step)) < 1e-14:
            return y
    msg = 'Newton did not converge'
    raise AssertionError(msg)


def adjoint(_x, y, cotangent):
    """``dL/dx = J^-T dL/dy``: differentiating ``F(y) = x`` gives ``J dy = dx``."""
    return np.linalg.solve(_jacobian(y).T, cotangent)


def wrong_sign(x, y, cotangent):
    return -adjoint(x, y, cotangent)


def untransposed(_x, y, cotangent):
    return np.linalg.solve(_jacobian(y), cotangent)


X = np.array([1.0, -2.0, 0.5, 3.0])


def test_the_adjoint_passes_and_a_sign_error_or_a_missing_transpose_is_caught():
    assert check_vjp(solve, adjoint, X) < 1e-6  # relative, in the gradient's norm
    # PLANTED CONTROLS: the check can say no.
    with pytest.raises(Refused, match='vjp'):
        check_vjp(solve, wrong_sign, X)
    with pytest.raises(Refused, match='vjp'):
        check_vjp(solve, untransposed, X)


def test_the_gradient_through_the_bridge_matches_finite_differences_of_the_whole_loss():
    op = classical_op(solve, adjoint)
    rt = Runtime()

    def total(x):
        return torch.sin(op(x)).sum() + (op(x) ** 2).sum()

    x = rt.tensor(X, grad=True)
    (grad,) = torch.autograd.grad(total(x), x)
    h, fd = 1e-6, np.zeros(N)
    for i in range(N):
        e = np.eye(N)[i] * h
        fd[i] = (float(total(rt.tensor(X + e))) - float(total(rt.tensor(X - e)))) / (2 * h)
    got = grad.cpu().numpy()
    assert grad.device == x.device and grad.dtype == torch.float64
    assert np.linalg.norm(got - fd) / np.linalg.norm(fd) < 1e-6  # relative; dimensionless


def test_a_network_upstream_of_the_classical_op_trains_end_to_end():
    rt = Runtime()
    rt.seeded()
    op = classical_op(solve, adjoint)
    z = rt.tensor(np.linspace(-1, 1, 8)[:, None])
    # The target: what the solver returns for a planted input map x(z) = (z, 2z, -z, z^2) * 2.
    zn = z.cpu().numpy()
    planted = (zn * [1, 2, -1, 0] + zn**2 * [0, 0, 0, 1]) * 2
    target = rt.tensor(np.stack([solve(row) for row in planted]))
    net = mlp([1, 16, N]).to(rt.torch_device, rt.torch_dtype)

    def loss():
        return ((torch.stack([op(row) for row in net(z)]) - target) ** 2).mean()

    history = minimise(net.parameters(), loss, Schedule(adam_steps=300, learning_rate=1e-2))
    assert history[-1] < 1e-2 * history[0]  # dimensionless: two decades; measured 1.6e-3

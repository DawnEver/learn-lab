"""The one training loop: Adam, then optional L-BFGS polish, on a loss closure; the history is the record."""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable
from dataclasses import dataclass

import torch

from learn_lab.errors import refuse

__all__ = ['Schedule', 'minimise']


@dataclass(frozen=True, slots=True)
class Schedule:
    """``adam_steps`` at ``learning_rate``, then ``lbfgs_steps`` L-BFGS iterations (0 skips the polish)."""

    adam_steps: int = 2000
    learning_rate: float = 1e-3
    lbfgs_steps: int = 0

    def __post_init__(self) -> None:
        if self.adam_steps < 0 or self.lbfgs_steps < 0 or self.adam_steps + self.lbfgs_steps == 0:
            refuse(f'a schedule takes at least one step, got adam={self.adam_steps} lbfgs={self.lbfgs_steps}')
        if not self.learning_rate > 0:
            refuse(f'a learning rate is > 0, got {self.learning_rate}')


def minimise(parameters: Iterable[torch.Tensor], loss: Callable[[], torch.Tensor], schedule: Schedule) -> list[float]:
    """Minimise ``loss()`` over *parameters*; return the loss after every step. A non-finite loss is refused."""
    parameters = list(parameters)
    history: list[float] = []

    def record(value: torch.Tensor) -> None:
        number = float(value.detach())
        if not math.isfinite(number):
            refuse(f'the training loss became {number} at step {len(history)}')
        history.append(number)

    adam = torch.optim.Adam(parameters, lr=schedule.learning_rate)
    for _ in range(schedule.adam_steps):
        adam.zero_grad(set_to_none=True)
        value = loss()
        value.backward()
        adam.step()
        record(value)
    if schedule.lbfgs_steps:
        lbfgs = torch.optim.LBFGS(
            parameters,
            max_iter=schedule.lbfgs_steps,
            line_search_fn='strong_wolfe',
            tolerance_grad=0.0,
            tolerance_change=0.0,
            history_size=50,
        )

        def closure() -> torch.Tensor:
            lbfgs.zero_grad(set_to_none=True)
            value = loss()
            value.backward()
            return value

        lbfgs.step(closure)
        # Not under no_grad: a physics loss differentiates the field wrt its inputs to form the residual.
        record(loss())
    return history

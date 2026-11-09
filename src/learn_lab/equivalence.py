"""Is a candidate model as good as its reference on the model's OWN acceptance statistic? Numpy only.

A reduced-precision or exported model (int8, f16, ONNX, OpenVINO) is admitted PER MODEL, judged against its f32
oracle: :func:`equivalence` evaluates both on the same inputs, scores each with the model's acceptance
:class:`Statistic` (its own unit -- a relative L2 is dimensionless, a field error is in the field's unit) and
refuses the candidate when its statistic exceeds the reference's by more than a :class:`Tolerance` in THAT unit.
A tolerance in another unit is refused: a bare number is not a bar. The report also carries the largest pointwise
deviation of the candidate from the reference, in the output's unit.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from learn_lab.errors import refuse, require_finite

__all__ = ['Equivalence', 'Statistic', 'Tolerance', 'equivalence']


@dataclass(frozen=True, slots=True)
class Statistic:
    """A model's acceptance statistic: ``measure(outputs, inputs) -> float`` in ``unit``; lower is better."""

    name: str
    unit: str
    measure: Callable[[np.ndarray, np.ndarray], float]


@dataclass(frozen=True, slots=True)
class Tolerance:
    """How much worse than the reference the candidate's statistic may be, in the statistic's unit."""

    value: float
    unit: str

    def __post_init__(self) -> None:
        if not np.isfinite(self.value) or self.value < 0:
            refuse(f'a tolerance is a finite non-negative amount, got {self.value!r} {self.unit}')
        if not self.unit.strip():
            refuse('a tolerance carries a unit; a bare number is not a bar')


@dataclass(frozen=True, slots=True)
class Equivalence:
    """The candidate's statistic beside the reference's; ``max_deviation`` is in the output's unit."""

    statistic: str
    unit: str
    reference: float
    candidate: float
    tolerance: float
    max_deviation: float


def equivalence(
    reference_fn: Callable[[np.ndarray], np.ndarray],
    candidate_fn: Callable[[np.ndarray], np.ndarray],
    inputs: np.ndarray,
    *,
    statistic: Statistic,
    tolerance: Tolerance,
) -> Equivalence:
    """The report of *candidate_fn* against *reference_fn* on *inputs*; refused when it exceeds *tolerance*."""
    if tolerance.unit != statistic.unit:
        refuse(f'{statistic.name} is in {statistic.unit!r}; a tolerance in {tolerance.unit!r} cannot bound it')
    inputs = np.asarray(inputs)
    reference = require_finite(np.asarray(reference_fn(inputs), float), 'the reference outputs')
    candidate = require_finite(np.asarray(candidate_fn(inputs), float), 'the candidate outputs')
    if candidate.shape != reference.shape:
        refuse(f'the candidate answers {candidate.shape}, the reference {reference.shape}')
    report = Equivalence(
        statistic=statistic.name,
        unit=statistic.unit,
        reference=float(statistic.measure(reference, inputs)),
        candidate=float(statistic.measure(candidate, inputs)),
        tolerance=tolerance.value,
        max_deviation=float(np.max(np.abs(candidate - reference))),
    )
    if not report.candidate <= report.reference + tolerance.value:
        refuse(
            f'the candidate {statistic.name} is {report.candidate:.6g} {statistic.unit}, the reference '
            f'{report.reference:.6g}: more than {tolerance.value:g} {statistic.unit} worse'
        )
    return report

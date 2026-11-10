"""What a model family tunes: a :class:`Space` of named dimensions, and the :class:`Family` that declares one.

A dimension is :class:`Real` (optionally on a log scale), :class:`Integer` or :class:`Choice`; each is checked
when built, so a space that cannot be searched never exists. The search over a space is the optimiser's
(optimi-lab); learn-lab states what is searched (:mod:`learn_lab.registry`) and how a point scores
(:mod:`learn_lab.tuning`).
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from numbers import Integral
from numbers import Real as _Number
from types import MappingProxyType

from learn_lab.errors import refuse

__all__ = ['Choice', 'Dimension', 'Family', 'Integer', 'Real', 'Space']


def _number(value: object) -> bool:
    return isinstance(value, _Number) and not isinstance(value, bool)


@dataclass(frozen=True, slots=True)
class Real:
    """A real setting in ``[lo, hi]``, searched on a log scale when *log* (then ``lo > 0``)."""

    lo: float
    hi: float
    log: bool = False

    def __post_init__(self) -> None:
        if not (_number(self.lo) and _number(self.hi) and self.lo < self.hi):
            refuse(f'a real dimension needs numbers lo < hi, got [{self.lo!r}, {self.hi!r}]')
        if self.log and not self.lo > 0:
            refuse(f'a log-scaled dimension needs lo > 0, got {self.lo!r}')

    def holds(self, value: object) -> bool:
        return _number(value) and self.lo <= value <= self.hi

    @property
    def lowest(self) -> float:
        return self.lo


@dataclass(frozen=True, slots=True)
class Integer:
    """An integer setting in ``[lo, hi]``."""

    lo: int
    hi: int

    def __post_init__(self) -> None:
        if (
            not all(isinstance(b, Integral) and not isinstance(b, bool) for b in (self.lo, self.hi))
            or self.lo >= self.hi
        ):
            refuse(f'an integer dimension needs integers lo < hi, got [{self.lo!r}, {self.hi!r}]')

    def holds(self, value: object) -> bool:
        return isinstance(value, Integral) and not isinstance(value, bool) and self.lo <= value <= self.hi

    @property
    def lowest(self) -> int:
        return self.lo


@dataclass(frozen=True, slots=True)
class Choice:
    """One of the declared *values* (any objects; equality decides membership)."""

    values: tuple[object, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, 'values', tuple(self.values))
        if not self.values:
            refuse('a choice dimension declares at least one value')

    def holds(self, value: object) -> bool:
        return any(type(value) is type(v) and value == v for v in self.values)

    @property
    def lowest(self) -> object:
        return self.values[0]


Dimension = Real | Integer | Choice


class Space(Mapping[str, Dimension]):
    """A frozen mapping from setting name to its :data:`Dimension`."""

    __slots__ = ('_dimensions',)

    def __init__(self, dimensions: Mapping[str, Dimension]) -> None:
        for name, dimension in dimensions.items():
            if not isinstance(name, str) or not name:
                refuse(f'a setting name is a non-empty string, got {name!r}')
            if not isinstance(dimension, Real | Integer | Choice):
                refuse(f'setting {name!r} is a Real, Integer or Choice, got {dimension!r}')
        self._dimensions = MappingProxyType(dict(dimensions))

    def __getitem__(self, name: str) -> Dimension:
        return self._dimensions[name]

    def __iter__(self) -> Iterator[str]:
        return iter(self._dimensions)

    def __len__(self) -> int:
        return len(self._dimensions)

    def __repr__(self) -> str:
        return f'Space({dict(self._dimensions)!r})'

    def problems(self, settings: Mapping[str, object]) -> list[str]:
        """Why *settings* are not a point of this space: every unknown name and out-of-dimension value."""
        return [
            f'{name!r} is not a tuned setting of {list(self)}'
            if name not in self
            else f'{name}={value!r} is outside {self[name]!r}'
            for name, value in settings.items()
            if name not in self or not self[name].holds(value)
        ]

    def holds(self, settings: Mapping[str, object]) -> bool:
        """Whether every named setting is declared and inside its dimension (unnamed ones keep their defaults)."""
        return not self.problems(settings)

    def lowest(self) -> dict[str, object]:
        """The corner of every dimension's lowest value (a Choice's first)."""
        return {name: dimension.lowest for name, dimension in self.items()}


@dataclass(frozen=True, slots=True)
class Family:
    """A model family by its constructor *build* and the :class:`Space` of settings it tunes by default.

    *shaped* families are deep: ``build(shapes=(x_shape, y_shape), checkpoint=..., **settings)`` returns a
    surrogate on rows flattened from sample-indexed arrays; the others are ``build(**settings)``.
    """

    build: Callable[..., object]
    space: Space
    shaped: bool = False

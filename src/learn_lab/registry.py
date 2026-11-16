"""The declared model families by name, each beside the :class:`~learn_lab.space.Space` it tunes by default.

:data:`SURROGATES` holds the L6 response surrogates; :func:`surrogate` builds one from its name and settings, or
refuses naming every declared surrogate. :func:`family` resolves any family by name, the deep ones from
:data:`learn_lab.deep.registry.DEEP` (``[torch]`` extra). An ensemble is composed from declarations, so it is
not a row; a bag is, since it has a default member (bagged regression trees) and that member can be fixed.
What is tuned is stated here; how a point scores is :mod:`learn_lab.tuning`; the search is optimi-lab's.
"""

from __future__ import annotations

import importlib
from typing import Final

from learn_lab.errors import refuse
from learn_lab.models.response.bagging import Bagging
from learn_lab.models.response.chaos import ChaosExpansion
from learn_lab.models.response.clough_tocher import CloughTocher
from learn_lab.models.response.co_kriging import CoKriging
from learn_lab.models.response.gaussian_process import GaussianProcess
from learn_lab.models.response.gradient_enhanced import GradientEnhancedKriging
from learn_lab.models.response.kernel import KERNEL_FAMILIES
from learn_lab.models.response.loss_law import LossLawResidual
from learn_lab.models.response.multi_output import MultiOutputKriging
from learn_lab.models.response.neighbours import NEIGHBOUR_WEIGHTS, NearestNeighbours
from learn_lab.models.response.neural import ACTIVATIONS, MultilayerPerceptron
from learn_lab.models.response.polynomial import Polynomial
from learn_lab.models.response.power_law import PowerLaw
from learn_lab.models.response.radial_basis import RADIAL_KERNELS, RadialBasis
from learn_lab.models.response.tree import SPLITTERS, Forest, RegressionTree
from learn_lab.space import Choice, Family, Integer, Real, Space

__all__ = ['SURROGATES', 'family', 'surrogate']

_RESTARTS: Final = Integer(0, 16)
_DEPTHS: Final = Choice((None, 2, 4, 8, 16))
_FEATURES: Final = Real(0.1, 1.0)

#: Every response surrogate: its constructor and the settings it tunes.
SURROGATES: Final = {
    'polynomial': Family(Polynomial, Space({'degree': Integer(1, 6)})),
    'radial_basis': Family(
        RadialBasis,
        Space({'kernel': Choice(RADIAL_KERNELS), 'epsilon': Real(1e-2, 1e2, log=True), 'smoothing': Real(0.0, 1.0)}),
    ),
    'gaussian_process': Family(GaussianProcess, Space({'family': Choice(KERNEL_FAMILIES), 'restarts': _RESTARTS})),
    'gradient_enhanced_kriging': Family(GradientEnhancedKriging, Space({'restarts': _RESTARTS})),
    'multi_output_kriging': Family(
        MultiOutputKriging, Space({'rank': Integer(1, 8), 'family': Choice(KERNEL_FAMILIES), 'restarts': _RESTARTS})
    ),
    'co_kriging': Family(CoKriging, Space({'family': Choice(KERNEL_FAMILIES), 'restarts': _RESTARTS})),
    'nearest_neighbours': Family(NearestNeighbours, Space({'k': Integer(1, 32), 'weights': Choice(NEIGHBOUR_WEIGHTS)})),
    'regression_tree': Family(
        RegressionTree,
        Space({
            'max_depth': _DEPTHS,
            'min_samples_split': Integer(2, 32),
            'min_samples_leaf': Integer(1, 16),
            'max_features': _FEATURES,
            'splitter': Choice(SPLITTERS),
        }),
    ),
    'forest': Family(
        Forest,
        Space({
            'n_trees': Integer(10, 500),
            'bootstrap': Choice((True, False)),
            'max_depth': _DEPTHS,
            'min_samples_leaf': Integer(1, 16),
            'max_features': _FEATURES,
            'splitter': Choice(SPLITTERS),
        }),
    ),
    'multilayer_perceptron': Family(
        MultilayerPerceptron,
        Space({
            'hidden': Choice(((8,), (32,), (64,), (32, 32), (64, 64))),
            'activation': Choice(ACTIVATIONS),
            'alpha': Real(1e-6, 1e-1, log=True),
        }),
    ),
    'chaos_expansion': Family(ChaosExpansion, Space({'degree': Integer(1, 6)})),
    # Nothing to tune: it passes through every sample; its settings steer only the nodal-gradient estimate.
    'clough_tocher': Family(CloughTocher, Space({})),
    # Nothing to tune: the exponents are fitted, and pinning one is physics, not a search dimension.
    'power_law': Family(PowerLaw, Space({})),
    # The law's exponents are textbook or fitted; the residual's base is a declaration tuned on its own row.
    'loss_law_residual': Family(LossLawResidual, Space({'fit_exponents': Choice((False, True))})),
    'bagging': Family(Bagging, Space({'n_members': Integer(2, 50), 'max_samples': Real(0.1, 1.0)})),
}

#: Where the deep rows live; imported on first use only, since torch is the optional ``[torch]`` extra
#: (NO-LAZY-IMPORT's heavy-optional exception, as :mod:`learn_lab.artifact`).
_DEEP: Final = 'learn_lab.deep.registry'


def family(name: object) -> Family:
    """The family *name* declares, response or deep; a stranger is refused naming the declared set."""
    if name in SURROGATES:
        return SURROGATES[name]
    try:
        deep = importlib.import_module(_DEEP).DEEP
    except ModuleNotFoundError:
        deep = {}
    if name not in deep:
        extra = '' if deep else ' (the deep families need the torch extra: install learn-lab[torch])'
        refuse(f'a model family is one of {[*SURROGATES, *deep]}, got {name!r}{extra}')
    return deep[name]


def surrogate(name: object, **settings: object) -> object:
    """The surrogate *name* declares, built with *settings*; a stranger is refused naming the set."""
    if name not in SURROGATES:
        refuse(f'a surrogate is one of {list(SURROGATES)}, got {name!r}')
    return SURROGATES[name].build(**settings)

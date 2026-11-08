"""The declared L6 surrogates by name, for a caller that names one in data (a TOML table, a CLI flag).

:func:`surrogate` resolves a name and its settings, or refuses naming every declared surrogate. An
ensemble is composed from declarations, so it is not itself a row.
"""

from __future__ import annotations

from typing import Final

from learn_lab.errors import refuse
from learn_lab.models.response.chaos import ChaosExpansion
from learn_lab.models.response.co_kriging import CoKriging
from learn_lab.models.response.gaussian_process import GaussianProcess
from learn_lab.models.response.gradient_enhanced import GradientEnhancedKriging
from learn_lab.models.response.multi_output import MultiOutputKriging
from learn_lab.models.response.neighbours import NearestNeighbours
from learn_lab.models.response.neural import MultilayerPerceptron
from learn_lab.models.response.polynomial import Polynomial
from learn_lab.models.response.radial_basis import RadialBasis
from learn_lab.models.response.tree import Forest, RegressionTree

__all__ = ['SURROGATES', 'surrogate']

SURROGATES: Final = {
    'polynomial': Polynomial,
    'radial_basis': RadialBasis,
    'gaussian_process': GaussianProcess,
    'gradient_enhanced_kriging': GradientEnhancedKriging,
    'multi_output_kriging': MultiOutputKriging,
    'co_kriging': CoKriging,
    'nearest_neighbours': NearestNeighbours,
    'regression_tree': RegressionTree,
    'forest': Forest,
    'multilayer_perceptron': MultilayerPerceptron,
    'chaos_expansion': ChaosExpansion,
}


def surrogate(name: object, **settings: object) -> object:
    """The surrogate *name* declares, built with *settings*; a stranger is refused naming the set."""
    if name not in SURROGATES:
        refuse(f'a surrogate is one of {list(SURROGATES)}, got {name!r}')
    return SURROGATES[name](**settings)

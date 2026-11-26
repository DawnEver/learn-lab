"""Response models ``x (n, d) -> y (n, k)``, every one self-implemented on numpy, ``scipy.special`` and numerics-lab.

A model is a frozen declaration of its settings; ``fit`` returns a fitted model whose
``predict(x, variance=, gradient=)`` returns a :class:`~learn_lab.protocol.Prediction` -- the mean ``(n, k)``,
and the latent variance ``(n, k)`` and mean gradient ``(n, k, d)`` when asked for and declared in ``gives``.

* response surfaces -- :class:`Polynomial` (+ :class:`Ridge` / :class:`Lasso`), :class:`RadialBasis`,
  :class:`ChaosExpansion`, :class:`CloughTocher` (a C1 interpolant inside its nodes' hull),
  :class:`PowerLaw` (a scaling law, fitted in log space),
  :class:`LawResidual` (any law as the mean, a learned residual on top);
* Kriging -- :class:`GaussianProcess` (kernels in :mod:`~learn_lab.models.response.kernel`, maximum likelihood,
  closed-form leave-one-out), :class:`GradientEnhancedKriging`, :class:`MultiOutputKriging` (ICM),
  :class:`CoKriging` (AR1 multi-fidelity);
* learners -- :class:`NearestNeighbors`, :class:`RegressionTree`, :class:`Forest` (random forest,
  extra trees, bagging), :class:`MultilayerPerceptron`;
* composition -- :class:`Ensemble`, :class:`Bagging`. Naming one and the settings it tunes is
  :mod:`learn_lab.registry`; scoring a setting for a search is :mod:`learn_lab.tuning`; judging one is
  :mod:`learn_lab.validation`.

scikit-learn is a TEST oracle only.
"""

from learn_lab.models.response.bagging import Bagging, FittedBagging
from learn_lab.models.response.chaos import ChaosExpansion, FittedChaos, legendre_table
from learn_lab.models.response.clough_tocher import CloughTocher, FittedCloughTocher
from learn_lab.models.response.co_kriging import CoKriging, FittedCoKriging
from learn_lab.models.response.ensemble import WEIGHTINGS, Ensemble, FittedEnsemble
from learn_lab.models.response.gaussian_process import FittedGaussianProcess, GaussianProcess, log_likelihood
from learn_lab.models.response.gradient_enhanced import FittedGradientEnhanced, GradientEnhancedKriging
from learn_lab.models.response.kernel import KERNEL_FAMILIES
from learn_lab.models.response.law_residual import FittedLawResidual, LawResidual
from learn_lab.models.response.multi_output import FittedMultiOutput, MultiOutputKriging
from learn_lab.models.response.neighbors import NEIGHBOR_WEIGHTS, FittedNeighbors, NearestNeighbors
from learn_lab.models.response.neural import ACTIVATIONS, MultilayerPerceptron, Network
from learn_lab.models.response.polynomial import FittedPolynomial, Lasso, Polynomial, Ridge
from learn_lab.models.response.power_law import FittedPowerLaw, PowerLaw
from learn_lab.models.response.radial_basis import RADIAL_KERNELS, FittedRadialBasis, RadialBasis
from learn_lab.models.response.tree import SPLITTERS, FittedForest, FittedTree, Forest, RegressionTree

__all__ = [
    'ACTIVATIONS',
    'KERNEL_FAMILIES',
    'NEIGHBOR_WEIGHTS',
    'RADIAL_KERNELS',
    'SPLITTERS',
    'WEIGHTINGS',
    'Bagging',
    'ChaosExpansion',
    'CloughTocher',
    'CoKriging',
    'Ensemble',
    'FittedBagging',
    'FittedChaos',
    'FittedCloughTocher',
    'FittedCoKriging',
    'FittedEnsemble',
    'FittedForest',
    'FittedGaussianProcess',
    'FittedGradientEnhanced',
    'FittedLawResidual',
    'FittedMultiOutput',
    'FittedNeighbors',
    'FittedPolynomial',
    'FittedPowerLaw',
    'FittedRadialBasis',
    'FittedTree',
    'Forest',
    'GaussianProcess',
    'GradientEnhancedKriging',
    'Lasso',
    'LawResidual',
    'MultiOutputKriging',
    'MultilayerPerceptron',
    'NearestNeighbors',
    'Network',
    'Polynomial',
    'PowerLaw',
    'RadialBasis',
    'RegressionTree',
    'Ridge',
    'legendre_table',
    'log_likelihood',
]

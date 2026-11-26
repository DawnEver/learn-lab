"""Retired spellings stay retired: no shim, no alias, no prose that keeps the old name alive.

THE REGISTRY is the table below -- the one place an old spelling may appear -- each with the name that
replaced it, the decision that retired it and its class. A rename is NOT backward compatible, so the scan covers the package and the tests. learn-lab
was founded 2026-11-07 with nothing retired; a rename adds its row here.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from dev_lab.dev.retirementclass import refusals

_ROOT = Path(__file__).resolve().parents[2]

#: old spelling -> (what replaced it, the decision that retired it, its class -- dev_lab.dev.retirementclass).
_AT = '2026-11-08, the artifact contract: a persisted module records the arguments it was built with'
_TUNE = '2026-11-08, user ruling: learn-lab states what is tuned; optimi-lab owns the search'
_DOMAIN = '2026-11-17, the domain boundary: learn-lab is generic; a domain law moves to its domain library'
_QUAD = "2026-11-17: quadrature rules are numerics-lab's; learn-lab converts them with Runtime.tensor"
_US = '2026-11-26, US-SPELLING-IS-CHECKED: the public names move to US spelling with no alias'
RETIRED: dict[str, tuple[str, str, str]] = {
    'networks import mlp': ('networks import MLP (a class recording its widths and activation name)', _AT, 'renamed'),
    'rate_net': (
        'FittedDynamics.net and FittedDynamics.time_scale',
        '2026-11-08: a fitted neural ODE holds its net and time span, not a closure over them',
        'renamed',
    ),
    # The private grid search left learn-lab; optimi-lab's searches minimize the Objective instead.
    'tuning import tune': (
        'tuning import objective (an Objective for optimi-lab to minimize)',
        _TUNE,
        'capability-dropped',
    ),
    'Trial(': ('Evaluation (one recorded Objective call)', _TUNE, 'renamed'),
    'loss_law': (
        'LawResidual (any law as the mean, a learned residual); the loss law itself is its domain library',
        _DOMAIN,
        'renamed',
    ),
    'TEXTBOOK_EXPONENTS': ('the exponent table of the domain library that states the law', _DOMAIN, 'renamed'),
    'learn_lab.generations': (
        'optimi_lab.analysis.generations',
        "2026-11-17: a run analysis is optimi-lab's",
        'renamed',
    ),
    'learn_lab.representation': ('learn_lab.point_graph', '2026-11-17: named for what it builds', 'renamed'),
    'physics import gauss_legendre': ('numerics_lab.quadrature.gauss_legendre_box', _QUAD, 'renamed'),
    'simplex_rule(': ('numerics_lab.quadrature.simplex_rule (numpy), then Runtime.tensor', _QUAD, 'renamed'),
    # spelling-off
    'learn_lab.deep.standardise': ('learn_lab.deep.standardize', _US, 'renamed'),
    'response.neighbours': ('learn_lab.models.response.neighbors', _US, 'renamed'),
    'nearest_neighbours': (
        'nearest_neighbors (the registry family key; nearest_neighbors_classifier likewise)',
        _US,
        'renamed',
    ),
    'NearestNeighbours': ('NearestNeighbors (NearestNeighborsClassifier likewise)', _US, 'renamed'),
    'FittedNeighbours': ('FittedNeighbors (FittedNeighborsClassifier likewise)', _US, 'renamed'),
    'NEIGHBOUR_WEIGHTS': ('NEIGHBOR_WEIGHTS', _US, 'renamed'),
    'minimise': ('minimize', _US, 'renamed'),
    'maximise_likelihood': ('maximize_likelihood', _US, 'renamed'),
    # spelling-on
}


def live_spellings(text: str, retired: Mapping[str, object] = RETIRED) -> set[str]:
    return {old for old in retired if old in text}


def _sources() -> list[Path]:
    files = [*(_ROOT / 'src').rglob('*.py'), *(_ROOT / 'tests').rglob('*.py')]
    return [path for path in files if path.resolve() != Path(__file__).resolve()]


def test_the_scan_sees_a_planted_retired_spelling() -> None:
    assert live_spellings('from learn_lab.old import x\nfine', {'learn_lab.old': 'learn_lab.new'}) == {'learn_lab.old'}


def test_no_source_spells_a_retired_name() -> None:
    files = _sources()
    assert len(files) > 5, f'the scan reached only {len(files)} files'
    offenders = {
        path.relative_to(_ROOT).as_posix(): found
        for path in files
        if (found := live_spellings(path.read_text(encoding='utf-8')))
    }
    assert offenders == {}


def test_every_retirement_is_classified() -> None:
    assert refusals(RETIRED) == []


def test_an_unclassified_retirement_is_refused() -> None:
    assert refusals({'old': ('new', 'why')}) != []

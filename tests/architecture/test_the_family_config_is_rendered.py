"""Every family config artefact is the live base plus learn-lab's declared delta -- a hand edit reds."""

from __future__ import annotations

from pathlib import Path
from typing import Final

import pytest
from dev_lab.dev.famtests import configrender, famfiles
from tests.architecture._famconfig import DELTAS, REPO

_ROOT: Final = Path(__file__).resolve().parents[2]
_HINT: Final = 're-render: <venv python> -m dev_lab.dev.famfiles --deltas tests/architecture/_famconfig.py'

#: The project-level files whose content is learn-lab's own, each with its reason: none beyond pyproject.toml.
_OWNED_HERE: Final[dict[str, str]] = {}


def test_every_family_base_is_declared() -> None:
    configrender.assert_every_base_is_accounted_for(deltas=DELTAS, repo=REPO)


@pytest.mark.parametrize('artefact', sorted(DELTAS))
def test_the_artefact_on_disk_is_the_base_plus_our_delta(artefact: str) -> None:
    configrender.assert_artefact_is_rendered(
        root=_ROOT, artefact=artefact, deltas=DELTAS, repo=REPO, rerender_hint=_HINT
    )


@pytest.mark.parametrize('artefact', sorted(DELTAS))
def test_our_delta_is_not_a_fork(artefact: str) -> None:
    configrender.assert_delta_is_not_a_fork(artefact=artefact, deltas=DELTAS, repo=REPO)


def test_every_project_file_has_one_source() -> None:
    famfiles.assert_every_project_file_is_accounted_for(root=_ROOT, owned_here=_OWNED_HERE)

"""ONE-BRANCH-PER-SESSION, MERGE-OR-DELETE: this repo DECLARES `[tool.dev_lab.branchset]`.

The LIVE branch set is the forge's `branchset` status, never a test (ruling 2026-10-24). The declared
set is the trunk `main` plus the integration branch `integrate/main`; a
lane is local, merges into integration and goes.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

from dev_lab.dev.branchset import declared_branchset

_ROOT: Final = Path(__file__).resolve().parents[2]


def test_the_declared_set_is_the_trunk_and_integration() -> None:
    declared = declared_branchset(_ROOT)
    assert declared.trunk == 'main'
    assert set(declared.sessions) == {'integrate/main'}

"""ONLY OUR OWN IS SWEPT: every shared-temp entry this repo creates carries :data:`lab_commons.temp.TEMP_PREFIX`.

dev-lab's Temp sweep removes only entries named with that prefix, so an unprefixed one piles up forever. The
body and its planted control are :mod:`dev_lab.dev.famtests.tempprefix`; this file names this repo's facts.
"""

from __future__ import annotations

from pathlib import Path

from dev_lab.dev.famtests import tempprefix

ROOT = Path(__file__).resolve().parents[2]

#: Files the walk must read; measured 49 ``.py`` under the roots on 2026-11-11.
FLOOR = 40


def test_every_temp_entry_carries_the_family_prefix() -> None:
    roots = [ROOT / name for name in ('src', 'scripts')]
    tempprefix.assert_every_temp_entry_carries_the_prefix(base=ROOT, roots=roots, floor=FLOOR)


def test_an_unprefixed_temp_call_is_found(tmp_path: Path) -> None:
    tempprefix.assert_the_scan_finds_a_planted_call(tmp_path=tmp_path)

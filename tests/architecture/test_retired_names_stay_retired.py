"""Retired spellings stay retired: no shim, no alias, no prose that keeps the old name alive.

THE REGISTRY is the table below -- the one place an old spelling may appear -- each with the name that
replaced it. A rename is NOT backward compatible, so the scan covers the package and the tests. learn-lab
was founded 2026-11-07 with nothing retired; a rename adds its row here.
"""

from __future__ import annotations

from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]

#: old spelling -> what replaced it.
RETIRED: dict[str, str] = {}


def live_spellings(text: str, retired: dict[str, str] = RETIRED) -> set[str]:
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

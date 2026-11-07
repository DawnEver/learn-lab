"""MONTHLY-PLANS-CARRIED: every month's summary catalogues its plans, and the next month carries the open ones.

The reader and the verdict are the family's (``dev_lab.dev.famtests.monthsummary``); what is this repo's is the
memory tree it reads and the first month it holds to the rule: 2026/11, the month learn-lab was founded.
"""

from __future__ import annotations

from pathlib import Path

from dev_lab.dev.famtests.monthsummary import assert_every_month_carries_its_plans, assert_the_reader_still_convicts

_ROOT = Path(__file__).resolve().parents[2]


def test_every_month_carries_its_plans() -> None:
    assert_every_month_carries_its_plans(_ROOT / '.claude' / 'memory', since='2026/11')


def test_the_reader_still_convicts(tmp_path: Path) -> None:
    """PLANTED CONTROL: the same reader refuses a missing summary and a dropped plan."""
    assert_the_reader_still_convicts(tmp_path)

"""REVIEW-COMMAND-IS-THE-FAMILY'S: `[tool.dev_lab.review]` declares only where it differs, else is absent."""

from __future__ import annotations

from pathlib import Path

from dev_lab.dev.famtests.reviewcommand import assert_the_review_declares_only_its_difference


def test_the_review_declares_only_its_difference() -> None:
    assert_the_review_declares_only_its_difference(Path(__file__).resolve().parents[2])

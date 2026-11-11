"""The GitHub Actions workflow is RENDERED from TOML (`dev_lab.dev.ciworkflows`): no drift, no hand-written file."""

from __future__ import annotations

from pathlib import Path

from dev_lab.dev.ciworkflows import drift, workflows

ROOT = Path(__file__).resolve().parents[2]


def test_the_repo_runs_the_family_pr_lifecycle() -> None:
    # review-prepare.yml gained 2026-11-10: a judge repo renders the bundle prepare its judge reads.
    assert set(workflows(ROOT)) == {
        'ci.yml',
        'continue.yml',
        'merge-tick.yml',
        'review-prepare.yml',
    }


def test_every_workflow_matches_its_toml() -> None:
    assert drift(ROOT) == []

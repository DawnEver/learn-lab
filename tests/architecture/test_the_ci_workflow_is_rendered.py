"""The GitHub Actions workflow is RENDERED from TOML (`dev_lab.dev.ciworkflows`): no drift, no hand-written file."""

from __future__ import annotations

from pathlib import Path

from dev_lab.dev.ciworkflows import drift, workflows

ROOT = Path(__file__).resolve().parents[2]


def test_the_repo_runs_the_family_pr_lifecycle() -> None:
    # review-again.yml left 2026-11-11 with dev-lab #443 (the judge is advisory; Actions alone gate);
    # review-prepare.yml left 2026-11-21 with dev-lab 6d2cabbd (D77: the judge fetches each head and prepares
    # its own bundle).
    assert set(workflows(ROOT)) == {
        'ci.yml',
        'continue.yml',
        'merge-tick.yml',
    }


def test_every_workflow_matches_its_toml() -> None:
    assert drift(ROOT) == []

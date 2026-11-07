"""learn-lab runs the family git hooks: `[tool.dev_lab.hooks]` over `githooks/base.toml`, no row added.

Installing the shims is the developer's (`<venv python> -m dev_lab.dev.hook_install --install --repo .`);
this file pins what the merged table DECLARES, as a named set of stages and the rows the registry cites.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

from dev_lab.dev.famtests.hookinstall import assert_stages_are_declared
from dev_lab.dev.famtests.lockpin import assert_the_lock_is_pinned
from dev_lab.dev.githooks.runner import hook_table

_ROOT: Final = Path(__file__).resolve().parents[2]

# post-merge and post-checkout: dev-lab VENV-FOLLOWS-THE-LOCK (user ruling 2026-11-04).
_DECLARED_STAGES: Final = frozenset({
    'commit-msg',
    'post-checkout',
    'post-commit',
    'post-merge',
    'pre-commit',
    'pre-push',
})


def _row_module(stage: str, hook_id: str) -> str:
    rows = {hook.id: hook for hook in hook_table(_ROOT)[stage]}
    assert hook_id in rows, f'{stage} has no {hook_id!r} row: {sorted(rows)}'
    return list(rows[hook_id].run)[-1]


def test_the_declared_stages_are_the_named_set() -> None:
    assert_stages_are_declared(root=_ROOT, declared_stages=_DECLARED_STAGES)


def test_every_push_runs_the_session_branch_invariants() -> None:
    """PUBLISH-IS-THE-LOCAL-REF, SESSION-HISTORY-ONLY-GROWS, INTEGRATION-IS-MERGES-ONLY at pre-push."""
    assert _row_module('pre-push', 'session-branch') == 'dev_lab.dev.sessionbranch'


def test_this_checkout_pins_its_lock() -> None:
    """PIN, judged on the forge over HEAD: the lock left the pre-push (dev-lab toolchain plan P3)."""
    assert_the_lock_is_pinned(_ROOT)


def test_every_merge_is_audited() -> None:
    """MERGE-DEVIATIONS-NAMED: a resolution that loses a test unnamed is refused at commit."""
    assert _row_module('commit-msg', 'merge-audit') == 'dev_lab.dev.mergeaudit'


def test_every_conflicted_merge_declares_its_resolutions() -> None:
    """MERGE-RESOLUTION-DECLARED: each conflicted file is declared `<path>: <class> -- <why>` at commit."""
    assert _row_module('commit-msg', 'merge-resolution') == 'dev_lab.dev.mergeresolution'

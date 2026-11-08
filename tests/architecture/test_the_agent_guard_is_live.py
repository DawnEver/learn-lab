"""AGENT-GUARD: learn-lab's agent guard is live, ships the declared rules, and refuses each shape BY ITS rule.

The declaration is `scripts/deny_rules.py`; the bodies are `dev_lab.dev.famtests.agentguard`.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Final

import pytest
from dev_lab.dev.famtests import agentguard, injectedtext
from dev_lab.dev.hooks import DENY_RULES

_ROOT: Final = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / 'scripts'))
from deny_rules import ADOPTION  # noqa: E402

#: THE FLOOR, BY NAME: the rules learn-lab ships (adopted at founding, 2026-11-07).
_SHIPPED: Final = frozenset({
    'BARE-INTERPRETER',
    'BARE-PYTHON',
    'BARE-TEST-INVOCATION',
    'CONSENT-CODE-FROM-THE-USER',  # gained 2026-11-07 with dev-lab 83c4f29
    'FORGE-WRITE-VIA-CLI',
    'FORGE-WRITE-VIA-GH-API',
    'FORGE-WRITE-VIA-HTTP',
    'GIT-COMMIT-AMEND',
    'GIT-RESET',
    'GIT-STAGE-EVERYTHING',
    'GIT-STASH',
    'HPC-THROUGH-THE-DOOR',  # gained 2026-11-07 with dev-lab 83c4f29
    'PUSH-FORCE',
    'PUSH-NO-VERIFY',
    'PUSH-REMOTE-DELETE',
    'RECURSIVE-DELETE',  # gained 2026-10-29 with dev-lab e8b8991
    'RECURSIVE-GREP',
    'REMOTE-WRITE-THROUGH-A-DOOR',  # gained 2026-11-08 with the dev-lab relock
    'REPORTS-LIVE-IN-MEMORY',
    'STATUS-VIA-DOORS',
    'SUBAGENT-NO-PUSH',
    'TREE-HAS-A-LIVE-RUN',
    'WRITES-STAY-IN-THE-TREE',  # gained 2026-11-07 with dev-lab 83c4f29
    'WORKTREE-BASE-IS-EXPLICIT',
    'WORKTREES-STAY-INSIDE',
})

_PROSE_ABOUT_A_DENIED_SHAPE: Final = "cat > notes.md <<'MD'\npytest is denied here; use the verdict entry point.\nMD"


def test_the_guard_is_live() -> None:
    agentguard.assert_the_guard_is_live(root=_ROOT)


def test_the_committed_rules_are_the_rendered_rules() -> None:
    agentguard.assert_committed_rules_are_rendered(root=_ROOT, adoption=ADOPTION)


def test_every_rule_is_accounted_for_and_every_remedy_exists() -> None:
    agentguard.assert_every_rule_is_accounted_for(root=_ROOT, adoption=ADOPTION)


def test_every_refusal_fits_the_progressive_budget() -> None:
    injectedtext.assert_refusals_fit(DENY_RULES, ADOPTION.remedies)


def test_the_refusal_budget_still_convicts() -> None:
    injectedtext.assert_the_budget_still_convicts()


def test_the_shipped_set_is_pinned_by_name() -> None:
    agentguard.assert_the_shipped_set_is_pinned(root=_ROOT, declared=_SHIPPED)


@pytest.mark.parametrize(
    ('command', 'rule'),
    [
        ('pytest tests', 'BARE-TEST-INVOCATION'),
        ('python -m pytest -q', 'BARE-TEST-INVOCATION'),
        ('git stash', 'GIT-STASH'),
        ('git reset --hard HEAD', 'GIT-RESET'),
        ('git add .', 'GIT-STAGE-EVERYTHING'),
        ('git push origin main --force-with-lease', 'PUSH-FORCE'),
        ('git push --no-verify', 'PUSH-NO-VERIFY'),
        ('gh issue create --title x --body y', 'FORGE-WRITE-VIA-CLI'),
        ('gh api -X PATCH repos/o/r/issues/7 -f state=closed', 'FORGE-WRITE-VIA-GH-API'),
        ('curl -X POST https://f.example/api/v1/repos/o/r/issues -d @i.json', 'FORGE-WRITE-VIA-HTTP'),
    ],
)
def test_a_forbidden_shape_is_refused_by_the_rule_that_owns_it(command: str, rule: str) -> None:
    agentguard.assert_refused_by(root=_ROOT, command=command, rule=rule)


@pytest.mark.parametrize(
    'command',
    [
        './.venv/Scripts/python.exe -m dev_lab.dev.verify',
        'git push origin main',
        'git worktree add --detach .claude/worktrees/scratch 1a2b3c4',
        './.venv/Scripts/python.exe -m dev_lab.dev.forge issue create --title x --body y',
        'gh issue list --state open',
        'git add -- src/learn_lab/__init__.py',
        _PROSE_ABOUT_A_DENIED_SHAPE,
    ],
)
def test_a_sanctioned_shape_is_allowed(command: str) -> None:
    """The other side of the ratchet: a guard that refuses everything passes every arm above."""
    agentguard.assert_sanctioned(root=_ROOT, command=command)

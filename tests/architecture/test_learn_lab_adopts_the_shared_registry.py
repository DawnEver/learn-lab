"""learn-lab ADOPTS the family rules registry: every rule is enforced here by a named mechanism, or absent.

``dev_lab.dev.rules`` holds the one canonical statement of each rule; ``assert_adopted`` resolves every
rule claimed here against a TRACKED mechanism in this tree and refuses a rule neither claimed nor declared
absent. It proves the accounting, not that the mechanisms pass -- that is the suite's business.

learn-lab is a LIBRARY on the family branch flow: lanes merge into ``integrate/main``, and ``main`` moves
by ``integrate promote`` on a full forge PASS.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

from dev_lab.dev.profile import RepoProfile
from dev_lab.dev.rules import Adoption, LintRule, TestPath, assert_adopted

_ROOT: Final = Path(__file__).resolve().parents[2]

_PROFILE: Final = RepoProfile(app_name='learn_lab', package='learn_lab', root=_ROOT, lint_config=_ROOT / 'ruff.toml')

_BRANCHES: Final = TestPath('tests/architecture/test_the_branches_are_the_declared_set.py')
_HOOKS: Final = TestPath('tests/architecture/test_the_hooks_are_the_family_base.py')

_MECHANISMS: Final = {
    'PROJECT-FILES-HAVE-ONE-SOURCE': (TestPath('tests/architecture/test_the_family_config_is_rendered.py'),),
    'RETIRED-NAMES-REGISTERED': (TestPath('tests/architecture/test_retired_names_stay_retired.py'),),
    'NO-LAZY-IMPORT': (LintRule(code='PLC0415'),),
    'ONE-BRANCH-PER-SESSION': (_BRANCHES,),
    'MERGE-OR-DELETE': (_BRANCHES,),
    'MERGE-DEVIATIONS-NAMED': (_HOOKS,),
    'MERGE-RESOLUTION-DECLARED': (_HOOKS,),
    'PUBLISH-IS-THE-LOCAL-REF': (_HOOKS,),
    'SESSION-HISTORY-ONLY-GROWS': (_HOOKS,),
    'INTEGRATION-IS-MERGES-ONLY': (_HOOKS,),
    'ORIGIN-HOLDS-ONLY-DECLARED-BRANCHES': (_HOOKS,),  # the same `session-branch` pre-push row
    'LATEST-DEPENDENCIES': (_HOOKS,),
    # Both are the forge's server-side steps
    # (dev_lab.dev.forgemerge, dev_lab.dev.forgereview) over a repo whose integrator table declares
    # `pr_merges_since`; learn-lab declares it, so its PRs merge and are reviewed by the kit alone.
    'INTEGRATION-TAKES-ONLY-FORGE-MERGES': (
        TestPath('pyproject.toml'),
        TestPath('tests/architecture/test_this_repo_declares_the_one_pr_flow.py'),
    ),
    'PR-REVIEW-IS-ONE-SHOT-ADVICE': (TestPath('pyproject.toml'),),
    # The base pre-push row `pr-current` this pyproject's hooks table runs.
    'PR-HEAD-IS-CURRENT': (TestPath('pyproject.toml'),),
    # The PR-flow rows are the forge's own admission and carry
    # steps (dev_lab.dev.pradmit, dev_lab.dev.forgecarry) over the same declared integrator table.
    'ONE-SESSION-ONE-PR': (TestPath('pyproject.toml'),),
    'NO-STACKED-PRS': (TestPath('pyproject.toml'),),
    'DOCS-ONLY-CARRIES': (TestPath('pyproject.toml'),),
    # Three more PR-flow rows of the forge's own admission over
    # the same declared integrator table (a PR opens as a draft, forge writes go through its doors, no push
    # lands on a merged PR).
    'DRAFT-UNTIL-READY': (TestPath('pyproject.toml'),),
    'FORGE-DOORS-ARE-CONSENT': (TestPath('pyproject.toml'),),
    'LANDED-BRANCH-CONTINUES': (TestPath('pyproject.toml'),),
    'NO-PUSH-ONTO-A-CLOSED-PR': (TestPath('pyproject.toml'),),
    # November 2026, the founding month, is the first month this tree holds.
    'MONTHLY-PLANS-CARRIED': (TestPath('tests/architecture/test_every_month_carries_its_plans.py'),),
    # The agent guard ships in every family repo; the rendered
    # `.claude/hooks/deny-rules.json` refuses these, each by its own rule, and the test pins it live.
    **dict.fromkeys(
        (
            'AGENT-GUARD',
            # The guard refuses a write outside the checkout.
            'WRITES-STAY-IN-THE-TREE',
            'SUBAGENT-NO-PUSH',
            'WORKTREES-STAY-INSIDE',
            'REPORTS-LIVE-IN-MEMORY',
            'INJECTED-TEXT-IS-PROGRESSIVE',
            # The guard refuses a raw recursive delete, and its
            # wiring (pinned live by the same test) runs `dev_lab.dev.pushwatch` on PostToolUse.
            'RECURSIVE-DELETE',
            'PUSHED-HEAD-IS-WATCHED',
            'SESSION-HEARS-ITS-PRS',  # dev-lab #438: the agent guard's Stop gate and PR record.
        ),
        (TestPath('tests/architecture/test_the_agent_guard_is_live.py'),),
    ),
}

#: Rules with no mechanism in this tree today, BY NAME -- deriving it from ``RULES`` would silently absorb a
#: rule added upstream instead of making this repo decide. A rule leaves by gaining a mechanism above.
_ABSENT: Final = frozenset({
    # Enforced by the kit's own commands on this repo's declaration, with no test of this tree's: forge,
    # verdicts, integration and promotion.
    'VERDICT-BAR-IS-THE-INCREMENT', 'NETWORK-RETRY-THEN-REPORT', 'SHARED-CHECKOUT', 'FORGE-THROUGH-THE-DOOR',
    'VERDICT-AS-STATUS', 'ISSUE-IS-INTENT', 'AUTO-MODE-RUNS-THE-DOORS', 'VERDICTS-FROM-THE-FORGE',
    'MAIN-SESSION-PUBLISHES', 'ONE-BOX-ONE-LOCK',
    'DOWNSTREAM-SYNCS-INTEGRATION', 'PRS-TARGET-INTEGRATION', 'TRUNK-MOVES-BY-FAST-FORWARD',
    'ONE-WRITER-PER-SESSION-BRANCH',
    # The environment: shims are installed per box.
    'HOOKS-ARE-WIRED', 'INSTALL-DOOR-DELIVERS-THE-DECLARATION', 'ENV-MUTATION-THROUGH-THE-DOOR',
    # Domain and measurement rules with no subject in a domain-free learned-models library.
    'IMPLEMENT-EVERYTHING', 'BAR-IS-A-CONSTANT', 'UNITS-GO-THROUGH-PINT', 'TOLERANCE-CARRIES-A-UNIT',
    'REGISTRY-OWNS-THE-DECISION', 'PRODUCTION-ENTRY-POINT', 'UNSUPPORTED-RAISES', 'REFUSAL-NAMES-THE-REMEDY',
    # The rule HAS a subject since 2026-11-08 -- learn_lab.artifact, a persisted format -- but no case: version 1
    # is the only schema ever issued, so there is no older one to migrate, and a version this reader does not
    # know is refused (tests/unit/test_artifact.py). It gains its mechanism with the second schema: a one-time
    # migration of every stored artifact, and this repo's own test of it.
    'FORMATS-MIGRATE-ONCE',
    'NO-REFLECTION', 'PUBLIC-SURFACE-DECLARED', 'FIX-THE-CAUSE',
    # Docs and memory: no rules pages, injected docs or dated-memory guard in this repo yet.
    'DOCS-SPLIT', 'MEMORY-SHAPE', 'INJECTED-DOC-WIDTH-CEILING',
    # Code placement and scratch: one src/ and one tests/ tree, no scratch.
    'CODE-IN-CODE-ROOTS', 'SCRATCH-IS-NEVER-COMMITTED', 'STRAY-FILE-REFUSED', 'SPLIT-BY-CONCEPT',
    'TRACKED-CITES-NO-EPHEMERAL',
    # Meta-rules about guards: carried by the family's shared bodies.
    'DECLARATION-LIES', 'PLANTED-CONTROL', 'FLOOR-ON-EVERY-SCAN', 'ESCAPE-HATCH-CEILING', 'RATCHET-TWO-SIDES',
    'NAMED-SETS-NOT-COUNTS', 'XFAIL-NOT-SKIP', 'MODULE-SIZE-ALARM',
})  # fmt: skip

#: The size of the absent set at founding (2026-11-07); lowering it is the only legal edit.
_ABSENT_CEILING: Final = 46


def test_every_rule_is_enforced_or_declared_absent() -> None:
    assert_adopted(_PROFILE, Adoption(app_name='learn_lab', mechanisms=_MECHANISMS, declared_absent=_ABSENT))


def test_the_absent_set_does_not_grow() -> None:
    assert len(_ABSENT) <= _ABSENT_CEILING, sorted(_ABSENT)


def test_no_rule_is_both_enforced_and_absent() -> None:
    assert not (_ABSENT & frozenset(_MECHANISMS))

"""learn-lab's agent-guard declaration: which family deny rules ship here, and the exits they name.

Render the file the engine reads with `./.venv/Scripts/python.exe scripts/deny_rules.py`;
`tests/architecture/test_the_agent_guard_is_live.py` reds when the committed file drifts from it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

from dev_lab.dev.hook_adoption import HookAdoption, render
from dev_lab.dev.hooks import Remedy
from lab_commons.log import emit

__all__ = ['ADOPTION', 'RULES_FILE', 'main']

#: The one verdict command this repo has: a module in a declared dev dependency, not a tracked file.
VERIFY: Final = Remedy('verdict-entry-point', './.venv/Scripts/python.exe -m dev_lab.dev.verify')

#: Where the engine reads its rules from, as `dev_lab.dev.agent_guard` spells it.
RULES_FILE: Final = Path('.claude') / 'hooks' / 'deny-rules.json'

ADOPTION: Final = HookAdoption(
    app_name='learn_lab',
    remedies={'BARE-TEST-INVOCATION': VERIFY, 'PUSH-NO-VERIFY': VERIFY},
    declared_absent=frozenset({'NETWORK-VERB', 'RAW-PROCESS-KILL'}),
)


def main() -> int:
    """Render this repo's shippable rules over the committed file."""
    target = Path(__file__).resolve().parents[1] / RULES_FILE
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(render(ADOPTION), encoding='utf-8')
    emit(f'wrote {target}', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

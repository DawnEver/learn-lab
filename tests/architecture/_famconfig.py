"""learn-lab's delta against each family config base -- data, computed by nothing.

The bases live in ``dev_lab.dev._famconfig_rows``; ``python -m dev_lab.dev.famfiles --deltas
tests/architecture/_famconfig.py`` turns base plus delta into the file.
"""

from __future__ import annotations

from typing import Final

from dev_lab.dev.famconfig import Delta

__all__ = ['DELTAS', 'REPO']

REPO: Final = 'learn-lab'

DELTAS: Final[dict[str, Delta]] = {
    '.gitignore': Delta(repo=REPO, added=(), dropped={}, ceiling=0),
    '.gitattributes': Delta(repo=REPO, added=(), dropped={}, ceiling=0),
}

"""A module whose need's extra is absent COLLECTS, carrying its marker, and FAILS by name if run -- never skips.

The default environment holds no extra (``[tool.dev_lab.dep] extras = []``), yet the part matrix and the base part
collect the whole tree there: importing a torch module would be a collection error. So a module whose resource
marker names a need this environment cannot serve is collected as one stand-in item carrying that marker (its part
still derives; ``-m 'not torch'`` deselects it) whose run fails naming the extra and the door that installs it.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from tests import needs


class _Unserved(pytest.Item):
    def __init__(self, *, extras: list[str], **kwargs: object) -> None:
        super().__init__(**kwargs)
        self.extras = extras

    def runtest(self) -> None:
        flags = ' '.join(f'--extra {extra}' for extra in self.extras)
        pytest.fail(
            f'{self.path.name} needs extra {self.extras}, not installed in this environment; '
            f'add it: python -m dev_lab.dev.dep --sync {needs.ROOT} {flags}',
            pytrace=False,
        )

    def reportinfo(self) -> tuple[Path, int, str]:
        return self.path, 0, self.name


class _UnservedModule(pytest.File):
    def collect(self) -> list[pytest.Item]:
        markers = needs.declared_markers(self.path.read_text(encoding='utf-8'))
        item = _Unserved.from_parent(self, name='needs', extras=needs.missing_extras(markers))
        for marker in sorted(markers):
            item.add_marker(marker)
        return [item]


def pytest_pycollect_makemodule(module_path: Path, parent: pytest.Collector) -> pytest.Collector | None:
    if needs.missing_extras(needs.declared_markers(module_path.read_text(encoding='utf-8'))):
        return _UnservedModule.from_parent(parent, path=module_path)
    return None

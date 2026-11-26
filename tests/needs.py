"""What a test module NEEDS, read from its text: the one reading the needs guard and the collection hook share.

A test declares a need by its RESOURCE MARKER (``pytestmark = pytest.mark.torch``): its part is named for its
need-set (``lab_commons.hpc.needsets``) and ``[tool.dev_lab.dep] needs`` maps the need to the extra serving it, so
``verify --part torch`` refuses an environment without that extra by name. Which imports make a need is DERIVED
from the same declaration: each distribution of a need's extra is an import root of that need (torch, onnx,
openvino, nncf), plus :data:`PACKAGE_NEEDS` -- ``learn_lab.deep`` imports torch at module scope
(tests/architecture/test_the_layers_point_one_way.py). A need's extra may carry another (``learn-lab[torch]``), and
then its marker covers that need too, so a module never names two needs one extra serves.
"""

from __future__ import annotations

import ast
import re
import tomllib
from functools import cache
from importlib import metadata
from pathlib import Path
from typing import Final

from lab_commons.extras import declared

ROOT: Final = Path(__file__).resolve().parents[1]
#: The project itself, as a requirement of one of its own extras (``learn-lab[torch]``).
_SELF: Final = re.compile(r'^\s*learn[-_]lab\[([^\]]+)\]', re.IGNORECASE)
_NAME: Final = re.compile(r'\s*([A-Za-z0-9][A-Za-z0-9._-]*)')
#: First-party packages whose import makes a need: torch lives only under ``learn_lab.deep``.
PACKAGE_NEEDS: Final = {'learn_lab.deep': 'torch'}


@cache
def _manifest() -> dict:
    return tomllib.loads((ROOT / 'pyproject.toml').read_text(encoding='utf-8'))


def need_extras() -> dict[str, str]:
    """``[tool.dev_lab.dep] needs``: need -> the extra serving it, as :func:`lab_commons.extras.declared` reads it."""
    return dict(declared(_manifest()).needs)


def _requirements(extra: str) -> list[str]:
    return list(_manifest()['project']['optional-dependencies'][extra])


def _distributions(extra: str) -> list[str]:
    """The distributions *extra* installs, following its self-references (``learn-lab[torch]``)."""
    found: list[str] = []
    for requirement in _requirements(extra):
        if own := _SELF.match(requirement):
            found += [name for inner in own.group(1).split(',') for name in _distributions(inner.strip())]
        elif named := _NAME.match(requirement):
            found.append(named.group(1))
    return found


def import_needs() -> dict[str, str]:
    """Import root -> need: every distribution its extra directly names, and :data:`PACKAGE_NEEDS`."""
    table = dict(PACKAGE_NEEDS)
    for need, extra in need_extras().items():
        for requirement in _requirements(extra):
            if not _SELF.match(requirement) and (named := _NAME.match(requirement)):
                table[named.group(1).replace('-', '_')] = need
    return table


def covers(marker: str) -> frozenset[str]:
    """The needs a module carrying *marker* has served: its own, and each need its extra pulls in by self-reference."""
    extras = need_extras()
    if marker not in extras:
        return frozenset()
    pulled = set(_distributions(extras[marker]))
    return frozenset({marker} | {need for need, extra in extras.items() if pulled >= set(_distributions(extra))})


def _imports(tree: ast.Module) -> tuple[set[str], set[str]]:
    """Absolute module names *tree* imports anywhere, and the sibling helper modules it imports relatively."""
    absolute: set[str] = set()
    relative: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            absolute |= {alias.name for alias in node.names}
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            absolute |= {node.module, *(f'{node.module}.{alias.name}' for alias in node.names)}
        elif isinstance(node, ast.ImportFrom) and node.level == 1:
            relative |= {node.module} if node.module else {alias.name for alias in node.names}
    return absolute, relative


def needs_of_source(source: str, *, helpers: dict[str, str] | None = None) -> frozenset[str]:
    """The needs *source* imports, directly or through the sibling *helpers* (name -> source) it imports."""
    table, helpers = import_needs(), helpers or {}
    seen: set[str] = set()
    found: set[str] = set()
    pending = [source]
    while pending:
        absolute, relative = _imports(ast.parse(pending.pop()))
        for name in absolute:
            parts = name.split('.')
            found |= {table[prefix] for i in range(len(parts)) if (prefix := '.'.join(parts[: i + 1])) in table}
        for name in relative - seen:
            seen.add(name)
            if name in helpers:
                pending.append(helpers[name])
    return frozenset(found)


def needs_of_module(path: Path) -> frozenset[str]:
    """The needs test module *path* imports, following its sibling helpers."""
    helpers = {p.stem: p.read_text(encoding='utf-8') for p in path.parent.glob('*.py') if p != path}
    return needs_of_source(path.read_text(encoding='utf-8'), helpers=helpers)


def declared_markers(source: str) -> frozenset[str]:
    """The ``pytest.mark.<name>`` marks a module-scope ``pytestmark`` in *source* carries (one, or a list)."""
    marks: set[str] = set()
    for node in ast.parse(source).body:
        if isinstance(node, ast.Assign) and any(getattr(t, 'id', '') == 'pytestmark' for t in node.targets):
            values = node.value.elts if isinstance(node.value, ast.List | ast.Tuple) else [node.value]
            marks |= {value.attr for value in values if isinstance(value, ast.Attribute)}
    return frozenset(marks)


def uncovered(source: str, *, helpers: dict[str, str] | None = None) -> frozenset[str]:
    """The needs *source* imports that no resource marker it carries serves -- the guard's finding."""
    served = frozenset().union(*(covers(marker) for marker in declared_markers(source)))
    return needs_of_source(source, helpers=helpers) - served


def _installed(name: str) -> bool:
    try:
        metadata.distribution(name)
    except metadata.PackageNotFoundError:
        return False
    return True


def missing_extras(markers: frozenset[str]) -> list[str]:
    """The extras serving *markers* that this environment lacks, in marker order."""
    extras = need_extras()
    return [
        extras[marker]
        for marker in sorted(markers)
        if marker in extras and not all(_installed(name) for name in _distributions(extras[marker]))
    ]

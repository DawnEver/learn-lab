"""Every test module that imports an extra's package carries the resource marker of the need that extra serves.

torch and openvino install only where a test needs them (user ruling 2026-11-21): the default environment holds
no extra, a part is named for its need-set, and ``verify --part torch`` refuses an environment lacking the extra
by name. That holds only while every module importing torch (directly, through ``learn_lab.deep`` or a sibling
helper) or onnx/openvino/nncf says so by its marker -- an unmarked one lands in ``tests/base`` and fails there. A
missing need is refused by name, never skipped: no module ``importorskip``s a need's package.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Final

from tests import needs

_TESTS: Final = needs.ROOT / 'tests'


def _modules() -> list[Path]:
    files = sorted(_TESTS.rglob('test_*.py'))
    assert len(files) >= 40, f'the scan reached only {len(files)} test modules'
    return files


def test_the_guard_convicts_a_planted_unmarked_import() -> None:
    assert needs.uncovered('import torch\n') == {'torch'}
    assert needs.uncovered('from learn_lab.deep.field import FieldModel\n') == {'torch'}
    assert needs.uncovered('import numpy\nimport openvino as ov\n') == {'openvino'}
    assert needs.uncovered('from . import _helper\n', helpers={'_helper': 'import nncf\n'}) == {'openvino'}
    marked = 'import pytest\nimport torch\npytestmark = pytest.mark.torch\n'
    assert needs.uncovered(marked) == frozenset()
    # The openvino extra carries learn-lab[torch]: its marker serves torch; the torch marker never serves openvino.
    both = 'import pytest\nimport torch\nimport openvino\npytestmark = pytest.mark.openvino\n'
    assert needs.uncovered(both) == frozenset()
    assert needs.uncovered(both.replace('mark.openvino', 'mark.torch')) == {'openvino'}


def test_every_need_is_a_declared_resource_marker() -> None:
    manifest = needs._manifest()
    resources = set(manifest['tool']['lab_commons']['tests']['resources'])
    registered = {line.split(':')[0] for line in manifest['tool']['pytest']['ini_options']['markers']}
    assert set(needs.need_extras()) == {'torch', 'openvino'}
    assert set(needs.need_extras()) <= resources & registered


def test_every_module_importing_an_extra_carries_its_need_marker() -> None:
    found = {
        path.relative_to(_TESTS).as_posix(): sorted(bad)
        for path in _modules()
        if (
            bad := needs.needs_of_module(path)
            - frozenset().union(*map(needs.covers, needs.declared_markers(path.read_text(encoding='utf-8'))))
        )
    }
    assert not found, f'mark each with pytestmark = pytest.mark.<need>: {found}'


def test_no_module_skips_a_missing_need() -> None:
    roots = set(needs.import_needs())
    found = []
    for path in sorted(_TESTS.rglob('*.py')):
        for node in ast.walk(ast.parse(path.read_text(encoding='utf-8'))):
            call = node.func if isinstance(node, ast.Call) else None
            if getattr(call, 'attr', '') == 'importorskip' and node.args:
                name = getattr(node.args[0], 'value', '')
                if isinstance(name, str) and name.split('.')[0] in roots:
                    found.append(f'{path.relative_to(_TESTS).as_posix()}: {name}')
    assert not found, f'XFAIL-NOT-SKIP: a need is its marker, never an importorskip: {found}'

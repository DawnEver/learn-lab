"""learn-lab is a leaf library: it never imports its consumers, and torch lives only under ``learn_lab.deep``.

So do onnx, openvino and nncf: the ``[openvino]`` extra, imported only by ``learn_lab.deep.export``.

The public boundary speaks numpy; torch is the optional ``[torch]`` extra. A static scan refuses a torch import
outside ``learn_lab/deep`` and any import of a consumer (optimi_lab, motronics); a subprocess import proves
``import learn_lab`` and every non-deep module leave torch unloaded.
"""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path
from typing import Final

_ROOT: Final = Path(__file__).resolve().parents[2]
_PACKAGE: Final = _ROOT / 'src' / 'learn_lab'
_CONSUMERS: Final = frozenset({'optimi_lab', 'motronics'})
_DEEP_ONLY: Final = frozenset({'torch', 'onnx', 'openvino', 'nncf'})


def imported_roots(source: str) -> set[str]:
    """Top-level packages *source* imports absolutely."""
    found: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            found |= {alias.name.split('.')[0] for alias in node.names}
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            found.add(node.module.split('.')[0])
    return found


def offences(source: str, *, in_deep: bool) -> set[str]:
    roots = imported_roots(source)
    return (roots & _CONSUMERS) | (set() if in_deep else roots & _DEEP_ONLY)


def _modules() -> list[Path]:
    files = sorted(_PACKAGE.rglob('*.py'))
    assert len(files) >= 3, f'the scan reached only {len(files)} modules'
    return files


def _is_deep(path: Path) -> bool:
    return path.relative_to(_PACKAGE).parts[0] == 'deep'


def test_the_scan_sees_planted_offences() -> None:
    planted = 'import torch.nn as nn\nfrom optimi_lab.x import y\nfrom motronics import z\nimport numpy\n'
    assert offences(planted, in_deep=False) == {'torch', 'optimi_lab', 'motronics'}
    assert offences(planted, in_deep=True) == {'optimi_lab', 'motronics'}


def test_no_module_imports_a_consumer_or_torch_outside_deep() -> None:
    found = {
        path.relative_to(_PACKAGE).as_posix(): bad
        for path in _modules()
        if (bad := offences(path.read_text(encoding='utf-8'), in_deep=_is_deep(path)))
    }
    assert found == {}


def test_importing_the_non_deep_modules_loads_no_torch() -> None:
    names = sorted(
        '.'.join(('learn_lab', *path.relative_to(_PACKAGE).with_suffix('').parts)).removesuffix('.__init__')
        for path in _modules()
        if not _is_deep(path)
    )
    probe = f'import importlib, sys\nfor n in {names!r}: importlib.import_module(n)\nprint("torch" in sys.modules)'
    run = subprocess.run([sys.executable, '-c', probe], capture_output=True, text=True, check=True, timeout=120)
    assert run.stdout.strip() == 'False', run.stdout + run.stderr

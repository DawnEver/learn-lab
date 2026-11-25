"""The artifact's declared dataclass field sets are every learn-lab dataclass's fields, as data.

``artifact.manifest`` restores a ``None`` dataclass field (TOML has no null) from :data:`DATACLASS_FIELDS`, not by
importing the class: a deep class imports torch, and reading a manifest must not. This pin is the single source
check: the table equals what reflection over every learn-lab module finds, so a field added to a dataclass and
not to the table fails here, naming the row to write.
"""

import dataclasses
import importlib
import pkgutil

import learn_lab
from learn_lab.artifact_fields import DATACLASS_FIELDS


def _reflected() -> dict[str, tuple[str, ...]]:
    rows = {}
    for info in pkgutil.walk_packages(learn_lab.__path__, 'learn_lab.'):
        module = importlib.import_module(info.name)
        for name, cls in vars(module).items():
            if (
                isinstance(cls, type)
                and dataclasses.is_dataclass(cls)
                and cls.__module__ == info.name
                and cls.__qualname__ == name
            ):
                rows[f'{info.name}:{name}'] = tuple(f.name for f in dataclasses.fields(cls))
    return rows


def _drift(declared: dict, reflected: dict) -> dict[str, object]:
    return {
        path: reflected.get(path, 'not a dataclass any more')
        for path in sorted(set(declared) | set(reflected))
        if declared.get(path) != reflected.get(path)
    }


def test_the_comparison_sees_one_dropped_field() -> None:
    """PLANTED CONTROL: a row missing one field is drift, so the equality below can say no."""
    reflected = {'learn_lab.x:A': ('a', 'b')}
    assert _drift({'learn_lab.x:A': ('a',)}, reflected) == {'learn_lab.x:A': ('a', 'b')}
    assert _drift(reflected, reflected) == {}


def test_every_learn_lab_dataclass_declares_its_fields_as_data() -> None:
    reflected = _reflected()
    assert len(reflected) >= 100, f'the scan reached only {len(reflected)} dataclasses'
    drift = _drift(DATACLASS_FIELDS, reflected)
    assert drift == {}, f'write these rows into learn_lab/artifact_fields.py DATACLASS_FIELDS: {drift}'

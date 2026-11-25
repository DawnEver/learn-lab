"""The artifact contract: a fitted model as a versioned, verifiable directory -- saved, reloaded, audited, reused.

``save(fitted, path, declaration=, data_identity=, metrics=)`` writes ONE new directory; ``load(path)`` returns
a fitted model whose every answer equals the original's bit for bit on the same device; :func:`manifest` returns
the verified record for an audit without building the model. The directory holds:

- ``manifest.toml`` -- the format and its version, the learn-lab version and git commit that wrote it, the
  family (its :data:`FAMILIES` name), the declaration it was fitted from (its constructor settings), the input
  and output schema (a response family's inputs carry the ENVELOPE its fit answers in), the seed, the runtime,
  the caller's data identity and metrics, the fitted STATE as a tagged tree, the SHA-256 of each payload file,
  and ``content_hash`` -- the SHA-256 of the rest of itself;
- ``arrays.npz`` -- every numpy array of the fit, read back with ``allow_pickle=False``;
- ``weights.pt`` -- a deep family's torch state as one flat ``{name: tensor}`` dict, read back with
  ``weights_only=True`` (only present for a deep family; written and read by :mod:`learn_lab.deep.artifact`).

Loading executes no code the artifact carries. Nothing is unpickled. The state tree names classes, and only a
dataclass or enum defined inside ``learn_lab`` resolves; a dataclass is rebuilt through its own constructor,
so its own checks run on what was read; a torch module only from the declared table of
:mod:`learn_lab.deep.artifact`. Every byte is hashed before any payload is parsed: one changed byte is refused.

A fit that holds something else -- a caller's own callable term in an equation library -- is refused at save,
naming where it sits. The format version is checked first: :data:`VERSION` is the only one ever issued, and any
other is refused (there is no older schema to read).

Manifests are TOML (family ruling 2026-11-25), and TOML has no null. The file is a lossless projection of the
record :func:`manifest` returns, which is sealed by ``content_hash`` exactly as written: every table is DECLARED
-- the record (:data:`FIELDS`), each family's schema (:data:`SCHEMA_FIELDS`), each dataclass node
(:data:`learn_lab.artifact_fields.DATACLASS_FIELDS`, data, so reading imports no model module) -- so a ``None``
field is absent on disk and an absent declared field reads back as ``None``, while an undeclared key is refused
by name. ``PowerLaw.fixed`` (``None`` = a free exponent) is stored as the list of its FIXED cells, ``{output,
input, exponent}``, the table's shape being the schema's (outputs, inputs) widths. A ``None`` inside any other
list is refused at save, naming its path. A ``manifest.json`` is refused by name, never read
(FORMATS-MIGRATE-ONCE).
"""

from __future__ import annotations

import dataclasses
import enum
import hashlib
import importlib
import json
import shutil
import subprocess
import tomllib
from collections.abc import Callable, Mapping
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as package_version
from pathlib import Path
from types import ModuleType
from typing import Final

import numpy as np
import tomlkit

from learn_lab.artifact_fields import DATACLASS_FIELDS
from learn_lab.errors import refuse
from learn_lab.validity import BoxEnvelope, HullEnvelope

__all__ = [
    'FAMILIES',
    'FIELDS',
    'FORMAT',
    'SCHEMA_FIELDS',
    'VERSION',
    'canonical',
    'family_of',
    'load',
    'manifest',
    'save',
    'sha256',
]

FORMAT: Final = 'learn-lab-artifact'
VERSION: Final = 1
MANIFEST: Final = 'manifest.toml'
#: The retired JSON manifest: refused by name, never read (FORMATS-MIGRATE-ONCE).
RETIRED_MANIFEST: Final = 'manifest.json'
MIGRATE_DOOR: Final = (
    'fit the declaration it names again and artifact.save the fit with this learn-lab, to a new directory: an '
    'artifact manifest is manifest.toml'
)
#: Every field of the record; an absent one is ``None``, an undeclared one is refused by name.
FIELDS: Final = (
    'format',
    'version',
    'learn_lab',
    'family',
    'declaration',
    'schema',
    'seed',
    'runtime',
    'data_identity',
    'metrics',
    'state',
    'payload',
    'content_hash',
)
ARRAYS: Final = 'arrays.npz'
WEIGHTS: Final = 'weights.pt'

#: The torch half of the codec. Imported by NAME, only when a fit holds torch state: the one deferred import
#: here, because torch is the optional ``[torch]`` extra and a numpy fit must save and load without it
#: (NO-LAZY-IMPORT's heavy-optional exception; ``tests/architecture/test_the_layers_point_one_way.py``).
_TORCH_CODEC: Final = 'learn_lab.deep.artifact'

_RESPONSE: Final = 'learn_lab.models.response.'

#: Every fitted family by name: ``(declaration, fitted)`` as ``module:Class``. Response names are the
#: registry's (:data:`learn_lab.registry.SURROGATES`). Paths, not classes: the deep rows must not import torch.
FAMILIES: Final[dict[str, tuple[str, str]]] = {
    'polynomial': (f'{_RESPONSE}polynomial:Polynomial', f'{_RESPONSE}polynomial:FittedPolynomial'),
    'power_law': (f'{_RESPONSE}power_law:PowerLaw', f'{_RESPONSE}power_law:FittedPowerLaw'),
    'law_residual': (f'{_RESPONSE}law_residual:LawResidual', f'{_RESPONSE}law_residual:FittedLawResidual'),
    'radial_basis': (f'{_RESPONSE}radial_basis:RadialBasis', f'{_RESPONSE}radial_basis:FittedRadialBasis'),
    'gaussian_process': (
        f'{_RESPONSE}gaussian_process:GaussianProcess',
        f'{_RESPONSE}gaussian_process:FittedGaussianProcess',
    ),
    'gradient_enhanced_kriging': (
        f'{_RESPONSE}gradient_enhanced:GradientEnhancedKriging',
        f'{_RESPONSE}gradient_enhanced:FittedGradientEnhanced',
    ),
    'multi_output_kriging': (
        f'{_RESPONSE}multi_output:MultiOutputKriging',
        f'{_RESPONSE}multi_output:FittedMultiOutput',
    ),
    'co_kriging': (f'{_RESPONSE}co_kriging:CoKriging', f'{_RESPONSE}co_kriging:FittedCoKriging'),
    'nearest_neighbours': (f'{_RESPONSE}neighbours:NearestNeighbours', f'{_RESPONSE}neighbours:FittedNeighbours'),
    'regression_tree': (f'{_RESPONSE}tree:RegressionTree', f'{_RESPONSE}tree:FittedTree'),
    'forest': (f'{_RESPONSE}tree:Forest', f'{_RESPONSE}tree:FittedForest'),
    'multilayer_perceptron': (f'{_RESPONSE}neural:MultilayerPerceptron', f'{_RESPONSE}neural:Network'),
    'clough_tocher': (f'{_RESPONSE}clough_tocher:CloughTocher', f'{_RESPONSE}clough_tocher:FittedCloughTocher'),
    'chaos_expansion': (f'{_RESPONSE}chaos:ChaosExpansion', f'{_RESPONSE}chaos:FittedChaos'),
    'ensemble': (f'{_RESPONSE}ensemble:Ensemble', f'{_RESPONSE}ensemble:FittedEnsemble'),
    'bagging': (f'{_RESPONSE}bagging:Bagging', f'{_RESPONSE}bagging:FittedBagging'),
    'nearest_neighbours_classifier': (
        'learn_lab.models.classifier:NearestNeighboursClassifier',
        'learn_lab.models.classifier:FittedNeighboursClassifier',
    ),
    'classification_tree': (
        'learn_lab.models.classifier:ClassificationTree',
        'learn_lab.models.classifier:FittedClassificationTree',
    ),
    'logistic_regression': (
        'learn_lab.models.classifier:LogisticRegression',
        'learn_lab.models.classifier:FittedLogistic',
    ),
    'sparse_regression': ('learn_lab.equation:SparseRegression', 'learn_lab.equation:IdentifiedLaw'),
    'deep_ensemble': ('learn_lab.deep.surrogate:DeepEnsemble', 'learn_lab.deep.surrogate:FittedEnsemble'),
    'physics_informed_field': ('learn_lab.deep.physics:PhysicsInformedField', 'learn_lab.deep.physics:FittedField'),
    'monotone_law': ('learn_lab.deep.constitutive:MonotoneLaw', 'learn_lab.deep.constitutive:FittedLaw'),
    'play_hysteresis': ('learn_lab.deep.hysteresis:PlayHysteresis', 'learn_lab.deep.hysteresis:FittedHysteresis'),
    'field_model': ('learn_lab.deep.field:FieldModel', 'learn_lab.deep.field:FittedFieldModel'),
    'deep_onet': ('learn_lab.deep.operator:DeepONet', 'learn_lab.deep.operator:FittedDeepONet'),
    'fno': ('learn_lab.deep.operator:FNO', 'learn_lab.deep.operator:FittedFNO'),
    'transformer_operator': (
        'learn_lab.deep.operator:TransformerOperator',
        'learn_lab.deep.operator:FittedTransformerOperator',
    ),
    'transolver': ('learn_lab.deep.operator:Transolver', 'learn_lab.deep.operator:FittedTransolver'),
    'graph_model': ('learn_lab.deep.graph:GraphModel', 'learn_lab.deep.graph:FittedGraphModel'),
    'neural_ode': ('learn_lab.deep.dynamics:NeuralODE', 'learn_lab.deep.dynamics:FittedDynamics'),
    'diffusion_field': ('learn_lab.deep.diffusion:DiffusionField', 'learn_lab.deep.diffusion:FittedDiffusionField'),
    'pde_net': ('learn_lab.deep.pdenet:PDENet', 'learn_lab.deep.pdenet:FittedPDENet'),
    'graph_operator': ('learn_lab.deep.operator:GraphOperator', 'learn_lab.deep.operator:FittedGraphOperator'),
    'learned_preconditioner': (
        'learn_lab.deep.preconditioner:LearnedPreconditioner',
        'learn_lab.deep.preconditioner:FittedPreconditioner',
    ),
}


def _vector(envelope: BoxEnvelope, k: int) -> dict:
    """A vector family's schema: the input width and the ENVELOPE its fit answers in, and the output width."""
    inputs = {
        'kind': 'vector',
        'width': envelope.width,
        'envelope': {'lower': envelope.lower.tolist(), 'upper': envelope.upper.tolist()},
    }
    return {'inputs': inputs, 'outputs': {'kind': 'vector', 'width': int(k)}}


def _hull(envelope: HullEnvelope, k: int) -> dict:
    """A hull-bound family's schema: its envelope is the hull's facets in its affine plane, not a box."""
    hull = {
        'origin': envelope.origin.tolist(),
        'basis': envelope.basis.tolist(),
        'normals': envelope.normals.tolist(),
        'offsets': envelope.offsets.tolist(),
        'slack': envelope.slack,
    }
    inputs = {'kind': 'vector', 'width': envelope.width, 'envelope': {'hull': hull}}
    return {'inputs': inputs, 'outputs': {'kind': 'vector', 'width': int(k)}}


def _classes(envelope: BoxEnvelope, classes: np.ndarray) -> dict:
    """A classifier's schema: a vector family's inputs, and the classes it tells apart as its output."""
    return {'inputs': _vector(envelope, 0)['inputs'], 'outputs': {'kind': 'classes', 'classes': classes.tolist()}}


#: The input/output schema of each numpy family, read off its fitted state; the deep rows are the codec's.
_SCHEMAS: Final[dict[str, Callable[[object], dict]]] = {
    'polynomial': lambda f: _vector(f.envelope, f.coefficients.shape[1]),
    'power_law': lambda f: _vector(f.envelope, f.level.size),
    'law_residual': lambda f: _vector(f.envelope, f.outputs),
    'radial_basis': lambda f: _vector(f.envelope, f.weights.shape[1]),
    'gaussian_process': lambda f: _vector(f.envelope, len(f.columns)),
    'gradient_enhanced_kriging': lambda f: _vector(f.envelope, len(f.columns)),
    'multi_output_kriging': lambda f: _vector(f.envelope, f.level.size),
    'co_kriging': lambda f: _vector(f.envelope, len(f.base.columns)),
    'nearest_neighbours': lambda f: _vector(f.envelope, f.y.shape[1]),
    'regression_tree': lambda f: _vector(f.envelope, f.value.shape[1]),
    'forest': lambda f: _vector(f.envelope, f.trees[0].value.shape[1]),
    'multilayer_perceptron': lambda f: _vector(f.envelope, f.weights[-1].shape[1]),
    'clough_tocher': lambda f: _hull(f.envelope, f.values.shape[1]),
    'chaos_expansion': lambda f: _vector(f.envelope, f.coefficients.shape[1]),
    'ensemble': lambda f: _schema(family_of(f.members[0]), f.members[0]),
    'bagging': lambda f: _schema(family_of(f.members[0]), f.members[0]),
    'nearest_neighbours_classifier': lambda f: _classes(f.envelope, f.classes),
    'classification_tree': lambda f: _classes(f.envelope, f.classes),
    'logistic_regression': lambda f: _classes(f.envelope, f.classes),
    'sparse_regression': lambda f: {
        'inputs': {'kind': 'state', 'variables': list(f.library.variables)},
        'outputs': {'kind': 'rate', 'targets': list(f.targets)},
    },
}


#: A declared table: each key maps to ``None`` (a value) or to the declared table it holds. An absent key reads as
#: ``None``; a key the table does not declare is refused by name.
type Table = Mapping[str, Table | None]


def _keys(*values: str, **tables: Table) -> dict[str, Table | None]:
    return dict.fromkeys(values) | tables


_WIDTH: Final = _keys('kind', 'width')
_VECTOR_INPUTS: Final = _keys('kind', 'width', envelope=_keys('lower', 'upper'))
_HULL: Final = _keys('origin', 'basis', 'normals', 'offsets', 'slack')
_BOX: Final = _keys('kind', 'width', 'lower', 'upper')  # the deep codec's ``_box``
_CHANNELS: Final = _keys('kind', 'channels', 'grid', 'lower', 'upper')  # its ``_channels``; grid None off a grid
_CHANNELS_OUT: Final = _keys('kind', 'channels')
_ON_POINTS: Final = _keys(inputs=_keys(functions=_CHANNELS, points=_BOX), outputs=_CHANNELS_OUT)
#: The schema of an ensemble of fits is its first member's: :func:`_schema_table` reads which family that is.
_MEMBERS: Final = 'the first member family'

#: Every family's declared schema table (the numpy families' :data:`_SCHEMAS`, the deep codec's ``_SCHEMAS``),
#: as data so reading a manifest imports no model module. A save whose schema strays from it is refused by name.
SCHEMA_FIELDS: Final[dict[str, Table | str]] = {
    **dict.fromkeys(
        [
            'polynomial',
            'power_law',
            'law_residual',
            'radial_basis',
            'gaussian_process',
            'gradient_enhanced_kriging',
            'multi_output_kriging',
            'co_kriging',
            'nearest_neighbours',
            'regression_tree',
            'forest',
            'multilayer_perceptron',
            'chaos_expansion',
        ],
        _keys(inputs=_VECTOR_INPUTS, outputs=_WIDTH),
    ),
    'clough_tocher': _keys(inputs=_keys('kind', 'width', envelope=_keys(hull=_HULL)), outputs=_WIDTH),
    'ensemble': _MEMBERS,
    'bagging': _MEMBERS,
    **dict.fromkeys(
        ['nearest_neighbours_classifier', 'classification_tree', 'logistic_regression'],
        _keys(inputs=_VECTOR_INPUTS, outputs=_keys('kind', 'classes')),
    ),
    'sparse_regression': _keys(inputs=_keys('kind', 'variables'), outputs=_keys('kind', 'targets')),
    'deep_ensemble': _keys(inputs=_BOX, outputs=_WIDTH),
    'physics_informed_field': _keys(inputs=_BOX, outputs=_WIDTH),
    'monotone_law': _keys(inputs=_BOX, outputs=_keys('kind')),
    'play_hysteresis': _keys(inputs=_BOX, outputs=_keys('kind', 'plays')),
    'field_model': _keys(inputs=_CHANNELS, outputs=_CHANNELS_OUT),
    'deep_onet': _keys(inputs=_keys(sensors=_CHANNELS, points=_BOX), outputs=_keys('kind')),
    'fno': _keys(inputs=_CHANNELS, outputs=_CHANNELS_OUT),
    'transformer_operator': _ON_POINTS,
    'transolver': _ON_POINTS,
    'graph_model': _keys(inputs=_keys(nodes=_CHANNELS, edges=_CHANNELS), outputs=_CHANNELS_OUT),
    'graph_operator': _ON_POINTS,
    'learned_preconditioner': _keys(inputs=_keys('kind', nodes=_CHANNELS, entries=_CHANNELS), outputs=_keys('kind')),
    'neural_ode': _keys(inputs=_BOX, outputs=_WIDTH),
    'diffusion_field': _keys(inputs=_BOX, outputs=_keys('kind', 'channels', 'grid')),
    'pde_net': _keys(inputs=_BOX, outputs=_keys('kind', 'terms')),
}
_LEARN_LAB: Final = _keys('version', git=_keys('sha', 'dirty'))
_RUNTIME: Final = _keys('device', 'dtype', 'seed')
#: The declared keys of each tagged node of a state tree (its tag first).
_TAG_FIELDS: Final[dict[str, tuple[str, ...]]] = {
    'array': ('array',),
    'scalar': ('scalar',),
    'enum': ('enum', 'value'),
    'tuple': ('tuple',),
    'frozenset': ('frozenset',),
    'dict': ('dict',),
    'dataclass': ('dataclass', 'fields'),
    'module': ('module', 'arguments', 'state'),
}
#: The one table whose ``None`` cells are stored by omission: a free exponent (:class:`PowerLaw`).
_POWER_LAW: Final = f'{_RESPONSE}power_law:PowerLaw'
_CELL: Final = ('output', 'input', 'exponent')


def _path_of(cls: type) -> str:
    return f'{cls.__module__}:{cls.__qualname__}'


def family_of(fitted: object) -> str:
    path = _path_of(type(fitted))
    for name, (_, fitted_path) in FAMILIES.items():
        if fitted_path == path:
            return name
    return refuse(f'{path} is not a fitted family an artifact holds; the families are {list(FAMILIES)}')


def _torch_codec() -> ModuleType:
    return importlib.import_module(_TORCH_CODEC)


def _schema(family: str, fitted: object) -> dict:
    return _SCHEMAS[family](fitted) if family in _SCHEMAS else _torch_codec().schema(family, fitted)


def _holds_torch(obj: object) -> bool:
    return any(base.__module__.split('.')[0] == 'torch' for base in type(obj).__mro__)


class _Writer:
    """Encodes a value into the JSON state tree, collecting its arrays and torch state on the side."""

    def __init__(self) -> None:
        self.arrays: dict[str, np.ndarray] = {}
        self.weights: dict[str, object] = {}

    def _array(self, array: np.ndarray, where: str) -> str:
        if array.dtype.hasobject:
            refuse(f'{where} is an object array; an artifact holds numeric arrays only')
        key = f'a{len(self.arrays)}'
        self.arrays[key] = array
        return key

    def encode(self, value: object, where: str) -> object:  # noqa: PLR0911 -- one return per tag
        if isinstance(value, np.ndarray):
            return {'array': self._array(value, where)}
        if isinstance(value, np.generic):  # before float: a numpy scalar keeps its own type
            return {'scalar': self._array(np.asarray(value), where)}
        if isinstance(value, enum.Enum):  # before str: a StrEnum keeps its class
            return {'enum': self._class(type(value), where), 'value': value.value}
        if value is None or isinstance(value, bool | int | float | str):
            return value
        if isinstance(value, list):
            return [self.encode(v, f'{where}[{i}]') for i, v in enumerate(value)]
        if isinstance(value, tuple):
            return {'tuple': [self.encode(v, f'{where}[{i}]') for i, v in enumerate(value)]}
        if isinstance(value, frozenset):
            items = [self.encode(v, f'{where}{{}}') for v in value]
            return {'frozenset': sorted(items, key=lambda item: json.dumps(item, sort_keys=True))}
        if isinstance(value, Mapping):
            return {
                'dict': [[self.encode(k, f'{where} key'), self.encode(v, f'{where}[{k!r}]')] for k, v in value.items()]
            }
        if dataclasses.is_dataclass(value) and not isinstance(value, type):
            return {
                'dataclass': self._class(type(value), where),
                'fields': {
                    f.name: self.encode(getattr(value, f.name), f'{where}.{f.name}') for f in dataclasses.fields(value)
                },
            }
        if _holds_torch(value):
            return _torch_codec().encode_module(
                value, self.weights, lambda v, w=where: self.encode(v, f'{w}.arguments'), where
            )
        return refuse(f'{where} is a {type(value).__qualname__}, which an artifact cannot hold as data')

    @staticmethod
    def _class(kind: type, where: str) -> str:
        if not kind.__module__.startswith('learn_lab.'):
            refuse(f'{where} is a {_path_of(kind)}; an artifact holds learn_lab types only')
        return _path_of(kind)


def _resolve(path: object) -> type:
    """A dataclass or enum class defined inside ``learn_lab``, by ``module:Class``; anything else is refused."""
    module, _, name = str(path).partition(':')
    if not module.startswith('learn_lab.') or not name.isidentifier():
        refuse(f'an artifact names only learn_lab classes, got {path!r}')
    try:
        cls = vars(importlib.import_module(module))[name]
    except (ImportError, KeyError):
        refuse(f'{path!r} names no class of this learn-lab')
    if not isinstance(cls, type) or not (dataclasses.is_dataclass(cls) or issubclass(cls, enum.Enum)):
        refuse(f'{path!r} is neither a dataclass nor an enum; an artifact rebuilds only those')
    return cls


_TAGS: Final = ('array', 'scalar', 'enum', 'tuple', 'frozenset', 'dict', 'dataclass', 'module')


class _Reader:
    def __init__(self, arrays: Mapping[str, np.ndarray], weights: Mapping[str, object], runtime: object) -> None:
        self.arrays, self.weights, self.runtime = arrays, weights, runtime

    def _array(self, key: object) -> np.ndarray:
        if key not in self.arrays:
            refuse(f'the state names array {key!r}, which the payload does not hold')
        return self.arrays[key]

    def decode(self, node: object) -> object:  # noqa: PLR0911 -- one return per tag
        if node is None or isinstance(node, bool | int | float | str):
            return node
        if isinstance(node, list):
            return [self.decode(v) for v in node]
        if not isinstance(node, dict) or not node:
            return refuse(f'an artifact state node is a value, a list or a tagged object, got {node!r}')
        tag = next((tag for tag in _TAGS if tag in node), None)  # JSON objects are unordered: find the tag
        if tag == 'array':
            return self._array(node['array'])
        if tag == 'scalar':
            return self._array(node['scalar'])[()]
        if tag == 'enum':
            return _resolve(node['enum'])(node['value'])
        if tag == 'tuple':
            return tuple(self.decode(v) for v in node['tuple'])
        if tag == 'frozenset':
            return frozenset(self.decode(v) for v in node['frozenset'])
        if tag == 'dict':
            return {self.decode(k): self.decode(v) for k, v in node['dict']}
        if tag == 'dataclass':
            return self._dataclass(_resolve(node['dataclass']), node['fields'])
        if tag == 'module':
            if self.runtime is None:
                refuse('the state holds a torch module but no top-level runtime to place it with')
            return _torch_codec().decode_module(node, self.weights, self.decode, self.runtime)
        return refuse(f'an artifact state node has no tag this reader knows: {sorted(node)}')

    def _dataclass(self, cls: type, fields: Mapping[str, object]) -> object:
        declared = {f.name: f for f in dataclasses.fields(cls)}
        if set(fields) != set(declared):
            refuse(f'{_path_of(cls)} has fields {sorted(declared)}, the artifact holds {sorted(fields)}')
        values = {name: self.decode(node) for name, node in fields.items()}
        made = cls(**{name: v for name, v in values.items() if declared[name].init})
        for name, v in values.items():
            if not declared[name].init:
                object.__setattr__(made, name, v)
        return made


def sha256(data: bytes) -> str:
    return 'sha256:' + hashlib.sha256(data).hexdigest()


def canonical(record: Mapping[str, object]) -> bytes:
    return json.dumps(record, sort_keys=True, separators=(',', ':'), allow_nan=True).encode()


# --- the record <-> its TOML projection: None by omission from a declared table, never inside a list ---


def _undeclared(node: Mapping, declared: object, where: str) -> None:
    if extra := sorted(set(node) - set(declared)):
        refuse(f'{where} holds undeclared field {extra[0]!r} (all: {extra}); it declares {sorted(declared)}')


def _table_node(node: object, where: str) -> Mapping:
    if not isinstance(node, Mapping):
        refuse(f'{where} is a table, got {node!r}')
    return node


def _schema_table(family: object, state: object, where: str = 'schema') -> Table:
    """The declared schema table of *family*; an ensemble's is its first member's, read off *state*."""
    table = SCHEMA_FIELDS.get(family) if isinstance(family, str) else None
    if table is None:
        refuse(f'{where} belongs to family {family!r}, which is not one of {list(FAMILIES)}')
    if table != _MEMBERS:
        return table
    try:
        first = state['fields']['members']['tuple'][0]
        member = next(name for name, (_, fitted) in FAMILIES.items() if fitted == first['dataclass'])
    except (KeyError, IndexError, TypeError, StopIteration):
        refuse(f'{where} of a {family} is its first member family, which its state does not name')
    return _schema_table(member, first, where)


def _fixed_shape(schema: object) -> tuple[int, int] | None:
    """(outputs, inputs) of a vector schema: the shape of every ``PowerLaw.fixed`` table in its record."""
    try:
        shape = (schema['outputs']['width'], schema['inputs']['width'])
    except (KeyError, TypeError):
        return None
    return shape if all(isinstance(n, int) and not isinstance(n, bool) for n in shape) else None


def _store_value(value: object, where: str) -> object:
    """A value with no declared table under it: it holds no ``None`` anywhere, which TOML could not write."""
    if value is None:
        refuse(f'{where} is None inside a list, which a TOML manifest cannot hold')
    if isinstance(value, list):
        return [_store_value(v, f'{where}[{i}]') for i, v in enumerate(value)]
    if isinstance(value, Mapping):
        return {k: _store_value(v, f'{where}.{k}') for k, v in value.items()}
    return value


def _store_table(node: object, table: Table, where: str) -> dict:
    stored = {}
    _undeclared(_table_node(node, where), table, where)
    for key, value in node.items():
        if value is not None:
            sub = table[key]
            here = f'{where}.{key}'
            stored[key] = _store_value(value, here) if sub is None else _store_table(value, sub, here)
    return stored


def _restore_table(node: object, table: Table, where: str) -> dict:
    _undeclared(_table_node(node, where), table, where)
    restored = {}
    for key, sub in table.items():
        value = node.get(key)
        restored[key] = value if value is None or sub is None else _restore_table(value, sub, f'{where}.{key}')
    return restored


def _tagged(node: object, where: str) -> tuple[str, tuple[str, ...]]:
    """The tag of a state-tree table and its declared keys; an untagged table is refused."""
    if isinstance(node, Mapping):
        tag = next((tag for tag in _TAG_FIELDS if tag in node), None)
        if tag is not None:
            _undeclared(node, _TAG_FIELDS[tag], where)
            return tag, _TAG_FIELDS[tag]
    return refuse(f'{where} is a state-tree table with none of the tags {list(_TAG_FIELDS)}: {node!r}')


def _declared_fields(path: object, where: str) -> tuple[str, ...]:
    fields = DATACLASS_FIELDS.get(path) if isinstance(path, str) else None
    if fields is None:
        refuse(f'{where} names {path!r}, which is not a learn_lab dataclass an artifact declares')
    return fields


def _store_tree(node: object, where: str, shape: tuple[int, int] | None) -> object:
    if isinstance(node, list):
        stored = []
        for i, value in enumerate(node):
            if value is None:
                refuse(f'{where}[{i}] is None inside a list, which a TOML manifest cannot hold')
            stored.append(_store_tree(value, f'{where}[{i}]', shape))
        return stored
    if not isinstance(node, Mapping):
        return node
    tag, keys = _tagged(node, where)
    if tag != 'dataclass':
        return {k: _store_tree(node[k], f'{where}.{k}', shape) for k in keys if node.get(k) is not None}
    path = node['dataclass']
    fields = _table_node(node.get('fields'), f'{where}.fields')
    _undeclared(fields, _declared_fields(path, where), f'{where}.fields')
    stored = {}
    for name, value in fields.items():
        here = f'{where}.fields.{name}'
        if value is None:
            continue
        if (path, name) == (_POWER_LAW, 'fixed'):
            stored[name] = _store_fixed(value, here, shape)
        else:
            stored[name] = _store_tree(value, here, shape)
    return {'dataclass': path, 'fields': stored}


def _restore_tree(node: object, where: str, shape: tuple[int, int] | None) -> object:
    if isinstance(node, list):
        return [_restore_tree(v, f'{where}[{i}]', shape) for i, v in enumerate(node)]
    if not isinstance(node, Mapping):
        return node
    tag, keys = _tagged(node, where)
    if tag != 'dataclass':
        return {k: None if node.get(k) is None else _restore_tree(node[k], f'{where}.{k}', shape) for k in keys}
    path = node['dataclass']
    declared = _declared_fields(path, where)
    fields = _table_node(node.get('fields', {}), f'{where}.fields')
    _undeclared(fields, declared, f'{where}.fields')
    restored = {}
    for name in declared:
        here = f'{where}.fields.{name}'
        value = fields.get(name)
        if value is None:
            restored[name] = None
        elif (path, name) == (_POWER_LAW, 'fixed'):
            restored[name] = _restore_fixed(value, here, shape)
        else:
            restored[name] = _restore_tree(value, here, shape)
    return {'dataclass': path, 'fields': restored}


def _store_fixed(node: object, where: str, shape: tuple[int, int] | None) -> list[dict]:
    """``PowerLaw.fixed`` -- ``{'tuple': [{'tuple': row}, ...]}``, ``None`` = free -- as its FIXED cells only."""
    try:
        rows = [row['tuple'] for row in node['tuple']]
    except (KeyError, TypeError):
        refuse(f'{where} is a table of exponent rows, got {node!r}')
    table = (len(rows), len(rows[0]) if rows else 0)
    if shape is None or table != shape or any(len(row) != shape[1] for row in rows):
        refuse(f'{where}: the fixed table is {table}, while its record declares {shape} (outputs, inputs)')
    return [
        {'output': i, 'input': j, 'exponent': _store_tree(x, f'{where}[{i}][{j}]', shape)}
        for i, row in enumerate(rows)
        for j, x in enumerate(row)
        if x is not None
    ]


def _restore_fixed(cells: object, where: str, shape: tuple[int, int] | None) -> dict:
    """The full ``PowerLaw.fixed`` table: every listed cell fixed, every unlisted one free (``None``)."""
    if shape is None:
        refuse(f'{where} needs its record to declare (outputs, inputs) widths, and the schema does not')
    if not isinstance(cells, list):
        refuse(f'{where} is the list of fixed exponents, got {cells!r}')
    rows: list[list[object]] = [[None] * shape[1] for _ in range(shape[0])]
    for k, cell in enumerate(cells):
        here = f'{where}[{k}]'
        _undeclared(_table_node(cell, here), _CELL, here)
        i, j = cell.get('output'), cell.get('input')
        if not all(isinstance(n, int) and not isinstance(n, bool) for n in (i, j)):
            refuse(f'{here} names its cell by integer output and input, got ({i!r}, {j!r})')
        if not (0 <= i < shape[0] and 0 <= j < shape[1]):
            refuse(f'{here} names cell ({i}, {j}) outside the {shape} table')
        if rows[i][j] is not None or 'exponent' not in cell:
            refuse(f'{here} names cell ({i}, {j}) twice or without its exponent')
        rows[i][j] = _restore_tree(cell['exponent'], f'{here}.exponent', shape)
    return {'tuple': [{'tuple': row} for row in rows]}


def _stored(record: Mapping[str, object]) -> dict:
    """The TOML projection of *record*: ``None`` by omission from its declared tables; refused where it cannot be."""
    _undeclared(record, FIELDS, 'the record')
    shape = _fixed_shape(record.get('schema'))
    stored = {}
    for name, value in record.items():
        if value is None:
            continue
        if name in ('declaration', 'state'):
            value = _store_tree(value, name, shape)
        elif name == 'schema':
            value = _store_table(value, _schema_table(record.get('family'), record.get('state')), name)
        elif name in ('learn_lab', 'runtime'):
            value = _store_table(value, _LEARN_LAB if name == 'learn_lab' else _RUNTIME, name)
        else:
            value = _store_value(value, name)
        stored[name] = value
    return stored


def _restored(raw: Mapping[str, object], path: Path) -> dict:
    """The record a TOML manifest at *path* stores: every declared absent field ``None``, undeclared refused."""
    _undeclared(raw, FIELDS, str(path / MANIFEST))
    record = {name: raw.get(name) for name in FIELDS}
    if record['schema'] is not None:
        record['schema'] = _restore_table(record['schema'], _schema_table(record['family'], record['state']), 'schema')
    for name, table in (('learn_lab', _LEARN_LAB), ('runtime', _RUNTIME)):
        if record[name] is not None:
            record[name] = _restore_table(record[name], table, name)
    shape = _fixed_shape(record['schema'])
    for name in ('declaration', 'state'):
        if record[name] is not None:
            record[name] = _restore_tree(record[name], name, shape)
    return record


def _git_commit() -> dict | None:
    """The commit of the learn-lab source tree that wrote the artifact, and whether it had local edits."""
    git = shutil.which('git')
    if git is None:
        return None
    here = str(Path(__file__).resolve().parent)
    try:
        sha = subprocess.run(
            [git, '-C', here, 'rev-parse', 'HEAD'], capture_output=True, text=True, check=True, timeout=10
        )
        dirty = subprocess.run(
            [git, '-C', here, 'status', '--porcelain', '--untracked-files=no', '--', '.'],
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return {'sha': sha.stdout.strip(), 'dirty': bool(dirty.stdout.strip())}


def _learn_lab_version() -> str | None:
    try:
        return package_version('learn-lab')
    except PackageNotFoundError:
        return None


def _seed(declaration: object) -> int | None:
    """The declaration's own ``seed``, else its runtime's; ``None`` for a declaration without randomness."""
    names = {f.name for f in dataclasses.fields(declaration)}
    if 'seed' in names:
        return declaration.seed
    if 'runtime' in names:
        return declaration.runtime.seed
    return None


def save(
    fitted: object,
    path: str | Path,
    *,
    declaration: object,
    data_identity: str,
    metrics: Mapping[str, float] | None = None,
) -> Path:
    """Write *fitted* as a new artifact directory at *path* and return it.

    *declaration* is the model it was fitted from (its settings are recorded and must be the family's own
    declaration type); *data_identity* is the caller's identity of the training data (a content hash, a dataset
    id) -- learn-lab never sees the data again, so it records what it is told; *metrics* are validation results
    by name, finite numbers in their own units. An existing *path* is refused: an artifact is never overwritten.
    """
    family = family_of(fitted)
    if _path_of(type(declaration)) != FAMILIES[family][0]:
        refuse(f'a {family} fit is declared by {FAMILIES[family][0]}, got {_path_of(type(declaration))}')
    if not isinstance(data_identity, str) or not data_identity.strip():
        refuse('an artifact needs the identity of its training data: a non-empty string, e.g. a content hash')
    metrics = dict(metrics or {})
    for name, value in metrics.items():
        if (
            not isinstance(name, str)
            or isinstance(value, bool)
            or not isinstance(value, int | float)
            or not np.isfinite(value)
        ):
            refuse(f'a metric is a finite number by name, got {name!r}: {value!r}')
    path = Path(path)
    if path.exists():
        refuse(f'{path} exists; an artifact is written to a new directory, never over another')
    writer = _Writer()
    state = writer.encode(fitted, 'the fit')
    settings = writer.encode(declaration, 'the declaration')
    runtime = state['fields'].get('runtime')
    record = {
        'format': FORMAT,
        'version': VERSION,
        'learn_lab': {'version': _learn_lab_version(), 'git': _git_commit()},
        'family': family,
        'declaration': settings,
        'schema': _schema(family, fitted),
        'seed': _seed(declaration),
        'runtime': None if runtime is None else {k: runtime['fields'][k] for k in ('device', 'dtype', 'seed')},
        'data_identity': data_identity,
        'metrics': metrics,
        'state': state,
    }
    _stored(record)  # what TOML cannot hold is refused before anything is written
    files: dict[str, bytes] = {}
    path.mkdir(parents=True)
    np.savez(path / ARRAYS, **writer.arrays)
    files[ARRAYS] = (path / ARRAYS).read_bytes()
    if writer.weights:
        _torch_codec().write_weights(writer.weights, path / WEIGHTS)
        files[WEIGHTS] = (path / WEIGHTS).read_bytes()
    record['payload'] = {name: sha256(data) for name, data in files.items()}
    record['content_hash'] = sha256(canonical(record))
    text = tomlkit.dumps(_stored(record), sort_keys=True)
    (path / MANIFEST).write_text(text, encoding='utf-8', newline='\n')
    return path


def manifest(path: str | Path) -> dict:
    """The verified manifest of the artifact at *path*: its version known, its hash and every payload file's intact.

    The record is restored from the TOML by its declared tables alone -- no model module is imported.
    """
    path = Path(path)
    if (path / RETIRED_MANIFEST).exists():
        refuse(f'{path / RETIRED_MANIFEST} is a retired JSON artifact manifest and is not read; {MIGRATE_DOOR}')
    try:
        raw = tomllib.loads((path / MANIFEST).read_text(encoding='utf-8'))
    except (OSError, ValueError) as exc:
        refuse(f'{path / MANIFEST} is not a readable artifact manifest: {exc}')
    if raw.get('format') != FORMAT:
        refuse(f'{path} is not a {FORMAT}')
    if raw.get('version') != VERSION:
        refuse(f'{path} is {FORMAT} version {raw.get("version")!r}; this learn-lab reads version {VERSION} only')
    record = _restored(raw, path)
    claimed = record.pop('content_hash', None)
    if claimed != sha256(canonical(record)):
        refuse(f'{path / MANIFEST} does not match its content hash: the manifest was changed after it was written')
    payload = record['payload'] or {}
    for name in (ARRAYS, WEIGHTS):
        present = (path / name).exists()
        if present != (name in payload):
            refuse(f'{path / name} is {"present" if present else "missing"}, the manifest says otherwise')
        if present and sha256((path / name).read_bytes()) != payload[name]:
            refuse(
                f'{path / name} does not match its hash in the manifest: the payload was changed after it was written'
            )
    record['content_hash'] = claimed
    return record


def load(path: str | Path, *, device: str | None = None) -> object:
    """The fitted model of the artifact at *path*, verified before any payload is parsed.

    A deep family is placed on its saved runtime device (``'auto'`` resolves on THIS machine) unless *device*
    (``'cpu'``, ``'cuda[:n]'``, ``'auto'``) places it elsewhere; the loaded runtime records where it now is.
    """
    path = Path(path)
    record = manifest(path)
    state = record['state']
    runtime = state.get('fields', {}).get('runtime') if isinstance(state, dict) else None
    if device is not None:
        if runtime is None:
            refuse(f'a {record["family"]} fit has no runtime; only a deep family is placed on a device')
        runtime['fields']['device'] = device
    with np.load(path / ARRAYS, allow_pickle=False) as npz:
        arrays = {key: npz[key] for key in npz.files}
    weights = _torch_codec().read_weights(path / WEIGHTS) if WEIGHTS in record['payload'] else {}
    reader = _Reader(arrays, weights, None)
    if runtime is not None:
        reader.runtime = reader.decode(runtime)
    fitted = reader.decode(state)
    if family_of(fitted) != record['family']:
        refuse(f'{path} declares family {record["family"]!r} but holds a {family_of(fitted)}')
    return fitted

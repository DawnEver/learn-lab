"""The artifact contract: a fitted model as a versioned, verifiable directory -- saved, reloaded, audited, reused.

``save(fitted, path, declaration=, data_identity=, metrics=)`` writes ONE new directory; ``load(path)`` returns
a fitted model whose every answer equals the original's bit for bit on the same device; :func:`manifest` returns
the verified record for an audit without building the model. The directory holds:

- ``manifest.json`` -- the format and its version, the learn-lab version and git commit that wrote it, the
  family (its :data:`FAMILIES` name), the declaration it was fitted from (its constructor settings), the input
  and output schema (a response family's inputs carry the ENVELOPE its fit answers in), the seed, the runtime,
  the caller's data identity and metrics, the fitted STATE as a tagged tree, the SHA-256 of each payload file, and ``content_hash`` -- the SHA-256 of the rest of itself;
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
"""

from __future__ import annotations

import dataclasses
import enum
import hashlib
import importlib
import json
import shutil
import subprocess
from collections.abc import Callable, Mapping
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as package_version
from pathlib import Path
from types import ModuleType
from typing import Final

import numpy as np

from learn_lab.errors import refuse
from learn_lab.validity import BoxEnvelope

__all__ = ['FAMILIES', 'FORMAT', 'VERSION', 'load', 'manifest', 'save']

FORMAT: Final = 'learn-lab-artifact'
VERSION: Final = 1
MANIFEST: Final = 'manifest.json'
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
    'chaos_expansion': (f'{_RESPONSE}chaos:ChaosExpansion', f'{_RESPONSE}chaos:FittedChaos'),
    'ensemble': (f'{_RESPONSE}ensemble:Ensemble', f'{_RESPONSE}ensemble:FittedEnsemble'),
    'bagging': (f'{_RESPONSE}bagging:Bagging', f'{_RESPONSE}bagging:FittedBagging'),
    'sparse_regression': ('learn_lab.equation:SparseRegression', 'learn_lab.equation:IdentifiedLaw'),
    'deep_ensemble': ('learn_lab.deep.surrogate:DeepEnsemble', 'learn_lab.deep.surrogate:FittedEnsemble'),
    'physics_informed_field': ('learn_lab.deep.physics:PhysicsInformedField', 'learn_lab.deep.physics:FittedField'),
    'monotone_law': ('learn_lab.deep.constitutive:MonotoneLaw', 'learn_lab.deep.constitutive:FittedLaw'),
    'play_hysteresis': ('learn_lab.deep.hysteresis:PlayHysteresis', 'learn_lab.deep.hysteresis:FittedHysteresis'),
    'field_model': ('learn_lab.deep.field:FieldModel', 'learn_lab.deep.field:FittedFieldModel'),
    'deep_onet': ('learn_lab.deep.operator:DeepONet', 'learn_lab.deep.operator:FittedDeepONet'),
    'fno': ('learn_lab.deep.operator:FNO', 'learn_lab.deep.operator:FittedFNO'),
    'graph_model': ('learn_lab.deep.graph:GraphModel', 'learn_lab.deep.graph:FittedGraphModel'),
    'neural_ode': ('learn_lab.deep.dynamics:NeuralODE', 'learn_lab.deep.dynamics:FittedDynamics'),
}


def _vector(envelope: BoxEnvelope, k: int) -> dict:
    """A vector family's schema: the input width and the ENVELOPE its fit answers in, and the output width."""
    inputs = {
        'kind': 'vector',
        'width': envelope.width,
        'envelope': {'lower': envelope.lower.tolist(), 'upper': envelope.upper.tolist()},
    }
    return {'inputs': inputs, 'outputs': {'kind': 'vector', 'width': int(k)}}


#: The input/output schema of each numpy family, read off its fitted state; the deep rows are the codec's.
_SCHEMAS: Final[dict[str, Callable[[object], dict]]] = {
    'polynomial': lambda f: _vector(f.envelope, f.coefficients.shape[1]),
    'radial_basis': lambda f: _vector(f.envelope, f.weights.shape[1]),
    'gaussian_process': lambda f: _vector(f.envelope, len(f.columns)),
    'gradient_enhanced_kriging': lambda f: _vector(f.envelope, len(f.columns)),
    'multi_output_kriging': lambda f: _vector(f.envelope, f.level.size),
    'co_kriging': lambda f: _vector(f.envelope, len(f.base.columns)),
    'nearest_neighbours': lambda f: _vector(f.envelope, f.y.shape[1]),
    'regression_tree': lambda f: _vector(f.envelope, f.value.shape[1]),
    'forest': lambda f: _vector(f.envelope, f.trees[0].value.shape[1]),
    'multilayer_perceptron': lambda f: _vector(f.envelope, f.weights[-1].shape[1]),
    'chaos_expansion': lambda f: _vector(f.envelope, f.coefficients.shape[1]),
    'ensemble': lambda f: _schema(_family_of(f.members[0]), f.members[0]),
    'bagging': lambda f: _schema(_family_of(f.members[0]), f.members[0]),
    'sparse_regression': lambda f: {
        'inputs': {'kind': 'state', 'variables': list(f.library.variables)},
        'outputs': {'kind': 'rate', 'targets': list(f.targets)},
    },
}


def _path_of(cls: type) -> str:
    return f'{cls.__module__}:{cls.__qualname__}'


def _family_of(fitted: object) -> str:
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


def _sha256(data: bytes) -> str:
    return 'sha256:' + hashlib.sha256(data).hexdigest()


def _canonical(record: Mapping[str, object]) -> bytes:
    return json.dumps(record, sort_keys=True, separators=(',', ':'), allow_nan=True).encode()


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
    family = _family_of(fitted)
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
    files: dict[str, bytes] = {}
    path.mkdir(parents=True)
    np.savez(path / ARRAYS, **writer.arrays)
    files[ARRAYS] = (path / ARRAYS).read_bytes()
    if writer.weights:
        _torch_codec().write_weights(writer.weights, path / WEIGHTS)
        files[WEIGHTS] = (path / WEIGHTS).read_bytes()
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
        'payload': {name: _sha256(data) for name, data in files.items()},
    }
    record['content_hash'] = _sha256(_canonical(record))
    (path / MANIFEST).write_text(json.dumps(record, indent=1, sort_keys=True, allow_nan=True), encoding='utf-8')
    return path


def manifest(path: str | Path) -> dict:
    """The verified manifest of the artifact at *path*: its version known, its hash and every payload file's intact."""
    path = Path(path)
    try:
        record = json.loads((path / MANIFEST).read_text(encoding='utf-8'))
    except (OSError, ValueError) as exc:
        refuse(f'{path / MANIFEST} is not a readable artifact manifest: {exc}')
    if not isinstance(record, dict) or record.get('format') != FORMAT:
        refuse(f'{path} is not a {FORMAT}')
    if record.get('version') != VERSION:
        refuse(f'{path} is {FORMAT} version {record.get("version")!r}; this learn-lab reads version {VERSION} only')
    claimed = record.pop('content_hash', None)
    if claimed != _sha256(_canonical(record)):
        refuse(f'{path / MANIFEST} does not match its content hash: the manifest was changed after it was written')
    payload = record.get('payload', {})
    for name in (ARRAYS, WEIGHTS):
        present = (path / name).exists()
        if present != (name in payload):
            refuse(f'{path / name} is {"present" if present else "missing"}, the manifest says otherwise')
        if present and _sha256((path / name).read_bytes()) != payload[name]:
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
    if _family_of(fitted) != record['family']:
        refuse(f'{path} declares family {record["family"]!r} but holds a {_family_of(fitted)}')
    return fitted

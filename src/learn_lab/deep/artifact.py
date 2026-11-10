"""The torch half of :mod:`learn_lab.artifact`: a family's modules, their weights, their schema.

A module is persisted as the ``arguments`` it records (see :mod:`learn_lab.deep.networks`) and its state dict;
it is rebuilt ONLY from :data:`MODULES` -- a name the table does not hold is refused -- under a forked RNG
(so loading never moves the caller's torch seed), then placed on the runtime's device and dtype before its
weights are copied in. Weights are written as one flat ``{name: tensor}`` dict on the CPU (a file is
device-free) and read with ``torch.load(weights_only=True)``: tensors and plain containers, never an
arbitrary object.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Final

import torch
from torch import nn

from learn_lab.deep.field import UNet
from learn_lab.deep.graph import MessagePassing
from learn_lab.deep.hysteresis import PlayBank
from learn_lab.deep.networks import MLP, MonotoneScalar
from learn_lab.deep.operator import _DeepONetNet, _FNONet
from learn_lab.deep.runtime import Runtime
from learn_lab.errors import refuse

__all__ = ['MODULES', 'decode_module', 'encode_module', 'read_weights', 'schema', 'write_weights']

#: The modules an artifact may rebuild, by name.
MODULES: Final[dict[str, type[nn.Module]]] = {
    'MLP': MLP,
    'MonotoneScalar': MonotoneScalar,
    'UNet': UNet,
    'DeepONetNet': _DeepONetNet,
    'FNONet': _FNONet,
    'MessagePassing': MessagePassing,
    'PlayBank': PlayBank,
}


def encode_module(
    module: object, weights: dict[str, torch.Tensor], encode: Callable[[object], object], where: str
) -> dict:
    """The state-tree node of *module*; its tensors go into *weights* under a fresh prefix, copied to the CPU."""
    name = next((name for name, cls in MODULES.items() if type(module) is cls), None)
    if name is None:
        refuse(f'{where} is a {type(module).__qualname__}; an artifact rebuilds only the modules {list(MODULES)}')
    prefix = f'm{len({key.split(".")[0] for key in weights})}'
    for key, tensor in module.state_dict().items():
        weights[f'{prefix}.{key}'] = tensor.detach().to('cpu', copy=True)
    return {'module': name, 'arguments': encode(module.arguments), 'state': prefix}


def decode_module(
    node: Mapping, weights: Mapping[str, torch.Tensor], decode: Callable[[object], object], runtime: Runtime
) -> nn.Module:
    """Rebuild the module of *node* on *runtime*'s device and dtype and load its weights exactly."""
    if node.get('module') not in MODULES:
        refuse(f'an artifact module is one of {list(MODULES)}, got {node.get("module")!r}')
    arguments = decode(node['arguments'])
    if not isinstance(arguments, dict):
        refuse(f'a module is rebuilt from its named arguments, got {arguments!r}')
    with torch.random.fork_rng(devices=[]):
        module = MODULES[node['module']](**arguments)
    prefix = f'{node["state"]}.'
    state = {key.removeprefix(prefix): tensor for key, tensor in weights.items() if key.startswith(prefix)}
    module = module.to(runtime.torch_device, runtime.torch_dtype)
    module.load_state_dict(state, strict=True)
    return module.eval()


def write_weights(weights: Mapping[str, torch.Tensor], path: Path) -> None:
    torch.save(dict(weights), path)


def read_weights(path: Path) -> dict[str, torch.Tensor]:
    """The flat ``{name: tensor}`` dict at *path*, on the CPU, through the weights-only unpickler."""
    weights = torch.load(path, map_location='cpu', weights_only=True)
    if not isinstance(weights, dict) or not all(
        isinstance(k, str) and isinstance(v, torch.Tensor) for k, v in weights.items()
    ):
        refuse(f'{path} is not a flat {{name: tensor}} weights dict')
    return weights


def _box(envelope: object) -> dict:
    return {
        'kind': 'vector',
        'width': envelope.width,
        'lower': envelope.lower.tolist(),
        'upper': envelope.upper.tolist(),
    }


def _channels(envelope: object) -> dict:
    return {
        'kind': 'channels',
        'channels': envelope.channels,
        'grid': None if envelope.shape is None else list(envelope.shape),
        'lower': envelope.box.lower.tolist(),
        'upper': envelope.box.upper.tolist(),
    }


def _channels_out(standardiser: object, kind: str = 'channels') -> dict:
    return {'kind': kind, 'channels': int(standardiser.shift.size)}


#: The input/output schema of each deep family, read off its envelopes and standardisers.
_SCHEMAS: Final[dict[str, Callable[[object], dict]]] = {
    'deep_ensemble': lambda f: {
        'inputs': _box(f.envelope),
        'outputs': {'kind': 'vector', 'width': int(f.y_shift.size)},
    },
    'physics_informed_field': lambda f: {
        'inputs': _box(f.domain),
        'outputs': {'kind': 'vector', 'width': f.net.arguments['widths'][-1]},
    },
    'monotone_law': lambda f: {'inputs': {**_box(f.envelope), 'kind': 'scalar'}, 'outputs': {'kind': 'scalar'}},
    'play_hysteresis': lambda f: {
        'inputs': {**_box(f.envelope), 'kind': 'scalar history'},
        'outputs': {'kind': 'scalar history', 'plays': f.bank.arguments['plays']},
    },
    'field_model': lambda f: {'inputs': _channels(f.envelope), 'outputs': _channels_out(f.outputs)},
    'deep_onet': lambda f: {
        'inputs': {'sensors': _channels(f.sensors), 'points': _box(f.points)},
        'outputs': {'kind': 'scalar per point'},
    },
    'fno': lambda f: {'inputs': _channels(f.envelope), 'outputs': _channels_out(f.outputs)},
    'graph_model': lambda f: {
        'inputs': {
            'nodes': _channels(f.node_envelope),
            'edges': None if f.edge_envelope is None else _channels(f.edge_envelope),
        },
        'outputs': _channels_out(f.outputs, 'node channels'),
    },
    'neural_ode': lambda f: {
        'inputs': _box(f.envelope),
        'outputs': {'kind': 'trajectory', 'width': f.envelope.width},
    },
}


def schema(family: str, fitted: object) -> dict:
    """The input/output schema of a deep *family*'s fit, read off its envelopes and standardisers."""
    if family not in _SCHEMAS:
        refuse(f'{family!r} is not a deep family; the deep families are {list(_SCHEMAS)}')
    return _SCHEMAS[family](fitted)

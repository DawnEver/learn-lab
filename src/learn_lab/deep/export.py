"""A fitted deep model as a portable inference graph: ONNX, then OpenVINO IR at f16 or int8. Needs ``[openvino]``.

The consumer owns the runtime; learn-lab exports and offers :func:`learn_lab.equivalence.equivalence`.

- :func:`to_onnx` writes ONE new directory: ``model.onnx`` (float32, one STATIC input shape, the standardisers
  folded into the graph so it maps raw inputs to raw outputs) and ``manifest.json`` -- the format and version,
  the family and its schema (the artifact's, :func:`learn_lab.deep.artifact.schema`), the input and output
  names and shapes, the SHA-256 of every payload file and ``content_hash`` over the rest. The envelope is NOT in
  the graph: the consumer checks it against the schema before calling. No file is pickled.
- :func:`quantize` writes a sibling directory ``<name>.<precision>`` holding ``model.xml``/``model.bin``:
  ``'f16'`` compresses the weights, ``'int8'`` is NNCF post-training quantization on caller-given calibration
  inputs. Whether that precision is ADMITTED is not decided here: judge it per model with ``equivalence``.
- :func:`manifest` returns a directory's verified manifest; one changed byte is refused.

A family whose graph does not export cleanly is refused by name (:data:`EXPORTABLE` lists those that do).
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Final, Literal

import numpy as np
import torch
from torch import nn

try:
    import nncf
    import openvino as ov
except ModuleNotFoundError as exc:  # pragma: no cover - the extra is the one remedy
    msg = 'learn_lab.deep.export needs the openvino extra: install learn-lab[openvino]'
    raise ModuleNotFoundError(msg) from exc

from learn_lab.artifact import canonical, family_of, sha256
from learn_lab.deep.artifact import schema
from learn_lab.deep.standardise import Standardiser
from learn_lab.errors import refuse, require_finite

__all__ = ['EXPORTABLE', 'FORMAT', 'VERSION', 'manifest', 'quantize', 'to_onnx']

FORMAT: Final = 'learn-lab-inference'
VERSION: Final = 1
MANIFEST: Final = 'manifest.json'
ONNX: Final = 'model.onnx'
IR: Final = ('model.xml', 'model.bin')
INPUT, OUTPUT = 'x', 'y'
OPSET: Final = 18


class _Raw(nn.Module):
    """``back(net(forward(x)))`` in float32: the standardisers folded around the network."""

    def __init__(self, net: nn.Module, inputs: Standardiser, outputs: Standardiser, channels: int) -> None:
        super().__init__()
        self.net = net
        shape = (1, -1) + (1,) * channels
        for name, value in (
            ('x_shift', inputs.shift),
            ('x_scale', inputs.scale),
            ('y_shift', outputs.shift),
            ('y_scale', outputs.scale),
        ):
            self.register_buffer(name, torch.as_tensor(np.asarray(value, float), dtype=torch.float32).reshape(shape))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net((x - self.x_shift) / self.x_scale) * self.y_scale + self.y_shift


def _grid(fitted: object) -> nn.Module:
    """A grid family (U-Net, FNO): channels standardised per channel over a 2-D grid."""
    net = _float32_copy(fitted.net)
    return _Raw(net, fitted.inputs, fitted.outputs, len(fitted.envelope.shape))


def _float32_copy(net: nn.Module) -> nn.Module:
    with torch.random.fork_rng(devices=[]):
        copy = type(net)(**net.arguments)
    copy.load_state_dict({k: v.detach().to('cpu') for k, v in net.state_dict().items()})
    return copy.to('cpu', torch.float32).eval()


#: The families whose graph exports cleanly, by artifact name, and how their raw-in/raw-out graph is built.
EXPORTABLE: Final[dict[str, Callable[[object], nn.Module]]] = {'field_model': _grid}


def to_onnx(fitted: object, path: str | Path, *, example: np.ndarray) -> Path:
    """Export *fitted* to a new directory *path* for inputs of exactly ``example.shape``; returns *path*."""
    family = family_of(fitted)
    if family not in EXPORTABLE:
        refuse(f'a {family} fit does not export to ONNX; the exportable families are {list(EXPORTABLE)}')
    example = fitted.envelope.require(require_finite(np.asarray(example, float), 'the example'), 'an export example')
    path = Path(path)
    if path.exists():
        refuse(f'{path} exists; an export is written to a new directory, never over another')
    module = EXPORTABLE[family](fitted)
    x = torch.as_tensor(example, dtype=torch.float32)
    with torch.no_grad():
        y = module(x)
    path.mkdir(parents=True)
    torch.onnx.export(
        module, (x,), path / ONNX, input_names=[INPUT], output_names=[OUTPUT], opset_version=OPSET, dynamo=False
    )
    record = {
        'family': family,
        'schema': schema(family, fitted),
        'precision': 'f32',
        'input': {'name': INPUT, 'shape': list(x.shape), 'dtype': 'float32'},
        'output': {'name': OUTPUT, 'shape': list(y.shape), 'dtype': 'float32'},
        'opset': OPSET,
    }
    return _seal(path, record, [ONNX])


def quantize(
    onnx_path: str | Path, calibration_inputs: np.ndarray, *, precision: Literal['int8', 'f16'] = 'int8'
) -> Path:
    """The OpenVINO IR of the export at *onnx_path* in *precision*, written to ``<onnx_path>.<precision>``.

    *calibration_inputs* are raw inputs ``(n, *shape[1:])`` with ``n`` a multiple of the static batch; int8
    calibrates its activation ranges on them (they should span the envelope the model is used in).
    """
    source = Path(onnx_path)
    record = manifest(source)
    if record['precision'] != 'f32' or ONNX not in record['payload']:
        refuse(f'{source} is a {record["precision"]} IR; quantize the f32 ONNX export, not a derived one')
    if precision not in {'int8', 'f16'}:
        refuse(f"a precision is 'int8' or 'f16', got {precision!r}")
    shape = record['input']['shape']
    calibration = require_finite(np.asarray(calibration_inputs, np.float32), 'the calibration inputs')
    if calibration.shape[1:] != tuple(shape[1:]) or not len(calibration) or len(calibration) % shape[0]:
        refuse(f'calibration inputs are (k * {shape[0]}, *{shape[1:]}), got {calibration.shape}')
    target = source.with_name(f'{source.name}.{precision}')
    if target.exists():
        refuse(f'{target} exists; a quantized model is written to a new directory, never over another')
    model = ov.convert_model(source / ONNX)
    if precision == 'int8':
        batches = list(calibration.reshape(-1, *shape))
        model = nncf.quantize(model, nncf.Dataset(batches), subset_size=len(batches))
    target.mkdir()
    ov.save_model(model, target / IR[0], compress_to_fp16=precision == 'f16')
    derived = {k: record[k] for k in ('family', 'schema', 'input', 'output', 'opset')}
    derived |= {'precision': precision, 'source': {ONNX: record['payload'][ONNX]}}
    return _seal(target, derived, list(IR))


def _seal(path: Path, record: dict, files: list[str]) -> Path:
    record = {'format': FORMAT, 'version': VERSION, **record}
    record['payload'] = {name: sha256((path / name).read_bytes()) for name in files}
    record['content_hash'] = sha256(canonical(record))
    (path / MANIFEST).write_text(json.dumps(record, indent=1, sort_keys=True), encoding='utf-8')
    return path


def manifest(path: str | Path) -> dict:
    """The verified manifest of the export at *path*: its format known, its hash and every payload file's intact."""
    path = Path(path)
    try:
        record = json.loads((path / MANIFEST).read_text(encoding='utf-8'))
    except (OSError, ValueError) as exc:
        refuse(f'{path / MANIFEST} is not a readable export manifest: {exc}')
    if not isinstance(record, dict) or record.get('format') != FORMAT or record.get('version') != VERSION:
        refuse(f'{path} is not a {FORMAT} version {VERSION}')
    claimed = record.pop('content_hash', None)
    if claimed != sha256(canonical(record)):
        refuse(f'{path / MANIFEST} does not match its content hash: the manifest was changed after it was written')
    for name, digest in record['payload'].items():
        if not (path / name).is_file() or sha256((path / name).read_bytes()) != digest:
            refuse(f'{path / name} is missing or does not match its hash in the manifest')
    record['content_hash'] = claimed
    return record

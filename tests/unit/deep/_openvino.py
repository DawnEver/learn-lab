"""The test side of an OpenVINO runtime: name a device kind, get exactly that device or a refusal by name.

learn-lab ships no runtime door (the consumer owns it); the tests need one to measure an export. ``'GPU'`` is the
INTEGRATED GPU only: OpenVINO also enumerates a discrete NVIDIA card through OpenCL, which these tests never use.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import openvino as ov

from learn_lab.deep import export

KINDS = ('CPU', 'GPU', 'NPU')


def resolve(available: list[str], types: dict[str, str], kind: str) -> str:
    """The OpenVINO device name of *kind* among *available* (``types``: name -> DEVICE_TYPE); else LookupError."""
    if kind == 'GPU':
        found = [name for name in available if name.startswith('GPU') and types.get(name) == 'integrated']
    else:
        found = [name for name in available if name == kind]
    if not found:
        msg = f'OpenVINO lists no {kind} device (integrated, for GPU); it lists {available}'
        raise LookupError(msg)
    return found[0]


def device(core: ov.Core, kind: str) -> str:
    gpus = [name for name in core.available_devices if name.startswith('GPU')]
    types = {name: str(core.get_property(name, 'DEVICE_TYPE')).rsplit('.', 1)[-1].lower() for name in gpus}
    return resolve(core.available_devices, types, kind)


@dataclass(frozen=True, slots=True)
class Compiled:
    request: ov.InferRequest
    seconds: float  # compile time

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Host in, host out: the transfer to and from the device is inside the call."""
        return self.request.infer({0: np.ascontiguousarray(x, np.float32)})[0].copy()


def compile_on(core: ov.Core, path: Path, kind: str) -> Compiled:
    record = export.manifest(path)
    model = path / (export.ONNX if export.ONNX in record['payload'] else export.IR[0])
    name = device(core, kind)
    start = time.perf_counter()
    compiled = core.compile_model(core.read_model(model), name)
    return Compiled(compiled.create_infer_request(), time.perf_counter() - start)

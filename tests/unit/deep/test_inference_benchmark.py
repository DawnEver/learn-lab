"""A dev MEASUREMENT, not a shipped feature: one U-Net's inference per OpenVINO device x precision.

A small U-Net is trained on synthetic data on the CPU (torch, float32; never a CUDA device), exported, and
quantized. For each device kind in ``LEARN_LAB_BENCHMARK_DEVICES`` (default ``CPU``; WS5 runs ``CPU,GPU,NPU``)
and each precision the device's ``OPTIMIZATION_CAPABILITIES`` admit, it records:

- ``compile_s`` -- read + compile, measured once and kept out of the latency;
- ``latency_ms`` -- the median of :data:`REPEATS` batch calls after one warm-up call, host array in and host array
  out, so the host<->device transfer is inside;
- ``max_error`` -- the largest |output - torch f32 output|, in the output's unit, beside that output's scale,
  and ``faithful`` -- whether it is under :data:`FAITHFUL` of the scale (admission itself is ``equivalence``'s).

A kind the runtime does not list RAISES by name (``_openvino.resolve``); nothing falls back to another device.
A compile that fails is recorded with its message. The table goes to ``output/logs/<yy>/<mm>/<dd>/benchmark/``.
Ship rule (user ruling 2026-11-08): a device/precision ships only at >= 2x the torch CPU f32 latency.
"""

import json
import os
import statistics
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip('openvino', reason='the benchmark needs the [openvino] extra')

import openvino as ov

from learn_lab.deep import export
from learn_lab.deep.field import FieldModel
from learn_lab.deep.fit import Schedule
from learn_lab.deep.runtime import Runtime

from . import _openvino

BATCH, GRID, REPEATS = 16, 64, 30
#: A cell whose max error exceeds this fraction of the output scale is not answering as the model does.
FAITHFUL = 0.1
PRECISIONS = {'f32': 'FP32', 'f16': 'FP16', 'int8': 'INT8'}
_ROOT = Path(__file__).resolve().parents[3]


def _median_ms(run, x):
    run(x)  # warm-up: first-call allocation is not a steady-state latency
    times = []
    for _ in range(REPEATS):
        start = time.perf_counter()
        run(x)
        times.append(time.perf_counter() - start)
    return 1e3 * statistics.median(times)


def test_inference_latency_and_error_per_device_and_precision(tmp_path):
    kinds = os.environ.get('LEARN_LAB_BENCHMARK_DEVICES', 'CPU').split(',')
    rng = np.random.default_rng(0)
    x = rng.uniform(-1, 1, (128, 1, GRID, GRID))
    y = x + 0.5 * (np.roll(x, 1, -1) + np.roll(x, -1, -1)) + 0.3 * x**2
    runtime = Runtime(device='cpu', dtype='float32')
    fitted = FieldModel(width=16, levels=3, schedule=Schedule(adam_steps=100), runtime=runtime).fit(x, y)
    queries = rng.uniform(-0.9, 0.9, (BATCH, 1, GRID, GRID))
    reference = fitted.predict(queries)
    scale = float(np.max(np.abs(reference)))
    rows = [{'device': 'torch CPU', 'precision': 'f32', 'latency_ms': _median_ms(fitted.predict, queries)}]
    onnx = export.to_onnx(fitted, tmp_path / 'unet', example=queries)
    paths = {'f32': onnx, **{p: export.quantize(onnx, x[:64], precision=p) for p in ('f16', 'int8')}}
    core = ov.Core()
    for kind in kinds:
        name = _openvino.device(core, kind)  # raises by name for a kind the runtime does not list
        capabilities = core.get_property(name, 'OPTIMIZATION_CAPABILITIES')
        for precision, path in paths.items():
            row = {'device': f'{kind} ({name}: {core.get_property(name, "FULL_DEVICE_NAME")})', 'precision': precision}
            if PRECISIONS[precision] not in capabilities:
                rows.append({**row, 'unsupported': f'{name} lists {capabilities}'})
                continue
            try:
                run = _openvino.compile_on(core, path, kind)
            except RuntimeError as exc:
                rows.append({**row, 'compile_failed': str(exc)})
                continue
            error = float(np.max(np.abs(run(queries) - reference)))
            row |= {'compile_s': run.seconds, 'latency_ms': _median_ms(run, queries), 'max_error': error}
            rows.append({**row, 'faithful': error < FAITHFUL * scale})
    record = {
        'when': datetime.now(UTC).isoformat(),
        'shape': [BATCH, 1, GRID, GRID],
        'output_scale': scale,
        'available': core.available_devices,
        'rows': rows,
    }
    now = datetime.now(UTC)
    where = _ROOT / 'output' / 'logs' / now.strftime('%y/%m/%d') / 'benchmark'
    where.mkdir(parents=True, exist_ok=True)
    (where / f'inference-{now:%H%M%S}.json').write_text(json.dumps(record, indent=1), encoding='utf-8')
    measured = [row for row in rows if 'latency_ms' in row]
    assert len(measured) >= 1 + len(kinds), rows  # every listed kind measured at least one precision
    assert all(np.isfinite(row['latency_ms']) and row['latency_ms'] > 0 for row in measured)
    # The f32 graph on the CPU is the model, so it must be faithful; a cell that is not is recorded, not shipped.
    assert next(row for row in rows if row['device'].startswith('CPU') and row['precision'] == 'f32')['faithful']

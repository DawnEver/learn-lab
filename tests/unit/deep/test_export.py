"""Export: a fitted U-Net as ONNX (raw in, raw out, static shape) and as f16/int8 OpenVINO IR, each sealed.

Every runtime here is OpenVINO on the CPU; torch trains and answers on the CPU in float32.
"""

import copy
import json
from dataclasses import replace

import numpy as np
import pytest
import torch

pytest.importorskip('openvino', reason='export needs the [openvino] extra')

import openvino as ov

from learn_lab.deep import export
from learn_lab.deep.field import FieldModel
from learn_lab.deep.fit import Schedule
from learn_lab.deep.operator import FNO
from learn_lab.deep.runtime import Runtime
from learn_lab.equivalence import Statistic, Tolerance, equivalence
from learn_lab.errors import Refused

from . import _openvino

CPU32 = Runtime(device='cpu', dtype='float32')
BATCH, GRID = 8, 16
#: f32 ONNX against torch f32: the same arithmetic in another kernel order, relative to the output scale.
SAME_PRECISION = 1e-4
#: How much worse than the f32 model's relative L2 to the planted truth a quantized model may be.
QUANTIZED = Tolerance(0.02, 'dimensionless')


def _truth(x):
    """A planted local operator: a 3-point smoothing in each axis, plus a pointwise square."""
    s = x + 0.5 * (np.roll(x, 1, -1) + np.roll(x, -1, -1) + np.roll(x, 1, -2) + np.roll(x, -1, -2))
    return s + 0.3 * x**2


RELATIVE_L2 = Statistic(
    'relative_l2', 'dimensionless', lambda p, x: float(np.linalg.norm(p - _truth(x)) / np.linalg.norm(_truth(x)))
)


@pytest.fixture(scope='module')
def fitted():
    x = np.random.default_rng(0).uniform(-1, 1, (256, 1, GRID, GRID))
    model = FieldModel(width=8, levels=2, schedule=Schedule(adam_steps=400, learning_rate=3e-3), runtime=CPU32)
    return model.fit(x, _truth(x))


@pytest.fixture(scope='module')
def queries():
    return np.random.default_rng(1).uniform(-0.95, 0.95, (4 * BATCH, 1, GRID, GRID))


@pytest.fixture(scope='module')
def exported(fitted, queries, tmp_path_factory):
    return export.to_onnx(fitted, tmp_path_factory.mktemp('export') / 'unet', example=queries[:BATCH])


def _batched(run):
    return lambda x: np.concatenate([run(x[i : i + BATCH]) for i in range(0, len(x), BATCH)])


def test_the_onnx_graph_answers_raw_inputs_as_the_fit_does(fitted, exported, queries):
    record = export.manifest(exported)
    assert record['family'] == 'field_model'
    assert record['input'] == {'name': 'x', 'shape': [BATCH, 1, GRID, GRID], 'dtype': 'float32'}
    assert set(record['payload']) == {'model.onnx'}
    run = _openvino.compile_on(ov.Core(), exported, 'CPU')
    assert run.precision == 'f32', 'an f32 export judged in bf16 measures the plugin, not the export'
    reference = fitted.predict(queries)
    scale = float(np.max(np.abs(reference)))
    assert np.max(np.abs(_batched(run)(queries) - reference)) < SAME_PRECISION * scale


def test_a_changed_byte_an_existing_path_and_a_stranger_example_are_refused(fitted, exported, queries, tmp_path):
    copy = tmp_path / 'copy'
    copy.mkdir()
    for name in ('model.onnx', 'manifest.json'):
        (copy / name).write_bytes((exported / name).read_bytes())
    data = bytearray((copy / 'model.onnx').read_bytes())
    data[-1] ^= 1
    (copy / 'model.onnx').write_bytes(bytes(data))
    with pytest.raises(Refused, match='does not match its hash'):
        export.manifest(copy)
    record = json.loads((exported / 'manifest.json').read_text(encoding='utf-8'))
    record['opset'] = 11
    (copy / 'manifest.json').write_text(json.dumps(record), encoding='utf-8')
    with pytest.raises(Refused, match='content hash'):
        export.manifest(copy)
    with pytest.raises(Refused, match='exists'):
        export.to_onnx(fitted, exported, example=queries[:BATCH])
    with pytest.raises(Refused, match='out_of_domain'):
        export.to_onnx(fitted, tmp_path / 'far', example=10 * queries[:BATCH])


def test_a_family_that_does_not_export_is_refused_by_name(tmp_path):
    x = np.random.default_rng(0).uniform(-1, 1, (4, 1, 16))
    fno = FNO(modes=(4,), width=4, layers=1, schedule=Schedule(adam_steps=1), runtime=CPU32).fit(x, x)
    with pytest.raises(Refused, match="fno fit does not export.*'field_model'"):
        export.to_onnx(fno, tmp_path / 'fno', example=x)


@pytest.mark.parametrize('precision', ['f16', 'int8'])
def test_a_quantized_model_is_sealed_and_judged_on_the_models_own_statistic(fitted, exported, queries, precision):
    path = export.quantize(exported, queries, precision=precision)
    assert path.name == f'unet.{precision}'
    record = export.manifest(path)
    assert record['precision'] == precision
    assert record['source'] == {'model.onnx': export.manifest(exported)['payload']['model.onnx']}
    run = _batched(_openvino.compile_on(ov.Core(), path, 'CPU'))
    report = equivalence(fitted.predict, run, queries, statistic=RELATIVE_L2, tolerance=QUANTIZED)
    assert report.max_deviation > 0  # a quantized graph is not the f32 graph
    with pytest.raises(Refused, match='quantize the f32 ONNX export'):
        export.quantize(path, queries, precision='f16')


def test_equivalence_refuses_an_export_with_perturbed_weights(fitted, queries, tmp_path):
    net = copy.deepcopy(fitted.net)  # PLANTED: a corrupted model the check must refuse
    generator = torch.Generator().manual_seed(0)
    with torch.no_grad():
        for parameter in net.parameters():
            parameter.add_(0.1 * parameter.abs().mean() * torch.randn(parameter.shape, generator=generator))
    corrupted = export.to_onnx(replace(fitted, net=net), tmp_path / 'corrupted', example=queries[:BATCH])
    run = _batched(_openvino.compile_on(ov.Core(), corrupted, 'CPU'))
    with pytest.raises(Refused, match='worse'):
        equivalence(fitted.predict, run, queries, statistic=RELATIVE_L2, tolerance=QUANTIZED)


def test_a_calibration_set_off_the_static_batch_is_refused(exported, queries):
    with pytest.raises(Refused, match='calibration inputs'):
        export.quantize(exported, queries[: BATCH + 1], precision='int8')


def test_a_device_openvino_does_not_list_raises_by_name_and_the_dgpu_is_never_the_gpu():
    with pytest.raises(LookupError, match='no NPU device'):
        _openvino.resolve(['CPU', 'GPU.0'], {'GPU.0': 'integrated'}, 'NPU')
    with pytest.raises(LookupError, match='no GPU device'):
        _openvino.resolve(['CPU', 'GPU.0'], {'GPU.0': 'discrete'}, 'GPU')
    assert _openvino.resolve(['GPU.0', 'GPU.1'], {'GPU.0': 'discrete', 'GPU.1': 'integrated'}, 'GPU') == 'GPU.1'

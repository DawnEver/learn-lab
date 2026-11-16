"""Hardware benchmark: fit time per step and inference latency/throughput per family x device x precision.

One command measures a declared set of families on this box and writes a JSON record plus a Markdown table, so
the same command run on another host (an A100, say) gives a comparable record::

    python -m learn_lab.deep.bench --size baseline --devices cpu,cuda,npu --out output/logs/<yy>/<mm>/<dd>/bench

Every timing is ``repeats`` wall-clock samples after ``warmup`` untimed calls, reported as median, p25, p75 and
min. ``fit`` is the wall time of a whole fit of ``steps`` Adam steps divided by ``steps`` (setup included, so it
is an upper bound on the step itself). ``infer`` is ``predict`` at batch 1 (latency) and at the size's large
batch (throughput, samples/s), host array in and host array out. The torch devices are ``cpu``, ``cuda`` and
``cuda:<n>``; ``npu`` is the OpenVINO NPU, reached through :mod:`learn_lab.deep.export` (fp32 ONNX and int8 IR,
static shapes, so each batch is its own export). A device that is not present is a row with status
``unavailable`` and its reason, never a fallback to another device; a family the export refuses is ``refused``.
A CUDA row carries the GPU utilisation sampled when it starts and ``contended`` when another process held the GPU.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

import numpy as np
import torch

from learn_lab.deep.field import FieldModel
from learn_lab.deep.fit import Schedule
from learn_lab.deep.operator import FNO, GraphOperator, TransformerOperator, Transolver
from learn_lab.deep.runtime import Runtime
from learn_lab.deep.surrogate import DeepEnsemble
from learn_lab.errors import refuse
from learn_lab.representation import PointGraph

try:  # the NPU leg only: the [openvino] extra
    import openvino as ov

    from learn_lab.deep import export
except ModuleNotFoundError:
    ov = export = None

__all__ = ['FAMILIES', 'FORMAT', 'SIZES', 'measure', 'render', 'write']

FORMAT: Final = 'learn-lab-bench/1'
SIZES: Final = ('tiny', 'baseline')
#: Utilisation (percent) at or above which another process is taken to hold the GPU.
CONTENDED_AT: Final = 10


@dataclass(frozen=True, slots=True)
class Problem:
    """One family at one size: how to fit it on a runtime, and its prediction on a batch of queries."""

    shape: str  # human-readable problem size
    batch: int  # the large (throughput) batch
    fit: Callable[[Runtime, int], object]  # (runtime, steps) -> fitted
    predict: Callable[[object, int], np.ndarray]  # (fitted, batch) -> outputs
    queries: Callable[[int], np.ndarray]  # batch -> raw inputs, for an export


def _mlp(size: str) -> Problem:
    n, d, hidden, batch = (64, 4, (16,), 32) if size == 'tiny' else (2048, 8, (256, 256, 256), 8192)
    rng = np.random.default_rng(0)
    x = rng.uniform(-1, 1, (n, d))
    y = np.sin(x).sum(1, keepdims=True)

    def fit(rt: Runtime, steps: int) -> object:
        return DeepEnsemble(hidden=hidden, members=1, schedule=Schedule(adam_steps=steps), runtime=rt).fit(x, y)

    def queries(b: int) -> np.ndarray:
        return rng.uniform(-0.9, 0.9, (b, d))

    return Problem(f'n={n} d={d} hidden={hidden}', batch, fit, lambda f, b: f.predict(queries(b)).mean, queries)


def _grid_problem(size: str, make: Callable[[Runtime, Schedule], object], name: str) -> Problem:
    n, grid, batch = (4, 16, 4) if size == 'tiny' else (64, 64, 64)
    rng = np.random.default_rng(0)
    x = rng.uniform(-1, 1, (n, 1, grid, grid))
    y = x + 0.5 * (np.roll(x, 1, -1) + np.roll(x, -1, -1)) + 0.3 * x**2

    def queries(b: int) -> np.ndarray:
        return rng.uniform(-0.9, 0.9, (b, 1, grid, grid))

    def fit(rt: Runtime, steps: int) -> object:
        return make(rt, Schedule(adam_steps=steps)).fit(x, y)

    return Problem(f'{name} n={n} grid={grid}x{grid}', batch, fit, lambda f, b: f.predict(queries(b)), queries)


def _fno(size: str) -> Problem:
    modes, width = ((4, 4), 8) if size == 'tiny' else ((12, 12), 32)

    def make(rt: Runtime, s: Schedule) -> FNO:
        return FNO(modes=modes, width=width, layers=4, schedule=s, runtime=rt)

    return _grid_problem(size, make, f'modes={modes} width={width}')


def _field(size: str) -> Problem:
    width = 8 if size == 'tiny' else 16

    def make(rt: Runtime, s: Schedule) -> FieldModel:
        return FieldModel(width=width, levels=2 if size == 'tiny' else 3, schedule=s, runtime=rt)

    return _grid_problem(size, make, f'U-Net width={width}')


def _cloud(size: str, points: int) -> tuple[int, int, np.ndarray, np.ndarray, np.ndarray, Callable]:
    n, batch = (4, 4) if size == 'tiny' else (32, 64)
    rng = np.random.default_rng(0)
    coords = rng.uniform(0, 1, (points, 2))
    x = rng.uniform(-1, 1, (n, 1, points))
    y = np.cos(3 * coords[:, 0])[None, None] * x + 0.2 * x**2

    def queries(b: int) -> np.ndarray:
        return rng.uniform(-0.9, 0.9, (b, 1, points))

    return n, batch, coords, x, y, queries


def _transolver(size: str) -> Problem:
    points, width = (64, 16) if size == 'tiny' else (2048, 64)
    n, batch, coords, x, y, queries = _cloud(size, points)

    def fit(rt: Runtime, steps: int) -> object:
        model = Transolver(
            width=width, layers=4 if size != 'tiny' else 1, schedule=Schedule(adam_steps=steps), runtime=rt
        )
        return model.fit(x, coords, y)

    shape = f'n={n} points={points} width={width} slices=32'
    return Problem(shape, batch, fit, lambda f, b: f.predict(queries(b), coords), queries)


def _transformer(size: str) -> Problem:
    points, width = (32, 8) if size == 'tiny' else (512, 32)
    n, batch, coords, x, y, queries = _cloud(size, points)

    def fit(rt: Runtime, steps: int) -> object:
        model = TransformerOperator(width=width, heads=4, layers=2, schedule=Schedule(adam_steps=steps), runtime=rt)
        return model.fit(x, coords, y)

    shape = f'n={n} points={points} width={width} softmax'
    return Problem(shape, batch, fit, lambda f, b: f.predict(queries(b), coords), queries)


def _graph_operator(size: str) -> Problem:
    points, k = (32, 4) if size == 'tiny' else (1024, 8)
    n, batch, coords, x, y, queries = _cloud(size, points)
    distance = np.linalg.norm(coords[:, None] - coords[None], axis=-1)
    nearest = np.argsort(distance, axis=1)[:, 1 : k + 1]
    edges = np.stack([nearest.ravel(), np.repeat(np.arange(points), k)]).astype(np.int64)
    graph = PointGraph(coords, edges, np.full(edges.shape[1], 1.0 / k))

    def fit(rt: Runtime, steps: int) -> object:
        return GraphOperator(width=16, layers=2, schedule=Schedule(adam_steps=steps), runtime=rt).fit(x, graph, y)

    return Problem(f'n={n} points={points} knn={k}', batch, fit, lambda f, b: f.predict(queries(b), graph), queries)


#: The benchmarked families by name: ``size -> Problem``.
FAMILIES: Final[dict[str, Callable[[str], Problem]]] = {
    'mlp': _mlp,
    'fno': _fno,
    'field': _field,
    'transolver': _transolver,
    'transformer': _transformer,
    'graph_operator': _graph_operator,
}


def _stats(samples: Sequence[float]) -> dict:
    q = np.percentile(samples, [25, 75])
    return {
        'median': statistics.median(samples),
        'p25': float(q[0]),
        'p75': float(q[1]),
        'min': min(samples),
        'samples': list(samples),
    }


def _timed(call: Callable[[], object], warmup: int, repeats: int) -> list[float]:
    for _ in range(warmup):
        call()
    out = []
    for _ in range(repeats):
        start = time.perf_counter()
        call()
        out.append(time.perf_counter() - start)
    return out


def _gpu_sample() -> dict:
    """GPU utilisation and compute processes now, from nvidia-smi; empty when it is absent."""
    smi = shutil.which('nvidia-smi')
    if smi is None:
        return {}
    query = [smi, '--query-gpu=utilization.gpu,memory.used', '--format=csv,noheader,nounits']
    util, memory = subprocess.run(query, capture_output=True, text=True, check=True, timeout=30).stdout.split(',')[:2]
    apps = [smi, '--query-compute-apps=pid', '--format=csv,noheader']
    others = [
        p.strip()
        for p in subprocess.run(apps, capture_output=True, text=True, check=True, timeout=30).stdout.splitlines()
        if p.strip() and p.strip() != str(os.getpid())
    ]
    return {'utilisation_pct': int(util), 'memory_used_mib': int(memory), 'other_processes': others}


def _torch_device(device: str) -> str | None:
    """Why *device* cannot run here, or None when it can."""
    if device == 'cpu':
        return None
    if device == 'cuda' or device.startswith('cuda:'):
        if not torch.cuda.is_available():
            return 'this torch sees no CUDA device'
        index = int(device.partition(':')[2] or 0)
        if index >= torch.cuda.device_count():
            return f'{device} asked for; this torch sees {torch.cuda.device_count()} CUDA device(s)'
        return None
    return f'{device!r} is not a device this benchmark knows (cpu, cuda[:n], npu)'


def _openvino_core() -> tuple[object | None, str | None]:
    if ov is None:
        return None, 'openvino is not installed (the [openvino] extra)'
    core = ov.Core()
    if 'NPU' not in core.available_devices:
        return None, f'OpenVINO lists no NPU; it lists {core.available_devices}'
    return core, None


def _torch_rows(family: str, problem: Problem, device: str, dtype: str, steps: int, warmup: int, repeats: int):
    base = {'family': family, 'device': device, 'precision': dtype, 'shape': problem.shape}
    if device.startswith('cuda'):
        sample = _gpu_sample()
        base |= {
            'gpu_at_start': sample,
            'contended': bool(sample)
            and (sample['utilisation_pct'] >= CONTENDED_AT or bool(sample['other_processes'])),
        }
    rt = Runtime(device=device, dtype=dtype)
    fits: list[object] = []
    seconds = _timed(lambda: fits.append(problem.fit(rt, steps)), warmup, repeats)
    yield base | {
        'phase': 'fit',
        'status': 'measured',
        'steps': steps,
        'per_step_s': _stats([s / steps for s in seconds]),
    }
    fitted = fits[-1]
    for batch in (1, problem.batch):
        timed = _timed(lambda b=batch: problem.predict(fitted, b), warmup, repeats)
        yield base | _infer_row(batch, timed)


def _infer_row(batch: int, seconds: list[float]) -> dict:
    row = {'phase': 'infer', 'status': 'measured', 'batch': batch, 'latency_s': _stats(seconds)}
    return row | {'throughput_per_s': batch / statistics.median(seconds)}


def _npu_rows(family: str, problem: Problem, core: object, steps: int, warmup: int, repeats: int, work: Path):
    fitted = problem.fit(Runtime(device='cpu', dtype='float32'), steps)
    for batch in (1, problem.batch):
        base = {'family': family, 'device': 'npu', 'phase': 'infer', 'shape': problem.shape, 'batch': batch}
        queries = problem.queries(batch)
        try:
            onnx = export.to_onnx(fitted, work / f'{family}-{batch}', example=queries)
        except ValueError as exc:
            yield base | {'precision': 'fp32,int8', 'status': 'refused', 'reason': str(exc)}
            continue
        calibration = np.concatenate([problem.queries(batch) for _ in range(max(1, 16 // batch))])
        paths = {
            'fp32': onnx / export.ONNX,
            'int8': export.quantize(onnx, calibration, precision='int8') / export.IR[0],
        }
        for precision, path in paths.items():
            row = base | {'precision': precision}
            try:
                start = time.perf_counter()
                request = core.compile_model(core.read_model(path), 'NPU').create_infer_request()
                row['compile_s'] = time.perf_counter() - start
            except RuntimeError as exc:
                yield row | {'status': 'compile_failed', 'reason': str(exc).strip()[:500]}
                continue
            x = np.ascontiguousarray(queries, np.float32)
            yield row | _infer_row(batch, _timed(lambda r=request, q=x: r.infer({0: q})[0].copy(), warmup, repeats))


def environment(core: object | None = None) -> dict:
    """The host, its devices and the library versions a record was measured with."""
    env = {
        'host': platform.node(),
        'platform': platform.platform(),
        'cpu': _cpu_name(),
        'cpu_logical_cores': os.cpu_count(),
        'python': platform.python_version(),
        'numpy': np.__version__,
        'torch': torch.__version__,
        'torch_threads': torch.get_num_threads(),
        'cuda_runtime': torch.version.cuda,
        'cudnn': torch.backends.cudnn.version() if torch.cuda.is_available() else None,
    }
    if torch.cuda.is_available():
        env['gpus'] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
        if smi := shutil.which('nvidia-smi'):
            query = [smi, '--query-gpu=driver_version', '--format=csv,noheader']
            env['nvidia_driver'] = subprocess.run(
                query, capture_output=True, text=True, check=False, timeout=30
            ).stdout.strip()
    if ov is not None:
        env['openvino'] = ov.__version__
    if core is not None:
        env['npu'] = {
            k: str(core.get_property('NPU', k)) for k in ('FULL_DEVICE_NAME', 'NPU_DRIVER_VERSION', 'DEVICE_GOPS')
        }
    return env


def _cpu_name() -> str:
    if platform.system() == 'Windows':
        query = ['powershell', '-NoProfile', '-Command', '(Get-CimInstance Win32_Processor).Name']
        try:
            return subprocess.run(query, capture_output=True, text=True, check=False, timeout=30).stdout.strip()
        except OSError:
            pass
    elif Path('/proc/cpuinfo').exists():
        for line in Path('/proc/cpuinfo').read_text().splitlines():
            if line.startswith('model name'):
                return line.split(':', 1)[1].strip()
    return platform.processor()


def measure(
    families: Sequence[str],
    devices: Sequence[str],
    *,
    size: str = 'baseline',
    precisions: Sequence[str] = ('float32', 'float64'),
    steps: int = 50,
    warmup: int = 1,
    repeats: int = 5,
    work: Path | None = None,
) -> dict:
    """The benchmark record of *families* on *devices*; *work* holds the NPU exports (a fresh directory)."""
    if unknown := set(families) - FAMILIES.keys():
        refuse(f'unknown families {sorted(unknown)}; the benchmarked families are {list(FAMILIES)}')
    if size not in SIZES:
        refuse(f'a size is one of {SIZES}, got {size!r}')
    if min(steps, repeats) < 1 or warmup < 0:
        refuse(f'steps and repeats are >= 1 and warmup >= 0, got {steps}, {repeats}, {warmup}')
    core, npu_reason = _openvino_core() if 'npu' in devices else (None, None)
    started = datetime.now(UTC)
    rows: list[dict] = []
    for family in families:
        problem = FAMILIES[family](size)
        for device in devices:
            if device == 'npu':
                if core is None:
                    rows.append({'family': family, 'device': device, 'status': 'unavailable', 'reason': npu_reason})
                    continue
                with tempfile.TemporaryDirectory(dir=work) as tmp:
                    rows.extend(_npu_rows(family, problem, core, steps, warmup, repeats, Path(tmp)))
                continue
            if reason := _torch_device(device):
                rows.append({'family': family, 'device': device, 'status': 'unavailable', 'reason': reason})
                continue
            for dtype in precisions:
                rows.extend(_torch_rows(family, problem, device, dtype, steps, warmup, repeats))
    return {
        'format': FORMAT,
        'started': started.isoformat(),
        'seconds': (datetime.now(UTC) - started).total_seconds(),
        'settings': {'size': size, 'steps': steps, 'warmup': warmup, 'repeats': repeats, 'devices': list(devices)},
        'environment': environment(core),
        'rows': rows,
    }


def _ms(stats: dict) -> str:
    return f'{1e3 * stats["median"]:.3g} [{1e3 * stats["p25"]:.3g}, {1e3 * stats["p75"]:.3g}]'


def render(record: dict) -> str:
    """The record as Markdown: the environment, then one table row per measurement (ms: median [p25, p75])."""
    env = record['environment']
    lines = [f'# learn-lab benchmark {record["started"]}', '', f'Settings: `{json.dumps(record["settings"])}`', '']
    lines += [f'- {k}: {v}' for k, v in env.items()] + ['']
    lines += ['| family | device | precision | phase | batch | median [p25, p75] ms | per s | note |']
    lines += ['|---|---|---|---|---|---|---|---|']
    for r in record['rows']:
        note = r.get('reason', '')
        if 'contended' in r:
            g = r['gpu_at_start']
            note = (
                f'CONTENDED util {g["utilisation_pct"]}%' if r['contended'] else f'clean util {g["utilisation_pct"]}%'
            )
        if 'compile_s' in r:
            note = f'compile {r["compile_s"]:.2f} s'
        if r['status'] != 'measured':
            lines.append(
                f'| {r["family"]} | {r["device"]} | {r.get("precision", "")} | {r.get("phase", "")} '
                f'| {r.get("batch", "")} | {r["status"]} | | {note[:120]} |'
            )
        elif r['phase'] == 'fit':
            lines.append(
                f'| {r["family"]} | {r["device"]} | {r["precision"]} | fit/step | | '
                f'{_ms(r["per_step_s"])} | {1 / r["per_step_s"]["median"]:.3g} steps | {note} |'
            )
        else:
            lines.append(
                f'| {r["family"]} | {r["device"]} | {r["precision"]} | infer | {r["batch"]} | '
                f'{_ms(r["latency_s"])} | {r["throughput_per_s"]:.3g} samples | {note} |'
            )
    return '\n'.join(lines) + '\n'


def write(record: dict, out: Path) -> tuple[Path, Path]:
    """``bench-<HHMMSS>.json`` and ``.md`` of *record* under *out*; returns both paths."""
    out.mkdir(parents=True, exist_ok=True)
    stem = out / f'bench-{datetime.fromisoformat(record["started"]):%H%M%S}'
    json_path, md_path = stem.with_suffix('.json'), stem.with_suffix('.md')
    json_path.write_text(json.dumps(record, indent=1), encoding='utf-8')
    md_path.write_text(render(record), encoding='utf-8')
    return json_path, md_path


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog='python -m learn_lab.deep.bench', description=__doc__.split('\n')[0])
    parser.add_argument('--families', default=','.join(FAMILIES))
    parser.add_argument('--devices', default='cpu,cuda,npu')
    parser.add_argument('--precisions', default='float32,float64', help='torch dtypes; the NPU runs fp32 and int8')
    parser.add_argument('--size', default='baseline', choices=SIZES)
    parser.add_argument('--steps', type=int, default=50)
    parser.add_argument('--warmup', type=int, default=1)
    parser.add_argument('--repeats', type=int, default=5)
    parser.add_argument('--out', type=Path, default=Path('output/logs') / f'{datetime.now(UTC):%y/%m/%d}' / 'bench')
    a = parser.parse_args(argv)
    a.out.mkdir(parents=True, exist_ok=True)
    record = measure(
        a.families.split(','),
        a.devices.split(','),
        size=a.size,
        precisions=a.precisions.split(','),
        steps=a.steps,
        warmup=a.warmup,
        repeats=a.repeats,
        work=a.out,
    )
    for path in write(record, a.out):
        sys.stdout.write(f'{path}\n')


if __name__ == '__main__':
    main()

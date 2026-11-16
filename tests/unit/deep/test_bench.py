"""The benchmark record: its schema, devices that are not present, positive timings and honoured repeat counts.

No timing threshold: a timing is only checked to be a positive count of the asked repeats.
"""

import json

import pytest

from learn_lab.deep import bench
from learn_lab.errors import Refused

REPEATS = 3


@pytest.fixture(scope='module')
def record():
    return bench.measure(
        ['mlp', 'fno'],
        ['cpu', 'cuda:97', 'gpu-that-is-not'],
        size='tiny',
        precisions=['float32'],
        steps=2,
        warmup=1,
        repeats=REPEATS,
    )


def test_the_record_carries_its_format_settings_environment_and_rows(record):
    assert record['format'] == bench.FORMAT
    assert record['settings'] == {
        'size': 'tiny',
        'steps': 2,
        'warmup': 1,
        'repeats': REPEATS,
        'devices': ['cpu', 'cuda:97', 'gpu-that-is-not'],
    }
    assert {'cpu', 'torch', 'python', 'platform'} <= record['environment'].keys()
    json.dumps(record)  # serialisable as it stands
    measured = [r for r in record['rows'] if r['status'] == 'measured']
    assert {(r['family'], r['phase'], r.get('batch')) for r in measured} == {
        (f, p, b) for f, b in (('mlp', 32), ('fno', 4)) for p, b in (('fit', None), ('infer', 1), ('infer', b))
    }


def test_a_device_that_is_not_present_is_reported_unavailable_with_its_reason(record):
    absent = [r for r in record['rows'] if r['device'] in {'cuda:97', 'gpu-that-is-not'}]
    assert len(absent) == 4  # each family x each absent device, never measured elsewhere
    assert all(r['status'] == 'unavailable' and r['reason'] for r in absent)


def test_timings_are_positive_and_the_repeat_count_is_honoured(record):
    for r in record['rows']:
        if r['status'] == 'measured':
            stats = r['per_step_s'] if r['phase'] == 'fit' else r['latency_s']
            assert len(stats['samples']) == REPEATS
            assert all(s > 0 for s in stats['samples'])
            assert stats['min'] <= stats['p25'] <= stats['median'] <= stats['p75']


def test_the_markdown_and_json_are_written_beside_each_other(record, tmp_path):
    json_path, md_path = bench.write(record, tmp_path / 'bench')
    assert json.loads(json_path.read_text(encoding='utf-8')) == record
    table = md_path.read_text(encoding='utf-8')
    assert table.count('| mlp | cpu | float32 |') == 3 and 'unavailable' in table


def test_an_unknown_family_or_size_is_refused():
    with pytest.raises(Refused, match='unknown families'):
        bench.measure(['nope'], ['cpu'])
    with pytest.raises(Refused, match='size'):
        bench.measure(['mlp'], ['cpu'], size='huge')

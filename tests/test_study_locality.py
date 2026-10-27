import json
import sys

from bench.study import run


def test_locality_records_actual_child_affinity_and_mapping_observation(tmp_path):
    prefix = tmp_path / 'observed'
    assert run([sys.executable, '-c', 'import time; time.sleep(.4)'], prefix, telemetry=False, gpu_exclusive=False) == 0
    result = json.loads((tmp_path / 'observed.locality.json').read_text())
    assert result['status'] == 'observed'
    assert result['threads'][0]['cpus_allowed_list']
    assert 'numa_maps' in result


def test_short_process_does_not_fabricate_locality(tmp_path):
    prefix = tmp_path / 'short'
    assert run([sys.executable, '-c', 'pass'], prefix, telemetry=False, gpu_exclusive=False) == 0
    result = json.loads((tmp_path / 'short.locality.json').read_text())
    assert result['status'] == 'unavailable'
    assert 'completed before' in result['reason']

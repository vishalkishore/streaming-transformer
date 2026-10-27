import itertools
import json
from pathlib import Path

import pytest

from bench.gates import check_native_gate, check_followup_gate
from bench.provenance import file_hash


@pytest.fixture
def evidence(tmp_path):
    """Synthetic result files test gate rejection, never stand in for GPU evidence."""
    rows = []
    variants = [('cpu', 1), ('cuda', 4), ('cublas', 4), ('graph', 4),
                ('cublas-graph', 4), ('fused', 1), ('fused', 4),
                ('persistent', 1), ('persistent', 4)]
    conditions = [(size, mode, backend, blocks, False)
                  for size, mode, (backend, blocks) in itertools.product(
                      ['small', 'main', 'large'], ['fp32', 'mixed'], variants)]
    conditions += [('main', mode, 'persistent', blocks, True)
                   for mode, blocks in itertools.product(['fp32', 'mixed'], [1, 4])]
    for size, mode, backend, blocks, stress in conditions:
        name = f'{size}-{mode}-{"stress" if stress else backend}-{blocks}'
        config = json.loads(Path(f'configs/{size}.json').read_text())
        count = 1000000 if stress else 3 * config['window'] + 9
        prefix = tmp_path / name
        command = ['synthetic-bench', '--events', str(count), '--output', str(prefix)]
        if stress:
            oracle = tmp_path / f'oracle-{mode}'
            oracle.mkdir(exist_ok=True)
            (oracle / 'manifest.json').write_text(json.dumps(dict(
                format='stream-sparse-oracle-v1', events=count, reset_every=10007)))
            command += ['--expected', str(oracle), '--reset-every', '10007']
        result = dict(provenance=dict(executable_sha256='test-hash'), mode='correctness',
                      dropped=0, completed=count, config=config, precision=mode,
                      backend=backend, blocks=blocks)
        if stress:
            result.update(cpu_oracle_active=False, expected_manifest_sha256=file_hash(oracle / 'manifest.json'))
        for metric in ['scores', 'keys', 'values', 'hidden']:
            result[metric] = dict(elements=1, failed=0, nonfinite=0)
        Path(str(prefix) + '.json').write_text(json.dumps(result))
        rows.append(dict(name=name, returncode=0, command=command))
    path = tmp_path / 'summary.json'
    path.write_text(json.dumps(rows))
    return path, rows


def test_requires_complete_distinct_matrix(evidence):
    path, rows = evidence
    assert len(check_native_gate(path, 'test-hash')) == 58
    path.write_text(json.dumps(rows[1:]))
    with pytest.raises(RuntimeError, match='54-condition'):
        check_native_gate(path, 'test-hash')
    path.write_text(json.dumps(rows + [rows[0]]))
    with pytest.raises(RuntimeError, match='distinct'):
        check_native_gate(path, 'test-hash')


def test_cpu_gate_accepts_absent_gpu_only_metadata(evidence):
    path, rows = evidence
    cpu = next(row for row in rows if row['name'] == 'small-fp32-cpu-1')
    command = cpu['command']
    result_path = Path(command[command.index('--output') + 1] + '.json')
    result = json.loads(result_path.read_text())
    result.pop('blocks')
    result_path.write_text(json.dumps(result))
    assert len(check_native_gate(path, 'test-hash')) == 58


def test_followup_receipt_requires_each_stage_once(tmp_path):
    path = tmp_path / 'followup.json'
    path.write_text(json.dumps([dict(name='ctest', returncode=0)]))
    with pytest.raises(RuntimeError, match='All six'):
        check_followup_gate(path, 'test-hash')


@pytest.mark.parametrize('mutation,pattern', [
    ('hash', 'executable differs'), ('short', 'Incomplete event'),
    ('wrong-backend', 'Mislabeled'), ('empty-cache', 'Numerical gate'),
    ('nonfinite', 'Numerical gate'), ('drift', 'State energy gate')])
def test_rejects_untrustworthy_result_artifacts(evidence, mutation, pattern):
    path, rows = evidence
    command = rows[-1]['command']
    result_path = Path(command[command.index('--output') + 1] + '.json')
    result = json.loads(result_path.read_text())
    if mutation == 'hash':
        result['provenance']['executable_sha256'] = 'old-binary'
    elif mutation == 'short':
        result['completed'] -= 1
    elif mutation == 'wrong-backend':
        result['backend'] = 'cpu'
    elif mutation == 'empty-cache':
        result['keys']['elements'] = 0
    elif mutation == 'nonfinite':
        result['scores']['nonfinite'] = 1
    else:
        result['cache_relative_l2_gate'] = dict(failed=1, nonfinite=0)
    result_path.write_text(json.dumps(result))
    with pytest.raises(RuntimeError, match=pattern):
        check_native_gate(path, 'test-hash')

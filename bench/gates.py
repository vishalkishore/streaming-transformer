"""Check native correctness evidence before permitting performance studies."""
import itertools
import json
from pathlib import Path

from bench.provenance import file_hash
from reference.numerical import POLICY


def check_result(command, executable_hash):
    def option(name):
        return command[command.index(name) + 1]

    prefix = command[command.index('--output') + 1]
    result = json.loads(Path(prefix + '.json').read_text())
    if result['provenance']['executable_sha256'] != executable_hash:
        raise RuntimeError('Gate executable differs from current benchmark executable')
    count = int(command[command.index('--events') + 1])
    if result['mode'] != 'correctness' or result['dropped'] or result['completed'] != count:
        raise RuntimeError(f'Incomplete correctness evidence: {prefix}')
    config = json.loads(Path(option('--config')).read_text())
    fixture = Path(option('--weights'))
    manifest = json.loads((fixture / 'manifest.json').read_text())
    trace = Path(option('--trace')) if '--trace' in command else fixture / 'trace.bin'
    if (result['config'] != config or result['precision'] != manifest['mode']
            or result['checkpoint_sha256'] != manifest['sha256']
            or result['trace_sha256'] != file_hash(trace)
            or result['backend'] != option('--backend')
            or result.get('blocks', 1 if result['backend'] == 'cpu' else None) != int(option('--blocks'))):
        raise RuntimeError(f'Mislabeled or stale correctness evidence: {prefix}')
    for metric in ['scores', 'keys', 'values', 'hidden', 'cache_relative_l2_gate', 'hidden_relative_l2_gate']:
        errors = result[metric]
        needs_elements = metric in ['scores', 'keys', 'values', 'hidden'] or result['precision'] == 'mixed'
        if (needs_elements and errors['elements'] <= 0) or errors['failed'] or errors['nonfinite']:
            raise RuntimeError(f'Numerical gate failed: {prefix} {metric}')
    return result


def check_followup_gate(path, executable_hash):
    rows = json.loads(Path(path).read_text())
    stages = ['ctest', 'memcheck', 'synccheck', 'extended', 'windows', 'torch']
    if sorted(row['name'] for row in rows) != sorted(stages) or any(row['returncode'] for row in rows):
        raise RuntimeError('All six sequential followup validation stages must pass')
    extended = json.loads(Path('results/extended-validation/summary.json').read_text())
    kinds = ['random', 'zeros', 'constant', 'impulses', 'alternating', 'repeated']
    required = set(itertools.product(['small', 'main', 'large'], ['fp32', 'mixed'], kinds))
    identities = [(row['size'], row['mode'], row['kind']) for row in extended]
    if len(identities) != len(required) or set(identities) != required:
        raise RuntimeError('Complete independent/adversarial trace coverage is required')
    for row in extended:
        if not row['acceptance']['passed'] or row['reference_only']:
            raise RuntimeError('Independent/adversarial trace gate has not passed')
        variants = [('cuda', 4), ('cublas', 4), ('persistent', 1), ('persistent', 4)]
        expected = set(variants) if row['kind'] != 'random' else set()
        actual = []
        for native in row['native']:
            if native['returncode']:
                raise RuntimeError('Native adversarial check failed')
            result = check_result(native['command'], executable_hash)
            actual.append((result['backend'], result['blocks']))
        if len(actual) != len(expected) or set(actual) != expected:
            raise RuntimeError('Native adversarial backend coverage is incomplete')
    windows = json.loads(Path('results/window-validation/summary.json').read_text())
    required = {f'{size}-w{window}-{mode}-{backend}-{blocks}'
                for size, window, mode, (backend, blocks) in itertools.product(
                    ['small', 'main', 'large'], [32, 64, 128, 256, 512], ['fp32', 'mixed'],
                    [('cuda', 4), ('cublas-graph', 4), ('persistent', 1), ('persistent', 4)])}
    if len(windows) != len(required) or {row['name'] for row in windows} != required:
        raise RuntimeError('All 120 window correctness cases are required')
    for row in windows:
        if row['returncode']:
            raise RuntimeError('Window correctness failed')
        check_result(row['command'], executable_hash)
    torch_rows = json.loads(Path('results/torch-validation-final/summary.json').read_text())
    required = {f'{size}-{mode}-{backend}' for size, mode, backend in itertools.product(
        ['small', 'main', 'large'], ['fp32', 'mixed'], ['eager', 'compile', 'graph'])}
    if len(torch_rows) != len(required) or {row['name'] for row in torch_rows} != required:
        raise RuntimeError('All 18 PyTorch correctness cases are required')
    for row in torch_rows:
        if row['returncode']:
            raise RuntimeError('PyTorch correctness failed')
        command = row['command']
        result = json.loads(Path(command[command.index('--output') + 1] + '.json').read_text())
        fixture = Path(command[command.index('--fixture') + 1])
        manifest = json.loads((fixture / 'manifest.json').read_text())
        trace = fixture / 'trace.bin'
        if (not result['correctness_checked'] or result['script_sha256'] != file_hash('bench/torch_bench.py')
                or result['acceptance_policy'] != POLICY or not result['inductor_emulate_precision_casts']
                or result['config'] != manifest['config'] or result['precision'] != manifest['mode']
                or result['checkpoint_sha256'] != manifest['sha256'] or result['trace_sha256'] != file_hash(trace)
                or result['events'] != trace.stat().st_size // (4 * manifest['config']['features'])
                or result['tf32'] or result['fp16_accumulation'] or result['fp16_reduced_precision_reduction']
                or result['inductor_use_fast_math']
                or any(value for key, value in result['failures'].items() if key != 'hidden_raw')):
            raise RuntimeError('PyTorch gate is stale or failed')
    policy = json.loads(Path('results/numerical-policy-confirmation.json').read_text())
    required = set(itertools.product(['small', 'main', 'large'], [42, 17, 314], ['random', 'constant']))
    identities = [(row['size'], row['seed'], row['kind']) for row in policy['cases']]
    if (policy['status'] != 'passed' or policy['policy'] != POLICY or len(identities) != len(required)
            or set(identities) != required or not all(row['passed'] for row in policy['cases'])):
        raise RuntimeError('Independent numerical-policy confirmation has not passed')


def check_native_gate(path, executable_hash):
    rows = json.loads(Path(path).read_text())
    variants = [('cpu', 1), ('cuda', 4), ('cublas', 4), ('graph', 4),
                ('cublas-graph', 4), ('fused', 1), ('fused', 4),
                ('persistent', 1), ('persistent', 4)]
    required = {f'{size}-{mode}-{backend}-{blocks}'
                for size, mode, (backend, blocks) in itertools.product(
                    ['small', 'main', 'large'], ['fp32', 'mixed'], variants)}
    required |= {f'main-{mode}-stress-{blocks}'
                 for mode, blocks in itertools.product(['fp32', 'mixed'], [1, 4])}
    names = [row['name'] for row in rows]
    if len(names) != len(set(names)) or not required.issubset(names):
        raise RuntimeError('Full 54-condition matrix and four distinct stress cases are required')
    for row in rows:
        if row['returncode'] != 0:
            raise RuntimeError(f'Correctness gate failed: {row["name"]}')
        command = row['command']

        def option(name, default=None):
            return command[command.index(name) + 1] if name in command else default

        result = json.loads(Path(option('--output') + '.json').read_text())
        if result['provenance']['executable_sha256'] != executable_hash:
            raise RuntimeError('Gate executable differs from current benchmark executable')
        if result['mode'] != 'correctness' or result['dropped'] != 0:
            raise RuntimeError(f'Incomplete correctness run: {row["name"]}')
        stress = '-stress-' in row['name']
        count = int(option('--events'))
        if result['completed'] != count or (stress and count < 1000000):
            raise RuntimeError(f'Incomplete event gate: {row["name"]}')
        size = row['name'].split('-')[0]
        precision = row['name'].split('-')[1]
        backend = 'persistent' if stress else row['name'][len(size) + len(precision) + 2:].rsplit('-', 1)[0]
        blocks = int(row['name'].rsplit('-', 1)[1])
        config = json.loads(Path(f'configs/{size}.json').read_text())
        if (result['config'] != config or result['precision'] != precision
                or result['backend'] != backend
                or result.get('blocks', 1 if backend == 'cpu' else None) != blocks):
            raise RuntimeError(f'Mislabeled gate evidence: {row["name"]}')
        if not stress and count != 3 * config['window'] + 9:
            raise RuntimeError(f'Fixture coverage is incomplete: {row["name"]}')
        if stress:
            source = option('--expected')
            if source in [None, 'fixture', 'cpu'] or result.get('cpu_oracle_active', True) or option('--reset-every') != '10007':
                raise RuntimeError(f'Stress requires a precomputed CPU oracle and resets: {row["name"]}')
            manifest = json.loads((Path(source) / 'manifest.json').read_text())
            if (manifest['format'] != 'stream-sparse-oracle-v1' or manifest['events'] != count
                    or manifest['reset_every'] != 10007
                    or result.get('expected_manifest_sha256') != file_hash(Path(source) / 'manifest.json')):
                raise RuntimeError(f'Stored stress oracle differs from replay: {row["name"]}')
        for metric in ['scores', 'keys', 'values', 'hidden']:
            errors = result[metric]
            if errors['elements'] <= 0 or errors['failed'] or errors['nonfinite']:
                raise RuntimeError(f'Numerical gate failed: {row["name"]} {metric}')
        for metric in ['cache_relative_l2_gate', 'hidden_relative_l2_gate']:
            if metric in result and (result[metric]['failed'] or result[metric]['nonfinite']):
                raise RuntimeError(f'State energy gate failed: {row["name"]} {metric}')
    return rows

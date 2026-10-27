"""Serial, resumable native/PyTorch studies with lossless samples and telemetry."""
import argparse
import hashlib
import itertools
import json
import math
import os
import random
import subprocess
import sys
import time
from pathlib import Path

from bench.study import run
from bench.gates import check_native_gate, check_followup_gate
from bench.provenance import native_libraries, file_hash


def read(path):
    return json.loads(Path(path).read_text())


def cached_result(prefix, identity):
    stamp_path = Path(str(prefix) + '.stamp.json')
    if not stamp_path.exists():
        return None
    saved = read(stamp_path)
    artifacts = saved.pop('artifacts', {})
    if saved != identity or set(artifacts) != {'.json', '.csv'}:
        return None
    for suffix, expected in artifacts.items():
        path = Path(str(prefix) + suffix)
        if not path.is_file() or file_hash(path) != expected:
            return None
    return read(Path(str(prefix) + '.json'))


def archive_attempt(prefix, records):
    prefix = Path(prefix)
    previous = [path for path in prefix.parent.glob(prefix.name + '.*') if path.is_file()]
    if not previous:
        return None
    archive = prefix.parent / 'attempts' / f'{prefix.name}-{time.time_ns()}'
    archive.mkdir(parents=True)
    for path in previous:
        path.rename(archive / path.name)
    for record in records:
        if record['label'] == prefix.name and 'archive' not in record:
            record['archive'] = str(archive)
    return archive


class Study:
    def __init__(self, args):
        self.a = args
        self.root = args.output
        self.root.mkdir(parents=True, exist_ok=True)
        self.binary_hash = hashlib.sha256(Path(args.binary).read_bytes()).hexdigest()
        self.libraries = native_libraries(args.binary)
        self.gpu_packages = subprocess.check_output(
            ['uv', 'pip', 'freeze', '--python', args.gpu_python], text=True) if args.torch else None
        self.torch_hash = hashlib.sha256(b''.join(Path(p).read_bytes() for p in [
            'bench/torch_bench.py', 'bench/provenance.py', 'reference/numerical.py',
            'bench/requirements-blackwell.lock'])).hexdigest()
        self.records = read(self.root / 'runs.json') if (self.root / 'runs.json').exists() else []
        self.selection = read(self.root / 'selection.json') if (self.root / 'selection.json').exists() else {}

    def execute(self, label, size, mode, backend, repeat, extra=None, events=None, controlled=True, group=None):
        prefix = self.root / label
        count = events or self.a.events
        core = self.a.worker_core if controlled else -1
        extra = list(extra or [])
        measurement = 'service'
        if '--mode' in extra:
            index = extra.index('--mode')
            measurement = extra[index + 1]
            del extra[index:index + 2]
        if backend.startswith('torch-'):
            command = [self.a.gpu_python, '-m', 'bench.torch_bench', '--fixture', f'fixtures/{size}-{mode}',
                       '--backend', backend[6:], '--events', str(count), '--warmup', '1000',
                       '--output', str(prefix), '--worker-core', str(core)]
        else:
            command = [self.a.binary, '--config', f'configs/{size}.json', '--weights', f'fixtures/{size}-{mode}',
                       '--backend', backend, '--events', str(count), '--warmup', '1000',
                       '--mode', measurement, '--output', str(prefix), '--worker-core', str(core),
                       '--producer-core', str(self.a.producer_core if controlled else -1),
                       '--clock-regime', 'default-dynamic-affinity' if controlled else 'default-dynamic-uncontrolled']
        command += extra
        stamp = dict(command=command, native_executable_sha256=self.binary_hash)
        discard = self.a.transport_warmup if measurement in ['ping-pong', 'gpu-ping', 'empty', 'empty-graph', 'copy'] else 0
        stamp['transport_warmup_discard'] = discard
        stamp['native_libraries'] = self.libraries
        fixture = Path(f'fixtures/{size}-{mode}')
        stamp['manifest_sha256'] = hashlib.sha256((fixture / 'manifest.json').read_bytes()).hexdigest()
        stamp['trace_sha256'] = hashlib.sha256((fixture / 'trace.bin').read_bytes()).hexdigest()
        if backend.startswith('torch-'):
            stamp['torch_sources_sha256'] = self.torch_hash
            stamp['gpu_python_packages'] = self.gpu_packages
        stamp_path = Path(str(prefix) + '.stamp.json')
        result_path = Path(str(prefix) + '.json')
        cached = cached_result(prefix, stamp)
        if cached is not None:
            return cached
        # Other clients must not be interrupted. Refuse contaminated final comparisons.
        clients = subprocess.run(['nvidia-smi', '--query-compute-apps=pid,process_name', '--format=csv,noheader'],
                                 text=True, capture_output=True, check=True).stdout.strip()
        if clients:
            raise RuntimeError(f'GPU has another compute client; retry later: {clients}')
        if archive_attempt(prefix, self.records):
            (self.root / 'runs.json').write_text(json.dumps(self.records, indent=2))
        before = subprocess.check_output(['nvidia-smi', '-q'], text=True)
        Path(str(prefix) + '.environment.txt').write_text(before)
        rc = run(command, prefix)
        result = read(result_path) if rc == 0 else {}
        completed = result.get('completed', result.get('events', 0))
        sample_failure = completed < (self.a.events if measurement == 'scheduled' else count)
        entry = dict(label=label, size=size, precision=mode, backend=backend, repeat=repeat, comparison_group=group or backend,
                     command=command, returncode=rc, sample_gate_passed=not sample_failure, final_sample_count=completed - discard >= 100000,
                     executable_sha256=self.binary_hash, measurement=measurement,
                     transport_warmup_discard=discard)
        self.records.append(entry)
        (self.root / 'runs.json').write_text(json.dumps(self.records, indent=2))
        print(label, rc, flush=True)
        if rc or sample_failure:
            raise RuntimeError(f'Failed condition: {prefix}.log')
        stamp['artifacts'] = {suffix: file_hash(Path(str(prefix) + suffix)) for suffix in ['.json', '.csv']}
        stamp_path.write_text(json.dumps(stamp, indent=2))
        return read(result_path)

    def options(self, size, mode, backend):
        choice = self.selection[f'{size}-{mode}'][backend]
        selected = choice.get('backend', backend)
        extra = []
        for key in ['blocks', 'transport', 'threads', 'backoff']:
            if key in choice:
                extra += ['--' + key, str(choice[key])]
        return selected, extra

    def tune(self, sizes=None, modes=None):
        probe = read('results/blackwell-hardware.json')
        grids = sorted(set([1, 4, 8, 16, probe['sm_count']]))
        for size, mode in itertools.product(sizes or self.a.sizes, modes or self.a.modes):
            candidates = []
            for backend, block, transport in itertools.product(['graph', 'cublas-graph'], grids, ['copy', 'mapped']):
                candidates.append(dict(group='graph', backend=backend, blocks=block, transport=transport))
            for threads in [1, 2, 4]:
                candidates.append(dict(group='cpu', backend='cpu', threads=threads))
            # Legal occupancy is queried from the actual compiled kernel, not the SM count.
            trial = self.execute(f'tune-{size}-{mode}-persistent-legal', size, mode, 'persistent', 0,
                                 ['--blocks', '1'], 100)
            legal = trial['legal_cooperative_blocks']
            for block in grids:
                if block <= legal:
                    candidates.append(dict(group='persistent', backend='persistent', blocks=block))
            config = read(f'configs/{size}.json')
            for candidate in candidates:
                extra = [v for key, value in candidate.items() if key not in ['group', 'backend']
                         for v in ['--' + key, str(value)]]
                label = f'tune-correctness-{size}-{mode}-' + '-'.join(str(v) for v in candidate.values())
                self.execute(label, size, mode, candidate['backend'], 0,
                             extra + ['--mode', 'correctness'], 3 * config['window'] + 9)
            measurements = {group: [] for group in ['cpu', 'graph', 'persistent']}
            for repeat in range(3):
                order = candidates.copy()
                random.Random(42 + repeat).shuffle(order)
                for candidate in order:
                    extra = [v for key, value in candidate.items() if key not in ['group', 'backend']
                             for v in ['--' + key, str(value)]]
                    label = f'tune-{size}-{mode}-{repeat}-' + '-'.join(str(v) for v in candidate.values())
                    result = self.execute(label, size, mode, candidate['backend'], repeat, extra, self.a.tune_events)
                    measurements[candidate['group']].append((candidate, result['service']['p99_us']))
            selected = {}
            for group, values in measurements.items():
                scores = {}
                for candidate, p99 in values:
                    key = json.dumps(candidate, sort_keys=True)
                    scores.setdefault(key, []).append(p99)
                best = min(scores, key=lambda key: sorted(scores[key])[len(scores[key]) // 2])
                selected[group] = json.loads(best)
            self.selection[f'{size}-{mode}'] = selected
            (self.root / 'selection.json').write_text(json.dumps(self.selection, indent=2))

    def noise(self):
        # Keep default clocks throughout. Affinity is the only changed control;
        # denied NUMA/governor controls must not be described as applied.
        for repeat in range(self.a.runs):
            for controlled in [False, True]:
                regime = 'affinity' if controlled else 'uncontrolled'
                transports = [(mode, backoff) for mode in ['ping-pong', 'gpu-ping'] for backoff in [0, 1]]
                transports += [(mode, 0) for mode in ['empty', 'empty-graph', 'copy']]
                random.Random(500 + repeat).shuffle(transports)
                for mode, backoff in transports:
                    # Transport output retains all samples. Its analyzer discards
                    # this explicit prefix, rather than claiming CLI model warmup
                    # affects the transport kernel (it does not).
                    self.execute(f'noise-transport-{regime}-{mode}-{backoff}-{repeat}',
                                 'main', 'fp32', 'graph', repeat,
                                 ['--mode', mode, '--backoff', str(backoff)],
                                 self.a.events + self.a.transport_warmup,
                                 controlled=controlled, group=f'transport-{mode}-{backoff}')
                for backend in ['cublas-graph', 'persistent']:
                    self.execute(f'noise-main-fp32-{regime}-{backend}-{repeat}',
                                 'main', 'fp32', backend, repeat, ['--blocks', '4'],
                                 controlled=controlled, group='graph' if backend == 'cublas-graph' else backend)

    def service(self):
        for repeat in range(self.a.runs):
            conditions = list(itertools.product(self.a.sizes, self.a.modes,
                               ['cpu', 'graph', 'persistent', 'cuda', 'cublas', 'fused']))
            if self.a.torch:
                conditions += list(itertools.product(self.a.sizes, self.a.modes,
                                   ['torch-eager', 'torch-compile', 'torch-graph']))
            random.Random(200 + repeat).shuffle(conditions)
            for size, mode, group in conditions:
                backend, extra = self.options(size, mode, group) if group in ['cpu', 'graph', 'persistent'] else (group, [])
                if group == 'fused':
                    _, extra = self.options(size, mode, 'persistent')
                if group in ['cuda', 'cublas']:
                    _, extra = self.options(size, mode, 'graph')
                self.execute(f'service-{size}-{mode}-{group}-{repeat}', size, mode, backend, repeat, extra, group=group)

    def scheduled(self):
        for size, mode in itertools.product(self.a.sizes, self.a.modes):
            anchor_path = self.root / f'service-{size}-{mode}-graph-0.json'
            capacity = read(anchor_path)['service_rate_per_second']
            slowest = min(read(self.root / f'service-{size}-{mode}-{group}-{repeat}.json')['service_rate_per_second']
                          for group, repeat in itertools.product(['cpu', 'graph', 'persistent'], range(self.a.runs)))
            anchor = dict(backend=self.selection[f'{size}-{mode}']['graph'], source=str(anchor_path), capacity=capacity)
            (self.root / f'anchor-{size}-{mode}.json').write_text(json.dumps(anchor, indent=2))
            for repeat in range(self.a.runs):
                order = list(itertools.product([.25, .5, .75, .9, 1.1], [0, 1], ['cpu', 'graph', 'persistent']))
                random.Random(300 + repeat).shuffle(order)
                for fraction, burst, group in order:
                    backend, extra = self.options(size, mode, group)
                    extra += ['--mode', 'scheduled', '--rate', str(capacity * fraction), '--bursty', str(burst)]
                    # Replace the single mode value; duplicate options are intentionally rejected by the native CLI.
                    label = f'scheduled-{size}-{mode}-{repeat}-{fraction}-{burst}-{group}'
                    # Offer the same larger schedule to every backend at this rate.
                    # Overload still drops explicitly, but completed sample counts
                    # must meet the final gate. The slowest measured path anchors
                    # a conservative offered count instead of hiding short runs.
                    offered = math.ceil((self.a.events + 8192) * max(1., capacity * fraction / (.75 * slowest)))
                    self.execute(label, size, mode, backend, repeat, extra, offered, group=group)

    def windows(self):
        from reference.export import export
        for size, mode, window in itertools.product(self.a.sizes, self.a.modes, [32, 64, 128, 256, 512]):
            variant = f'{size}-w{window}'
            config = read(f'configs/{size}.json')
            config['window'] = window
            config_path = Path(f'configs/{variant}.json')
            config_path.write_text(json.dumps(config, indent=2) + '\n')
            fixture = Path(f'fixtures/{variant}-{mode}')
            if not (fixture / 'manifest.json').exists():
                export(config_path, fixture, mode, count=3 * window + 9)
            self.tune([variant], [mode])
            for repeat in range(self.a.runs):
                order = ['cpu', 'graph', 'persistent']
                random.Random(repeat).shuffle(order)
                for group in order:
                    backend, extra = self.options(variant, mode, group)
                    self.execute(f'window-{variant}-{mode}-{repeat}-{group}', variant, mode, backend, repeat, extra, group=group)

    def saturation(self):
        # These curves use each backend's own measured capacity. They are distinct
        # from scheduled(), whose absolute offered rates are identical across paths.
        for size, mode in itertools.product(self.a.sizes, self.a.modes):
            capacities = {}
            for group in ['cpu', 'graph', 'persistent']:
                rates = sorted(read(self.root / f'service-{size}-{mode}-{group}-{repeat}.json')
                               ['service_rate_per_second'] for repeat in range(self.a.runs))
                capacities[group] = rates[len(rates) // 2]
            (self.root / f'saturation-anchors-{size}-{mode}.json').write_text(
                json.dumps(dict(kind='backend-specific', capacities=capacities), indent=2))
            for repeat in range(self.a.runs):
                conditions = list(itertools.product(['cpu', 'graph', 'persistent'], [.25, .5, .75, .9, 1.1, 1.5]))
                random.Random(400 + repeat).shuffle(conditions)
                for group, fraction in conditions:
                    backend, extra = self.options(size, mode, group)
                    extra += ['--mode', 'scheduled', '--rate', str(capacities[group] * fraction), '--bursty', '0']
                    offered = math.ceil((self.a.events + 8192) * max(1., fraction / .75))
                    self.execute(f'saturation-{size}-{mode}-{repeat}-{fraction}-{group}',
                                 size, mode, backend, repeat, extra, offered, group=group)

    def idle(self):
        # Gap sweeps are explicitly diagnostics; final conditions are in service/scheduled.
        for mode, repeat, gap in itertools.product(self.a.modes, range(self.a.runs), [0, 100, 1000, 10000]):
            for group in ['graph', 'persistent']:
                backend, extra = self.options('main', mode, group)
                self.execute(f'idle-main-{mode}-{repeat}-{gap}-{group}', 'main', mode, backend, repeat,
                             extra + ['--idle-us', str(gap)], self.a.idle_events, group=group)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--binary', default='build-blackwell/stream-bench')
    p.add_argument('--gpu-python', default='.venv-gpu/bin/python')
    p.add_argument('--output', type=Path, default=Path('results/final-study'))
    p.add_argument('--sizes', nargs='+', default=['small', 'main', 'large'])
    p.add_argument('--modes', nargs='+', default=['fp32', 'mixed'])
    p.add_argument('--phases', nargs='+', choices=['noise', 'tune', 'service', 'scheduled', 'saturation', 'windows', 'idle'],
                   default=['noise', 'tune', 'service', 'scheduled', 'saturation', 'windows', 'idle'])
    p.add_argument('--events', type=int, default=100000)
    p.add_argument('--runs', type=int, default=5)
    p.add_argument('--tune-events', type=int, default=2000)
    p.add_argument('--idle-events', type=int, default=2000)
    p.add_argument('--transport-warmup', type=int, default=1000)
    p.add_argument('--worker-core', type=int, default=96)
    p.add_argument('--producer-core', type=int, default=97)
    p.add_argument('--torch', action='store_true')
    p.add_argument('--gate', type=Path, required=True)
    p.add_argument('--followup-gate', type=Path, default=Path('results/validation-followup/summary.json'))
    a = p.parse_args()
    study = Study(a)
    check_native_gate(a.gate, study.binary_hash)
    check_followup_gate(a.followup_gate, study.binary_hash)
    for phase in a.phases:
        getattr(study, phase)()


if __name__ == '__main__':
    main()

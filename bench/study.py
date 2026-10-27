"""Reproducible comparison matrix. Final conditions use 100k events x five runs.

Runs serially; never kills other GPU clients. Telemetry uses an independent
process with the same sampling interval on CPU and GPU conditions.
"""
import argparse
import itertools
import json
import random
import subprocess
import time
import datetime
from pathlib import Path


def locality(pid):
    process = Path(f'/proc/{pid}')
    try:
        threads = []
        for task in (process / 'task').iterdir():
            status = (task / 'status').read_text().splitlines()
            fields = {line.split(':', 1)[0]: line.split(':', 1)[1].strip() for line in status if ':' in line}
            threads.append(dict(tid=int(task.name), name=fields.get('Name'),
                                cpus_allowed_list=fields.get('Cpus_allowed_list'),
                                mems_allowed_list=fields.get('Mems_allowed_list')))
        return dict(status='observed', pid=pid, threads=threads,
                    numa_maps=(process / 'numa_maps').read_text(),
                    monotonic_ns=time.monotonic_ns(),
                    interpretation='Observed mappings/affinity; allocation policies and helper-thread roles are not inferred')
    except OSError as error:
        return dict(status='unavailable', pid=pid, reason=str(error))


def run(command, prefix, telemetry=True):
    prefix = Path(prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    Path(str(prefix) + '.command.json').write_text(json.dumps(command, indent=2))
    monitor = None
    telemetry_file = None
    if telemetry:
        telemetry_file = Path(str(prefix) + '.telemetry.csv').open('w')
        monitor = subprocess.Popen(['nvidia-smi', '--query-gpu=timestamp,power.draw,clocks.current.graphics,clocks.current.memory,temperature.gpu,utilization.gpu',
                                    '--format=csv', '-lms', '200'], stdout=telemetry_file, stderr=subprocess.STDOUT)
    try:
        start_wall, start_monotonic = time.time_ns(), time.monotonic_ns()
        with Path(str(prefix) + '.log').open('w') as output:
            result = subprocess.Popen(command, stdout=output, stderr=subprocess.STDOUT)
            try:
                result.wait(timeout=.2)
                observed = dict(status='unavailable', reason='Process completed before locality sample', pid=result.pid)
            except subprocess.TimeoutExpired:
                observed = locality(result.pid)
                result.wait()
            Path(str(prefix) + '.locality.json').write_text(json.dumps(observed, indent=2))
        Path(str(prefix) + '.clock.json').write_text(json.dumps(dict(
            start_wall_ns=start_wall, start_monotonic_ns=start_monotonic,
            end_wall_ns=time.time_ns(), end_monotonic_ns=time.monotonic_ns(),
            utc_offset_seconds=datetime.datetime.now().astimezone().utcoffset().total_seconds(),
            telemetry_interval_ms=200), indent=2))
        return result.returncode
    finally:
        if monitor:
            monitor.terminate()
            monitor.wait()
            telemetry_file.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--binary', default='build/stream-bench')
    p.add_argument('--output', default='results/study')
    p.add_argument('--events', type=int, default=100000)
    p.add_argument('--runs', type=int, default=5)
    p.add_argument('--sizes', nargs='+', default=['small', 'main', 'large'])
    p.add_argument('--modes', nargs='+', default=['fp32', 'mixed'])
    p.add_argument('--worker-core', type=int, default=-1)
    p.add_argument('--producer-core', type=int, default=-1)
    p.add_argument('--phases', nargs='+', default=['service', 'scheduled', 'grid', 'idle', 'threads', 'windows'])
    a = p.parse_args()
    root = Path(a.output)
    root.mkdir(parents=True, exist_ok=True)
    results = []

    def execute(size, precision, backend, repeat, phase='service', extra=None, suffix=''):
        name = f'{phase}-{size}-{precision}-{backend}-{repeat}{suffix}'
        prefix = root / name
        command = [a.binary, '--config', f'configs/{size}.json', '--weights', f'fixtures/{size}-{precision}',
                   '--backend', backend, '--events', str(a.events), '--warmup', '1000',
                   '--mode', 'scheduled' if phase == 'scheduled' else 'service', '--output', str(prefix),
                   '--worker-core', str(a.worker_core), '--producer-core', str(a.producer_core)]
        if extra:
            command.extend(extra)
        rc = run(command, prefix)
        results.append(dict(name=name, returncode=rc, command=command))
        (root / 'runs.json').write_text(json.dumps(results, indent=2))
        print(name, rc, flush=True)
        if rc:
            raise RuntimeError(f'{name} failed; see {prefix}.log')
        return json.loads(Path(str(prefix) + '.json').read_text())

    for size, precision in itertools.product(a.sizes, a.modes):
        fixture = Path(f'fixtures/{size}-{precision}')
        if not (fixture / 'manifest.json').exists():
            subprocess.run(['.venv/bin/python', '-m', 'reference.export', '--config', f'configs/{size}.json',
                            '--mode', precision, '--output', str(fixture), '--events', '64'], check=True)
    if 'service' in a.phases:
        for repeat in range(a.runs):
            conditions = list(itertools.product(a.sizes, a.modes,
                              ['cpu', 'cuda', 'cublas', 'graph', 'cublas-graph', 'fused', 'persistent']))
            random.Random(42 + repeat).shuffle(conditions)
            for size, precision, backend in conditions:
                execute(size, precision, backend, repeat)
    if 'scheduled' in a.phases:
        for size, precision in itertools.product(a.sizes, a.modes):
            # A common absolute arrival schedule anchored to the measured custom graph.
            reference = json.loads((root / f'service-{size}-{precision}-graph-0.json').read_text())
            capacity = reference['service_rate_per_second']
            (root / f'arrival-anchor-{size}-{precision}.json').write_text(json.dumps(dict(backend='graph',
                reference=f'service-{size}-{precision}-graph-0.json', capacity=capacity), indent=2))
            for repeat in range(a.runs):
                conditions = list(itertools.product([.25, .5, .75, .9, 1.1], [0, 1],
                                                    ['cpu', 'graph', 'cublas-graph', 'persistent']))
                random.Random(100 + repeat).shuffle(conditions)
                for fraction, bursty, backend in conditions:
                    execute(size, precision, backend, repeat, 'scheduled',
                            ['--rate', str(capacity * fraction), '--bursty', str(bursty)],
                            f'-load{fraction}-burst{bursty}')
    if 'grid' in a.phases:
        probe = json.loads(subprocess.check_output(['build/stream-probe']))
        blocks = sorted(set([1, 4, 8, 16, probe['sm_count']]))
        for repeat, size, precision in itertools.product(range(a.runs), a.sizes, a.modes):
            for backend, block in itertools.product(['graph', 'fused', 'persistent'], blocks):
                execute(size, precision, backend, repeat, 'grid', ['--blocks', str(block)], f'-blocks{block}')
    if 'threads' in a.phases:
        for size, repeat, threads in itertools.product(a.sizes, range(a.runs), [1, 2, 4]):
            execute(size, 'fp32', 'cpu', repeat, 'threads', ['--threads', str(threads)], f'-threads{threads}')
    if 'idle' in a.phases:
        for repeat, precision, backend, gap in itertools.product(range(a.runs), a.modes, ['graph', 'persistent'], [0, 100, 1000, 10000]):
            execute('main', precision, backend, repeat, 'idle', ['--idle-us', str(gap)], f'-gap{gap}')
    if 'windows' in a.phases:
        # Shape changes use separately exported, hash-identified checkpoints with the same seed.
        for size, window, precision in itertools.product(a.sizes, [32, 64, 128, 256, 512], a.modes):
            variant = f'{size}-w{window}'
            config = json.loads(Path(f'configs/{size}.json').read_text())
            config['window'] = window
            Path(f'configs/{variant}.json').write_text(json.dumps(config, indent=2))
            subprocess.run(['.venv/bin/python', '-m', 'reference.export', '--config', f'configs/{variant}.json',
                            '--mode', precision, '--output', f'fixtures/{variant}-{precision}', '--events', '32'], check=True)
            for repeat, backend in itertools.product(range(a.runs), ['cpu', 'graph', 'persistent']):
                execute(variant, precision, backend, repeat, 'windows')


if __name__ == '__main__':
    main()

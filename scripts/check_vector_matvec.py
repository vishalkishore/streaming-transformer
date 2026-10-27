"""Validate the pair-load GEMV experiment before short serial GPU comparisons."""
import argparse
import itertools
import json
import random
from pathlib import Path

from bench.study import run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('results/vector-matvec'))
    parser.add_argument('--benchmark', action='store_true')
    parser.add_argument('--runs', type=int, default=3)
    parser.add_argument('--worker-core', type=int, default=96)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    records = []

    def execute(size, precision, backend, implementation, mode, events, repeat=0):
        prefix = args.output / f'{mode}-{size}-{precision}-{backend}-{implementation}-{events}-r{repeat}'
        command = ['build-blackwell/stream-bench', '--config', f'configs/{size}.json',
                   '--weights', f'fixtures/{size}-{precision}', '--backend', backend,
                   '--blocks', '110', '--matvec', implementation, '--mode', mode,
                   '--events', str(events), '--output', str(prefix),
                   '--worker-core', str(args.worker_core), '--transport',
                   'copy' if backend == 'graph' else 'mapped']
        if mode == 'service':
            command += ['--warmup', '256', '--clock-regime', 'default-dynamic-affinity-short-diagnostic']
        rc = run(command, prefix)
        if rc:
            raise RuntimeError(f'Failed experiment: {prefix}')
        result = json.loads(Path(str(prefix) + '.json').read_text())
        if result['cpu_oracle_active']:
            raise RuntimeError('Stored fixtures required; no live CPU oracle')
        records.append(dict(size=size, precision=precision, backend=backend, implementation=implementation,
                            mode=mode, events=events, repeat=repeat, command=command, result=result))
        (args.output / 'summary.json').write_text(json.dumps(dict(
            scope='Bounded correctness and short diagnostics; not final performance evidence', runs=records), indent=2))
        print(prefix.name, 'PASS', f'p99={result["service"]["p99_us"]:.3f} us', flush=True)

    for size, precision, backend in itertools.product(['small', 'main', 'large'], ['fp32', 'mixed'],
                                                      ['graph', 'persistent']):
        config = json.loads(Path(f'configs/{size}.json').read_text())
        execute(size, precision, backend, 'vector', 'correctness', 3 * config['window'] + 9)
    if args.benchmark:
        for precision in ['fp32', 'mixed']:
            for events, repeat in [(1000, 0)] + [(10000, i + 1) for i in range(args.runs)]:
                cases = list(itertools.product(['graph', 'persistent'], ['scalar', 'vector']))
                random.Random(731 + repeat).shuffle(cases)
                for backend, implementation in cases:
                    execute('main', precision, backend, implementation, 'service', events, repeat)


if __name__ == '__main__':
    main()

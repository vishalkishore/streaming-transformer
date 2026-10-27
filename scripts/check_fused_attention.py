"""Check fused short-window attention, then run bounded exclusive GPU A/B tests."""
import argparse
import itertools
import json
import random
from pathlib import Path

from bench.study import run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('results/fused-attention'))
    parser.add_argument('--benchmark', action='store_true')
    parser.add_argument('--runs', type=int, default=3)
    parser.add_argument('--worker-core', type=int, default=96)
    parser.add_argument('--reference-binary', type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    records = []

    def execute(size, precision, backend, implementation, mode, events, repeat=0, blocks=110):
        prefix = args.output / f'{mode}-{size}-{precision}-{backend}-{implementation}-b{blocks}-{events}-r{repeat}'
        binary = args.reference_binary if implementation == 'previous-staged' else Path('build-blackwell/stream-bench')
        command = [str(binary), '--config', f'configs/{size}.json', '--weights', f'fixtures/{size}-{precision}',
                   '--backend', backend, '--blocks', str(blocks), '--matvec', 'scalar', '--mode', mode,
                   '--events', str(events), '--output', str(prefix), '--worker-core', str(args.worker_core),
                   '--transport', 'copy' if backend == 'graph' else 'mapped']
        if implementation != 'previous-staged':
            command += ['--attention', implementation]
        if mode == 'service':
            command += ['--warmup', '256', '--clock-regime', 'default-dynamic-affinity-short-diagnostic']
        if run(command, prefix):
            raise RuntimeError(f'Failed fused-attention experiment: {prefix}')
        result = json.loads(Path(str(prefix) + '.json').read_text())
        if result['cpu_oracle_active']:
            raise RuntimeError('A live CPU oracle is forbidden in this experiment')
        records.append(dict(size=size, precision=precision, backend=backend, implementation=implementation,
                            mode=mode, events=events, repeat=repeat, blocks=blocks, command=command, result=result))
        (args.output / 'summary.json').write_text(json.dumps(dict(
            scope='Bounded correctness and short diagnostics, not final performance evidence', runs=records), indent=2))
        print(prefix.name, 'PASS', f'p99={result["service"]["p99_us"]:.3f} us', flush=True)

    for size, precision, backend in itertools.product(['small', 'main', 'large'], ['fp32', 'mixed'],
                                                      ['graph', 'persistent']):
        window = json.loads(Path(f'configs/{size}.json').read_text())['window']
        execute(size, precision, backend, 'fused', 'correctness', 3 * window + 9)
    for precision in ['fp32', 'mixed']:
        for backend, blocks in [('graph', 4), ('persistent', 1)]:
            execute('main', precision, backend, 'fused', 'correctness', 777, blocks=blocks)
    if args.benchmark:
        implementations = ['staged', 'fused'] + (['previous-staged'] if args.reference_binary else [])
        for precision in ['fp32', 'mixed']:
            for events, repeat in [(1000, 0)] + [(10000, i + 1) for i in range(args.runs)]:
                cases = list(itertools.product(['graph', 'persistent'], implementations))
                random.Random(739 + repeat).shuffle(cases)
                for backend, implementation in cases:
                    execute('main', precision, backend, implementation, 'service', events, repeat)


if __name__ == '__main__':
    main()

"""Short serial Main diagnostic; CPU once, then GPU grid/graph comparisons."""
import argparse
import json
import random
from pathlib import Path

from bench.study import run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('results/quick-main'))
    parser.add_argument('--worker-core', type=int, default=96)
    parser.add_argument('--skip-cpu', action='store_true', help='Reuse the earlier CPU baseline for an optimization comparison')
    parser.add_argument('--full-graphs', action='store_true', help='Sweep the graph grids as well as persistent grids')
    parser.add_argument('--runs', type=int, default=1, help='Separate varied-order 10k diagnostic repetitions')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    records = []

    def execute(backend, blocks, events, repeat=0, transport='copy'):
        suffix = '-mapped' if transport == 'mapped' and backend in ['graph', 'cublas-graph'] else ''
        label = f'{backend}-{blocks}-{events}' + suffix + (f'-r{repeat}' if repeat else '')
        prefix = args.output / label
        command = ['build-blackwell/stream-bench', '--config', 'configs/main.json', '--weights', 'fixtures/main-fp32',
                   '--backend', backend, '--blocks', str(blocks), '--events', str(events), '--warmup', '256',
                   '--worker-core', str(args.worker_core), '--mode', 'service', '--output', str(prefix),
                   '--transport', transport,
                   '--clock-regime', 'default-dynamic-affinity-short-diagnostic']
        rc = run(command, prefix)
        if rc:
            raise RuntimeError(f'Failed diagnostic {prefix}.log')
        result = json.loads(Path(str(prefix) + '.json').read_text())
        if result['cpu_oracle_active']:
            raise RuntimeError('CPU oracle must not execute in timing runs')
        records.append(dict(label=label, command=command, result=result, repeat=repeat))
        (args.output / 'summary.json').write_text(json.dumps(dict(
            scope='Short diagnostic only; not final repeated performance evidence', runs=records), indent=2))
        print(f'{label}: p50={result["service"]["p50_us"]:.2f} us, '
              f'p99={result["service"]["p99_us"]:.2f} us, rate={result["service_rate_per_second"]:.1f}/s', flush=True)
        return result

    if not args.skip_cpu:
        execute('cpu', 1, 1000)
    graphs = []
    for backend in ['graph', 'cublas-graph']:
        for blocks in [1, 4, 8, 16, 110] if args.full_graphs else [4]:
            for transport in ['copy', 'mapped'] if args.full_graphs else ['copy']:
                graphs.append((backend, blocks, transport, execute(backend, blocks, 1000, transport=transport)))
    persistent = []
    for blocks in [1, 4, 8, 16, 110]:
        persistent.append(('persistent', blocks, 'mapped', execute('persistent', blocks, 1000, transport='mapped')))
    best_graph = min(graphs, key=lambda row: row[3]['service']['p99_us'])
    best_persistent = min(persistent, key=lambda row: row[3]['service']['p99_us'])
    for repeat in range(args.runs):
        order = [best_graph, best_persistent]
        random.Random(900 + repeat).shuffle(order)
        for backend, blocks, transport, _ in order:
            execute(backend, blocks, 10000, repeat, transport)


if __name__ == '__main__':
    main()

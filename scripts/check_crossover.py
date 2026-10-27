"""Bounded graph/persistence survey; CPU inference is never part of timing.

Device traces begin with an empty cache; service traces reset after warm-up.
Their averages/quantiles must not be subtracted to estimate transport overhead.
"""
import argparse
import itertools
import json
import random
import statistics
from pathlib import Path

from bench.study import run


def summarize(records):
    comparisons = []
    groups = sorted({(r['size'], r['precision'], r['mode']) for r in records})
    for size, precision, mode in groups:
        selected = [r for r in records if (r['size'], r['precision'], r['mode']) == (size, precision, mode)]
        metric = 'device_us_per_event' if mode == 'device' else 'service_p99_us'
        values = {backend: [r[metric] for r in selected if r['backend'] == backend]
                  for backend in ['graph', 'persistent']}
        if not all(values.values()):
            continue
        graph, persistent = (statistics.median(values[b]) for b in ['graph', 'persistent'])
        comparisons.append(dict(size=size, precision=precision, mode=mode, metric=metric,
                                graph_us=graph, persistent_us=persistent,
                                persistent_over_graph=persistent / graph,
                                runs_by_backend={b: len(v) for b, v in values.items()}))
    return comparisons


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('results/crossover-survey'))
    parser.add_argument('--sizes', nargs='+', default=['small', 'main', 'large'])
    parser.add_argument('--events', type=int, default=1000)
    parser.add_argument('--runs', type=int, default=1)
    parser.add_argument('--start-repeat', type=int, default=0)
    parser.add_argument('--blocks', type=int, default=110)
    parser.add_argument('--worker-core', type=int, default=96)
    args = parser.parse_args()
    if args.events <= 0 or args.runs <= 0:
        parser.error('events and runs must be positive')
    args.output.mkdir(parents=True, exist_ok=True)
    records = []
    for repeat in range(args.start_repeat, args.start_repeat + args.runs):
        cases = list(itertools.product(args.sizes, ['fp32', 'mixed'], ['device', 'service'],
                                       ['graph', 'persistent']))
        random.Random(2700 + repeat).shuffle(cases)
        for size, precision, mode, backend in cases:
            prefix = args.output / f'{size}-{precision}-{mode}-{backend}-b{args.blocks}-r{repeat}'
            command = ['build-blackwell/stream-bench', '--config', f'configs/{size}.json',
                       '--weights', f'fixtures/{size}-{precision}', '--backend', backend,
                       '--blocks', str(args.blocks), '--mode', mode, '--events', str(args.events),
                       '--warmup', '256', '--matvec', 'scalar', '--attention', 'staged',
                       '--worker-core', str(args.worker_core), '--transport',
                       'copy' if backend == 'graph' else 'mapped', '--output', str(prefix),
                       '--clock-regime', 'default-dynamic-affinity-short-diagnostic']
            if run(command, prefix):
                raise RuntimeError(f'Failed {prefix}; inspect retained log')
            result = json.loads(Path(str(prefix) + '.json').read_text())
            if result['cpu_oracle_active'] or result.get('dropped', 0):
                raise RuntimeError('CPU oracle active or dropped events')
            record = dict(size=size, precision=precision, mode=mode, backend=backend,
                          repeat=repeat, events=args.events, prefix=str(prefix),
                          device_us_per_event=result.get('device_us_per_event'),
                          service_p99_us=result.get('service', {}).get('p99_us'))
            records.append(record)
            report = dict(scope='Short diagnostic; no final crossover claim',
                          boundaries='Device: empty-cache finite preloaded trace; service: reset after warmup, host completion. Do not subtract.',
                          blocks=args.blocks, runs=records, comparisons=summarize(records))
            (args.output / 'summary.json').write_text(json.dumps(report, indent=2) + '\n')
            print(prefix.name, record['device_us_per_event'] if mode == 'device' else record['service_p99_us'], flush=True)


if __name__ == '__main__':
    main()

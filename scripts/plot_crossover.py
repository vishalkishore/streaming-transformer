"""Audit short crossover records and plot each timing boundary separately."""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('summaries', nargs='+', type=Path)
    parser.add_argument('--output', type=Path, default=Path('results/crossover-survey-figure'))
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    data, hashes, count = [], set(), 0
    for path in args.summaries:
        summary = json.loads(path.read_text())
        for record in summary['runs']:
            prefix = record['prefix']
            guard = json.loads(Path(prefix + '.gpu-processes.json').read_text())
            if not guard['valid'] or not guard['empty_before_launch'] or guard['violations']:
                raise RuntimeError(f'Invalid GPU exclusivity: {prefix}')
            result = json.loads(Path(prefix + '.json').read_text())
            if result['cpu_oracle_active'] or result.get('dropped', 0):
                raise RuntimeError(f'Invalid timing: {prefix}')
            hashes.add(result['provenance']['executable_sha256'])
            count += 1
        data.extend(dict(row, blocks=summary['blocks']) for row in summary['comparisons'])
    if len(hashes) != 1:
        raise RuntimeError('Mixed benchmark executable identities')
    figure, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    labels = sorted({(row['size'], row['precision']) for row in data})
    for ax, mode in zip(axes, ['device', 'service']):
        for grid in sorted({row['blocks'] for row in data}):
            selected = {(row['size'], row['precision']): row for row in data
                        if row['mode'] == mode and row['blocks'] == grid}
            xs = [i for i, label in enumerate(labels) if label in selected]
            ax.scatter(xs, [selected[labels[i]]['persistent_over_graph'] for i in xs],
                       label=f'{grid} blocks', s=40)
        ax.axhline(1, color='black', linewidth=1, linestyle='--')
        ax.set_xticks(range(len(labels)), [f'{size}\n{precision}' for size, precision in labels], rotation=45)
        ax.set_ylabel('Persistence / graph latency (lower is better)')
        ax.set_title('Preloaded device mean' if mode == 'device' else 'Host-service p99')
        ax.legend()
        ax.grid(axis='y', alpha=.25)
    figure.suptitle('Short diagnostics; different timing boundaries; no final crossover claim')
    for suffix in ['.png', '.svg']:
        figure.savefig(str(args.output) + suffix, dpi=160)
    audit = dict(runs=count, valid_gpu_evidence=count, executable_sha256=next(iter(hashes)),
                 summaries=[dict(path=str(p), sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in args.summaries],
                 comparisons=data, sampling_limit='Process observations sampled once per second, not continuous proof')
    Path(str(args.output) + '.json').write_text(json.dumps(audit, indent=2) + '\n')
    print(json.dumps(dict(runs=count, executable_sha256=next(iter(hashes)))))


if __name__ == '__main__':
    main()

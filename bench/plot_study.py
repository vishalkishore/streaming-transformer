"""Export crossover, window and load figures from a verified study summary."""
import argparse
import collections
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def parameter_count(config):
    width, ffn, layers = config['width'], config['ffn'], config['layers']
    return config['features'] * width + layers * (4 * width * width + 3 * width * ffn + 2 * width) + 2 * width


def draw_curves(series, path, title, xlabel, ylabel, log_x=False, fraction=False):
    figure, axes = plt.subplots(figsize=(7, 4.5))
    diagnostic = False
    for label, points in sorted(series.items()):
        points.sort(key=lambda point: point[0])
        x = [point[0] for point in points]
        y = np.array([point[1] for point in points])
        low, high = np.array([point[2] for point in points]), np.array([point[3] for point in points])
        diagnostic |= not all(point[4] for point in points)
        axes.errorbar(x, y, yerr=np.stack([y - low, high - y]), marker='o', capsize=3, label=label)
    axes.set(title=title + (' — diagnostic/incomplete evidence' if diagnostic else ''), xlabel=xlabel, ylabel=ylabel)
    if log_x:
        if log_x == 'symlog':
            axes.set_xscale('symlog', linthresh=1)
        else:
            axes.set_xscale('log')
    if fraction:
        axes.set_ylim(0, 1)
    axes.grid(alpha=.2)
    axes.legend(fontsize='small')
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def plot_report(report, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    figures = collections.defaultdict(lambda: collections.defaultdict(list))
    descriptions = {}
    for condition in report['conditions']:
        work = condition['workload']
        phase, precision = work['phase'], work['precision']
        for group, backend in condition['backends'].items():
            summary = backend['summary']
            point = [None, summary['median_run_p99_us'], summary['min_run_p99_us'],
                     summary['max_run_p99_us'], summary['final_evidence_eligible']]
            if phase == 'service':
                name = f'crossover-{precision}'
                point[0] = parameter_count(work['config'])
                descriptions[name] = (f'{precision}: model-size crossover', 'Model parameters', 'Median run service p99 (µs)', True, False)
            elif phase == 'window':
                size = work['size'].split('-w', 1)[0]
                name = f'window-{size}-{precision}'
                point[0] = work['config']['window']
                descriptions[name] = (f'{size} {precision}: attention window', 'Window positions', 'Median run service p99 (µs)', True, False)
            elif phase in ['scheduled', 'saturation']:
                name = f'{phase}-{work["size"]}-{precision}-burst{work["bursty"]}'
                point[0] = float(work['rate'])
                descriptions[name] = (f'{work["size"]} {precision}: {phase}, burst={work["bursty"]}',
                                      'Offered events/second', 'Median run arrival p99 (µs)', False, False)
                fractions = [run['drop_fraction'] for run in backend['runs']]
                drop_name = name + '-drops'
                descriptions[drop_name] = (f'{work["size"]} {precision}: explicit drops',
                                           'Offered events/second', 'Dropped fraction', False, True)
                figures[drop_name][group].append([point[0], float(np.median(fractions)), min(fractions),
                                                 max(fractions), point[4]])
            elif phase == 'idle':
                name = f'idle-{work["size"]}-{precision}'
                point[0] = float(work['idle_us'])
                descriptions[name] = (f'{work["size"]} {precision}: first event after idle',
                                      'Idle gap (µs)', 'Median run service p99 (µs)', 'symlog', False)
            else:
                continue
            figures[name][group].append(point)
            powers = [run.get('telemetry', {}).get('power_w', {}).get('sample_mean') for run in backend['runs']]
            valid = [value for value in powers if value is not None]
            if valid:
                power_name = name + '-power'
                title, xlabel, _, log_x, _ = descriptions[name]
                descriptions[power_name] = (title + ': measured-interval power', xlabel,
                                            'GPU power (W), per-run sampled mean', log_x, False)
                enough = point[4] and len(valid) == len(powers) and all(
                    run['telemetry']['power_w']['sample_count'] >= 5 for run in backend['runs'])
                figures[power_name][group].append([point[0], float(np.median(valid)), min(valid), max(valid), enough])
    inventory = []
    for name, series in sorted(figures.items()):
        path = output / (name + '.png')
        draw_curves(series, path, *descriptions[name])
        inventory.append(dict(path=str(path), series=list(series),
                              error_bars='min/max of separate-run percentiles (drop plots use separate-run drop fractions)',
                              center='median across runs; never pooled samples'))
    (output / 'inventory.json').write_text(json.dumps(inventory, indent=2) + '\n')
    return inventory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('summary', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = json.loads(args.summary.read_text())
    plot_report(report, args.output)


if __name__ == '__main__':
    main()

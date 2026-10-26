"""Recompute host latency distributions and correlate sampled telemetry.

Analyze each run separately. Cross-run summaries retain the distribution of run
percentiles; pooling samples would hide run-to-run variability and overload.
"""
import argparse
import csv
import datetime
import json
from pathlib import Path

import numpy as np


def distribution(values):
    values = np.asarray(values, dtype=np.float64)
    if not len(values):
        return {}
    if not np.all(np.isfinite(values)) or np.any(values < 0):
        raise ValueError('Invalid latency samples')
    # Match the native nearest-rank definition instead of interpolating tails.
    ordered = np.sort(values)
    result = {name: float(ordered[int(np.ceil(q * len(ordered))) - 1])
              for name, q in [('p50_us', .5), ('p95_us', .95), ('p99_us', .99),
                              ('p999_us_diagnostic', .999)]}
    return dict(result, worst_us=float(ordered[-1]))


def samples(path):
    data = np.genfromtxt(path, delimiter=',', names=True, dtype=np.float64, ndmin=1)
    if not len(data) or not np.array_equal(data['event'], np.arange(len(data))):
        raise ValueError('Samples must contain every offered event ID in order')
    dropped = data['dropped'] if 'dropped' in data.dtype.names else np.zeros(len(data))
    if not np.all(np.isin(dropped, [0, 1])):
        raise ValueError('Invalid drop flags')
    kept = data[dropped == 0]
    if not len(kept) or not np.isfinite(kept['score']).all():
        raise ValueError('No finite completed scores')
    submit, complete = kept['submit_ns'], kept['complete_ns']
    if not np.all(np.isfinite(submit)) or not np.all(np.isfinite(complete)) or np.any(submit <= 0):
        raise ValueError('Invalid host timestamps')
    result = dict(offered=len(data), completed=len(kept), dropped=int(dropped.sum()),
                  service=distribution((complete - submit) / 1000),
                  measured_start_ns=float(submit.min()), measured_end_ns=float(complete.max()))
    if 'intended_ns' in data.dtype.names:
        result['arrival'] = distribution((complete - kept['intended_ns']) / 1000)
        result['queueing'] = distribution((submit - kept['enqueue_ns']) / 1000)
        result['injection'] = distribution((data['enqueue_ns'] - data['intended_ns']) / 1000)
        result['measured_start_ns'] = float(data['intended_ns'].min())
    result['drop_fraction'] = result['dropped'] / result['offered']
    return result, kept


def telemetry(prefix, start_ns, end_ns):
    clock_path, csv_path = Path(str(prefix) + '.clock.json'), Path(str(prefix) + '.telemetry.csv')
    if not clock_path.exists() or not csv_path.exists():
        return dict(status='unavailable', sample_count=0)
    clock = json.loads(clock_path.read_text())
    start_offset = clock['start_wall_ns'] - clock['start_monotonic_ns']
    end_offset = clock['end_wall_ns'] - clock['end_monotonic_ns']
    # A wall-clock step invalidates correlation; never silently average across it.
    drift = end_offset - start_offset
    if abs(drift) > 5_000_000:
        return dict(status='clock-correlation-invalid', offset_drift_ns=drift, sample_count=0)
    zone = datetime.timezone(datetime.timedelta(seconds=clock['utc_offset_seconds']))
    values = []
    invalid = 0
    with csv_path.open() as file:
        reader = csv.reader(file)
        next(reader, None)
        for row in reader:
            try:
                stamp = datetime.datetime.strptime(row[0].strip(), '%Y/%m/%d %H:%M:%S.%f').replace(tzinfo=zone)
                monotonic = stamp.timestamp() * 1e9 - (start_offset + end_offset) / 2
                if start_ns <= monotonic <= end_ns:
                    # Missing metrics are recorded as NaN, not substituted with zero.
                    metrics = []
                    for item in row[1:6]:
                        try:
                            metrics.append(float(item.strip().split()[0]))
                        except (ValueError, IndexError):
                            metrics.append(float('nan'))
                    if len(metrics) != 5:
                        raise ValueError('Incomplete telemetry row')
                    values.append([monotonic, *metrics])
            except (ValueError, IndexError):
                invalid += 1
    result = dict(status='sampled' if values else 'no-steady-state-samples', sample_count=len(values),
                  invalid_rows=invalid, offset_drift_ns=drift,
                  correlation_uncertainty_ns=abs(drift) / 2 + 1_000_000,
                  interval_ms=clock['telemetry_interval_ms'],
                  boundary='first measured intended/submission to final completion; startup excluded')
    if values:
        matrix = np.asarray(values)
        for index, name in enumerate(['power_w', 'graphics_mhz', 'memory_mhz', 'temperature_c', 'gpu_utilization_percent'], 1):
            valid = matrix[:, index][np.isfinite(matrix[:, index])]
            result[name] = dict(sample_count=len(valid), sample_mean=float(valid.mean()) if len(valid) else None,
                                min=float(valid.min()) if len(valid) else None,
                                max=float(valid.max()) if len(valid) else None)
        result['observed_span_seconds'] = float((matrix[-1, 0] - matrix[0, 0]) / 1e9)
    return result


def analyze_run(prefix):
    prefix = Path(prefix)
    metadata = json.loads(Path(str(prefix) + '.json').read_text())
    if metadata.get('mode', 'service') not in ['service', 'scheduled'] or metadata.get('correctness_checked'):
        raise ValueError('Correctness/device runs are not host performance evidence')
    result, kept = samples(Path(str(prefix) + '.csv'))
    if metadata.get('completed', metadata['events']) != result['completed']:
        raise ValueError('Metadata completion count differs from raw samples')
    if metadata.get('dropped', 0) != result['dropped']:
        raise ValueError('Metadata drop count differs from raw samples')
    result.update(prefix=str(prefix), backend=metadata['backend'], precision=metadata['precision'],
                  config=metadata['config'], checkpoint_sha256=metadata['checkpoint_sha256'],
                  trace_sha256=metadata['trace_sha256'], startup_ms=metadata['startup_ms'],
                  service_rate_per_second=metadata['service_rate_per_second'],
                  cpu_core_equivalents=metadata['cpu_core_equivalents'],
                  final_sample_count=result['completed'] >= 100000,
                  provenance=metadata.get('provenance'), mode=metadata.get('mode', 'service'))
    if 'deadline_us' in metadata:
        duration = (kept['complete_ns'] - kept['intended_ns']) / 1000
        result['deadline_misses_completed'] = int(np.count_nonzero(duration > metadata['deadline_us']))
        result['deadline_failures_including_drops'] = result['deadline_misses_completed'] + result['dropped']
    result['telemetry'] = telemetry(prefix, result['measured_start_ns'], result['measured_end_ns'])
    return result


def analyze_transport(prefix, warmup):
    prefix = Path(prefix)
    metadata = json.loads(Path(str(prefix) + '.json').read_text())
    if metadata['mode'] not in ['ping-pong', 'gpu-ping', 'empty', 'empty-graph', 'copy']:
        raise ValueError('Not a transport measurement')
    data = np.genfromtxt(Path(str(prefix) + '.csv'), delimiter=',', names=True, dtype=np.float64, ndmin=1)
    if (metadata['failed'] or len(data) != metadata['events']
            or not np.array_equal(data['sequence'], np.arange(1, len(data) + 1))):
        raise ValueError('Incomplete or failed transport evidence')
    if warmup < 0 or warmup >= len(data):
        raise ValueError('Invalid transport warmup discard')
    measured = data[warmup:]
    host = distribution(measured['host_observation_us'])
    result = dict(prefix=str(prefix), mode=metadata['mode'], boundary=metadata['boundary'],
                  poll_policy=metadata['poll_policy'], raw_events=len(data), warmup_discarded=warmup,
                  measured_events=len(measured), final_sample_count=len(measured) >= 100000,
                  host_observation=host, provenance=metadata['provenance'],
                  cpu_core_equivalents_including_warmup=metadata['cpu_seconds'] / metadata['wall_seconds'],
                  telemetry_status='No per-event monotonic timestamps; steady-state correlation unavailable')
    if metadata['mode'] in ['ping-pong', 'gpu-ping']:
        cycle_distribution = distribution(measured['device_round_trip_or_wait_cycles'])
        result['device_cycles'] = {key.replace('_us', '_cycles'): value for key, value in cycle_distribution.items()}
        result['device_cycle_boundary'] = 'GPU full round trip' if metadata['mode'] == 'gpu-ping' else 'GPU request polling wait'
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('prefixes', type=Path, nargs='+')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--plot', action='store_true')
    args = parser.parse_args()
    report = [analyze_run(prefix) for prefix in args.prefixes]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    if args.plot:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        figure, axes = plt.subplots(figsize=(7, 4))
        for prefix, run in zip(args.prefixes, report):
            _, kept = samples(Path(str(prefix) + '.csv'))
            start = kept['intended_ns'] if run['mode'] == 'scheduled' else kept['submit_ns']
            ordered = np.sort((kept['complete_ns'] - start) / 1000)
            axes.step(ordered, np.arange(1, len(ordered) + 1) / len(ordered),
                      where='post', label=f'{prefix.name} (dropped {run["drop_fraction"]:.2%})')
        axes.set(xlabel='Host latency (µs)', ylabel='Completed-event empirical CDF', xscale='log')
        axes.legend(fontsize='small')
        figure.tight_layout()
        figure.savefig(args.output.with_suffix('.png'), dpi=180)
        plt.close(figure)


if __name__ == '__main__':
    main()

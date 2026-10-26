"""Summarize repeated study conditions without pooling their latency samples."""
import argparse
import collections
import json
from pathlib import Path

import numpy as np

from bench.analyze import analyze_run, analyze_transport
from bench.runner import cached_result


def option(command, name, default=None):
    return command[command.index(name) + 1] if name in command else default


def workload(record, run):
    command = record['command']
    phase = record['label'].split('-', 1)[0]
    return dict(phase=phase, size=record['size'], precision=record['precision'],
                config=run['config'], checkpoint_sha256=run['checkpoint_sha256'],
                trace_sha256=run['trace_sha256'], offered=run['offered'],
                rate=option(command, '--rate'), bursty=option(command, '--bursty', '0'),
                idle_us=option(command, '--idle-us', '0'), warmup=option(command, '--warmup'),
                worker_core=option(command, '--worker-core', '-1'),
                clock_regime=option(command, '--clock-regime', 'default-dynamic-affinity'
                                    if option(command, '--worker-core', '-1') != '-1'
                                    else 'default-dynamic-uncontrolled'))


def repeated_condition(runs):
    repeats = [run['repeat'] for run in runs]
    if len(repeats) != len(set(repeats)):
        raise ValueError('Duplicate repetition IDs in a condition')
    metric = 'arrival' if runs[0]['mode'] == 'scheduled' else 'service'
    p99 = [run[metric]['p99_us'] for run in runs]
    immutable = all(run.get('provenance') and run['provenance'].get('git_commit') for run in runs)
    identities = {json.dumps({key: (run.get('provenance') or {}).get(key)
                              for key in ['git_commit', 'executable_sha256', 'source_sha256',
                                          'python_executable_sha256']}, sort_keys=True) for run in runs}
    same_implementation = len(identities) == 1
    enough = len(runs) >= 5 and all(run['final_sample_count'] for run in runs)
    return dict(repetitions=len(runs), repeat_ids=repeats, percentile_boundary=metric,
                p99_us_by_run=p99, median_run_p99_us=float(np.median(p99)),
                min_run_p99_us=float(min(p99)), max_run_p99_us=float(max(p99)),
                stddev_run_p99_us=float(np.std(p99, ddof=1)) if len(p99) > 1 else None,
                final_repetition_gate=enough, immutable_provenance=immutable,
                same_implementation=same_implementation,
                final_evidence_eligible=enough and immutable and same_implementation,
                dropped_by_run=[run['dropped'] for run in runs],
                completed_by_run=[run['completed'] for run in runs],
                service_rate_by_run=[run['service_rate_per_second'] for run in runs],
                cpu_core_equivalents_by_run=[run['cpu_core_equivalents'] for run in runs])


def compare(reference, candidate):
    ref = {run['repeat']: run for run in reference}
    test = {run['repeat']: run for run in candidate}
    if set(ref) != set(test):
        return dict(status='unmatched-repetitions', reference_repeat_ids=sorted(ref),
                    candidate_repeat_ids=sorted(test), repeatable_improvement_candidate=False)
    metric = 'arrival' if reference[0]['mode'] == 'scheduled' else 'service'
    paired = []
    for repeat in sorted(ref):
        baseline, actual = ref[repeat], test[repeat]
        if any(baseline[key] != actual[key] for key in ['config', 'checkpoint_sha256', 'trace_sha256', 'offered', 'precision']):
            raise ValueError('Comparison changes the model, trace, precision or offered event count')
        p99_ref, p99_test = baseline[metric]['p99_us'], actual[metric]['p99_us']
        if p99_ref <= 0 or p99_test <= 0:
            raise ValueError('Nonpositive p99 cannot support a latency ratio')
        paired.append(dict(repeat=repeat, reference_p99_us=p99_ref, candidate_p99_us=p99_test,
                           improvement_fraction=1 - p99_test / p99_ref,
                           reference_drops=baseline['dropped'], candidate_drops=actual['dropped']))
    eligible = repeated_condition(reference)['final_evidence_eligible'] and repeated_condition(candidate)['final_evidence_eligible']
    return dict(status='paired', percentile_boundary=metric, pairs=paired,
                median_paired_improvement_fraction=float(np.median([p['improvement_fraction'] for p in paired])),
                all_repeats_lower_p99=all(p['improvement_fraction'] > 0 for p in paired),
                repeatable_improvement_candidate=eligible and all(
                    p['improvement_fraction'] > 0 and p['candidate_drops'] <= p['reference_drops'] for p in paired),
                interpretation='Candidate result only; requires baseline tuning, noise controls and full release audit')


def summarize(root):
    root = Path(root)
    records = json.loads((root / 'runs.json').read_text())
    conditions = collections.defaultdict(lambda: collections.defaultdict(list))
    attempts, transports, seen = [], [], set()
    for record in records:
        if 'archive' in record or record['returncode'] or not record.get('sample_gate_passed', False):
            attempts.append(record)
            continue
        if record['label'].startswith('tune-'):
            continue
        if record['label'] in seen:
            raise ValueError('Ambiguous duplicate active run label')
        seen.add(record['label'])
        prefix = root / record['label']
        stamp = json.loads(Path(str(prefix) + '.stamp.json').read_text())
        stamp.pop('artifacts', None)
        if cached_result(prefix, stamp) is None:
            raise ValueError(f'Missing or modified raw artifacts: {prefix}')
        if record.get('measurement') in ['ping-pong', 'gpu-ping', 'empty', 'empty-graph', 'copy']:
            transport = analyze_transport(prefix, record['transport_warmup_discard'])
            transport.update(repeat=record['repeat'], worker_core=option(record['command'], '--worker-core', '-1'))
            transports.append(transport)
            continue
        run = analyze_run(prefix)
        run['repeat'] = record['repeat']
        run['actual_backend'] = record['backend']
        key = json.dumps(workload(record, run), sort_keys=True)
        conditions[key][record['comparison_group']].append(run)
    result = dict(conditions=[], transport_runs=transports, prior_or_failed_attempts=attempts,
                  aggregation='Each percentile comes from one run; samples are never pooled across runs')
    for key, groups in sorted(conditions.items()):
        row = dict(workload=json.loads(key), backends={}, comparisons=[])
        for group, runs in groups.items():
            runs.sort(key=lambda run: run['repeat'])
            row['backends'][group] = dict(summary=repeated_condition(runs), runs=runs)
        for reference, candidate in [('graph', 'persistent'), ('cpu', 'graph'), ('cpu', 'persistent')]:
            if reference in groups and candidate in groups:
                row['comparisons'].append(dict(reference=reference, candidate=candidate,
                                                **compare(groups[reference], groups[candidate])))
        result['conditions'].append(row)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('study', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = summarize(args.study)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')


if __name__ == '__main__':
    main()

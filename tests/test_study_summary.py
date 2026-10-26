import copy
import json

import pytest

from bench.summarize import compare, repeated_condition, summarize
from bench.provenance import file_hash


def runs(p99_values, drops=0):
    return [dict(repeat=i, mode='service', service=dict(p99_us=value),
                 final_sample_count=True, completed=100000, offered=100000,
                 dropped=drops, precision='fp32', config={'width': 128},
                 checkpoint_sha256='synthetic-weights', trace_sha256='synthetic-trace',
                 service_rate_per_second=1000., cpu_core_equivalents=1.,
                 provenance=dict(git_commit='synthetic-commit', executable_sha256='synthetic-executable'))
            for i, value in enumerate(p99_values)]


def test_repeated_summary_preserves_run_variability_and_rejects_short_evidence():
    data = runs([10, 10, 10, 10, 100])
    result = repeated_condition(data)
    assert result['median_run_p99_us'] == 10
    assert result['max_run_p99_us'] == 100
    assert result['p99_us_by_run'] == [10, 10, 10, 10, 100]
    assert result['final_evidence_eligible']
    data[-1]['final_sample_count'] = False
    assert not repeated_condition(data)['final_evidence_eligible']
    assert not repeated_condition(data[:4])['final_evidence_eligible']


def test_pairs_include_losing_runs_and_drops_prevent_attractive_claims():
    baseline = runs([10] * 5)
    candidate = runs([8, 8, 8, 8, 12])
    result = compare(baseline, candidate)
    assert result['median_paired_improvement_fraction'] == pytest.approx(.2)
    assert not result['all_repeats_lower_p99']
    assert not result['repeatable_improvement_candidate']
    assert not compare(baseline, runs([8] * 5, drops=1))['repeatable_improvement_candidate']
    assert compare(baseline, runs([8] * 5))['repeatable_improvement_candidate']


def test_mismatched_repetitions_and_workloads_do_not_make_comparisons():
    baseline, candidate = runs([10] * 5), runs([8] * 5)
    assert compare(baseline, candidate[:4])['status'] == 'unmatched-repetitions'
    candidate[2]['trace_sha256'] = 'another-trace'
    with pytest.raises(ValueError, match='changes the model'):
        compare(baseline, candidate)
    candidate = copy.deepcopy(baseline)
    candidate[-1]['provenance']['executable_sha256'] = 'changed-executable'
    assert not repeated_condition(candidate)['same_implementation']
    candidate = copy.deepcopy(baseline)
    candidate[-1]['provenance']['git_commit'] = None
    assert not repeated_condition(candidate)['immutable_provenance']
    candidate[-1]['repeat'] = 0
    with pytest.raises(ValueError, match='Duplicate repetition'):
        repeated_condition(candidate)


def test_study_summary_reads_verified_artifacts_and_retains_failures(tmp_path):
    records = [dict(label='service-synthetic-failed', returncode=2, archive='synthetic-archive')]
    for group, latency in [('graph', 10), ('persistent', 8)]:
        for repeat in range(5):
            label = f'service-synthetic-{group}-{repeat}'
            prefix = tmp_path / label
            raw = tmp_path / (label + '.csv')
            raw.write_text('event,submit_ns,complete_ns,score\n'
                           f'0,100000,{100000 + latency * 1000},1\n')
            metadata = dict(events=1, backend=group, precision='fp32', config={'width': 8},
                            checkpoint_sha256='synthetic', trace_sha256='synthetic',
                            startup_ms=0., service_rate_per_second=1000., cpu_core_equivalents=1.,
                            mode='service', provenance={'git_commit': 'synthetic'})
            result = tmp_path / (label + '.json')
            result.write_text(json.dumps(metadata))
            identity = dict(command=['synthetic-bench'])
            stamp = dict(identity, artifacts={'.json': file_hash(result), '.csv': file_hash(raw)})
            (tmp_path / (label + '.stamp.json')).write_text(json.dumps(stamp))
            records.append(dict(label=label, returncode=0, sample_gate_passed=True,
                                command=identity['command'], repeat=repeat, size='synthetic',
                                precision='fp32', backend=group, comparison_group=group))
    (tmp_path / 'runs.json').write_text(json.dumps(records))
    report = summarize(tmp_path)
    assert len(report['prior_or_failed_attempts']) == 1
    assert len(report['conditions']) == 1
    condition = report['conditions'][0]
    assert condition['backends']['graph']['summary']['p99_us_by_run'] == [10] * 5
    comparison = condition['comparisons'][0]
    assert comparison['all_repeats_lower_p99']
    assert not comparison['repeatable_improvement_candidate']  # Only one synthetic sample per run.
    json.dumps(report, allow_nan=False)
    raw.write_text('modified')
    with pytest.raises(ValueError, match='modified raw artifacts'):
        summarize(tmp_path)

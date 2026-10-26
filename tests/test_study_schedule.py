import json
from types import SimpleNamespace

from bench.runner import Study


def study_with_recorded_capacities(tmp_path):
    study = Study.__new__(Study)
    study.root = tmp_path
    study.a = SimpleNamespace(sizes=['small'], modes=['fp32'], runs=1, events=100000)
    study.selection = {'small-fp32': {
        'cpu': {'backend': 'cpu', 'threads': 1},
        'graph': {'backend': 'cublas-graph', 'blocks': 4},
        'persistent': {'backend': 'persistent', 'blocks': 1}}}
    for group, capacity in [('cpu', 1000), ('graph', 2000), ('persistent', 4000)]:
        (tmp_path / f'service-small-fp32-{group}-0.json').write_text(
            json.dumps({'service_rate_per_second': capacity}))
    calls = []

    def capture(label, size, mode, backend, repeat, extra, offered, group):
        calls.append(dict(label=label, group=group, offered=offered,
                          rate=float(extra[extra.index('--rate') + 1]),
                          burst=extra[extra.index('--bursty') + 1]))

    study.execute = capture
    return study, calls


def test_comparative_load_uses_identical_absolute_rates_and_counts(tmp_path):
    study, calls = study_with_recorded_capacities(tmp_path)
    study.scheduled()
    assert len(calls) == 30
    for rate in [500, 1000, 1500, 1800, 2200]:
        for burst in ['0', '1']:
            matched = [call for call in calls if call['rate'] == rate and call['burst'] == burst]
            assert {call['group'] for call in matched} == {'cpu', 'graph', 'persistent'}
            assert len({call['offered'] for call in matched}) == 1
            assert matched[0]['offered'] >= 100000


def test_saturation_curves_explicitly_use_each_backend_capacity(tmp_path):
    study, calls = study_with_recorded_capacities(tmp_path)
    study.saturation()
    assert len(calls) == 18
    for group, capacity in [('cpu', 1000), ('graph', 2000), ('persistent', 4000)]:
        matched = [call for call in calls if call['group'] == group]
        assert sorted(call['rate'] / capacity for call in matched) == [.25, .5, .75, .9, 1.1, 1.5]
        assert all(call['offered'] >= 100000 for call in matched)
    anchor = json.loads((tmp_path / 'saturation-anchors-small-fp32.json').read_text())
    assert anchor['kind'] == 'backend-specific'


def test_noise_controls_preserve_workload_and_transport_warmup_count(tmp_path):
    study, _ = study_with_recorded_capacities(tmp_path)
    study.a.transport_warmup = 1000
    calls = []

    def capture(*args, **kwargs):
        calls.append((args, kwargs))

    study.execute = capture
    study.noise()
    assert len(calls) == 18
    for controlled in [False, True]:
        selected = [(args, kwargs) for args, kwargs in calls if kwargs['controlled'] == controlled]
        transport = [args for args, kwargs in selected if args[0].startswith('noise-transport')]
        assert len(transport) == 7
        assert all(args[6] == 101000 for args in transport)
        assert sum(args[3] == 'persistent' for args, kwargs in selected) == 1
        assert sum(args[3] == 'cublas-graph' for args, kwargs in selected) == 1

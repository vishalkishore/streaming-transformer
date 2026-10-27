from types import SimpleNamespace

import bench.runner as runner


def test_each_candidate_is_validated_before_any_timing(tmp_path, monkeypatch):
    study = runner.Study.__new__(runner.Study)
    study.root = tmp_path
    study.a = SimpleNamespace(sizes=['small'], modes=['fp32'], tune_events=100)
    study.selection = {}
    monkeypatch.setattr(runner, 'read', lambda path: {'sm_count': 16} if 'hardware' in str(path) else {'window': 3})
    calls = []

    def execute(label, size, mode, backend, repeat, extra, events):
        calls.append(dict(label=label, backend=backend, extra=extra, events=events))
        return dict(legal_cooperative_blocks=4, service=dict(p99_us=10.))

    study.execute = execute
    study.tune()
    checked = set()
    for call in calls:
        if call['label'].endswith('-persistent-legal'):
            continue
        extra = call['extra'].copy()
        correctness = '--mode' in extra
        if correctness:
            index = extra.index('--mode')
            assert extra[index + 1] == 'correctness'
            del extra[index:index + 2]
        identity = (call['backend'], tuple(extra))
        if correctness:
            checked.add(identity)
            assert call['events'] == 18
        else:
            assert identity in checked
            assert call['events'] == 100
        if call['backend'] == 'persistent':
            assert int(extra[extra.index('--blocks') + 1]) <= 4
    assert len(checked) == 21  # 16 graph/transport choices + 3 CPU + 2 legal persistent.
    assert len(calls) == 1 + 21 + 3 * 21
    assert set(study.selection['small-fp32']) == {'cpu', 'graph', 'persistent'}

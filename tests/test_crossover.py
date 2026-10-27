from scripts.check_crossover import summarize


def test_crossover_keeps_boundaries_and_repetitions_separate():
    rows = []
    for mode, graph, persistent in [('device', [10, 12, 50], [9, 11, 60]),
                                     ('service', [20, 22, 90], [25, 27, 100])]:
        metric = 'device_us_per_event' if mode == 'device' else 'service_p99_us'
        for backend, values in [('graph', graph), ('persistent', persistent)]:
            rows.extend(dict(size='main', precision='fp32', mode=mode, backend=backend,
                             **{metric: value}) for value in values)
    result = {row['mode']: row for row in summarize(rows)}
    assert result['device']['graph_us'] == 12
    assert result['device']['persistent_over_graph'] == 11 / 12
    assert result['service']['graph_us'] == 22
    assert result['service']['persistent_over_graph'] == 27 / 22
    assert result['service']['runs_by_backend'] == {'graph': 3, 'persistent': 3}


def test_crossover_does_not_compare_missing_backend():
    assert summarize([dict(size='small', precision='mixed', mode='device',
                           backend='graph', device_us_per_event=5)]) == []

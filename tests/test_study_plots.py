from bench.plot_study import parameter_count, plot_report


def test_fixed_model_parameter_counts():
    assert parameter_count(dict(features=32, width=64, ffn=128, layers=1)) == 43264
    assert parameter_count(dict(features=32, width=128, ffn=256, layers=2)) == 332544
    assert parameter_count(dict(features=32, width=256, ffn=512, layers=4)) == 2632192


def test_plot_exports_mark_diagnostic_data_and_keep_drop_figures(tmp_path):
    report = dict(conditions=[])
    for phase in ['service', 'window', 'scheduled', 'saturation']:
        backend = dict(summary=dict(median_run_p99_us=10., min_run_p99_us=9.,
                                    max_run_p99_us=12., final_evidence_eligible=False),
                       runs=[dict(drop_fraction=.1)])
        report['conditions'].append(dict(
            workload=dict(phase=phase, precision='fp32', size='synthetic', rate='1000', bursty='0',
                          config=dict(features=32, width=64, ffn=128, layers=1, window=64)),
            backends=dict(graph=backend)))
    inventory = plot_report(report, tmp_path)
    assert len(inventory) == 6
    assert len(list(tmp_path.glob('*.png'))) == 6
    assert (tmp_path / 'inventory.json').exists()
    assert all(path.stat().st_size > 1000 for path in tmp_path.glob('*.png'))


def test_idle_power_plot_requires_observed_power_samples(tmp_path):
    backend = dict(summary=dict(median_run_p99_us=10., min_run_p99_us=9.,
                               max_run_p99_us=12., final_evidence_eligible=False),
                   runs=[dict(telemetry=dict(power_w=dict(sample_mean=40., sample_count=10)))])
    report = dict(conditions=[dict(
        workload=dict(phase='idle', precision='mixed', size='main', idle_us='0'),
        backends=dict(persistent=backend))])
    inventory = plot_report(report, tmp_path)
    assert len(inventory) == 2
    assert (tmp_path / 'idle-main-mixed-power.png').is_file()

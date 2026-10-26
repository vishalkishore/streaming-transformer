import json

import numpy as np
import pytest

from bench.analyze import distribution, samples, telemetry, analyze_transport


def test_percentiles_match_native_nearest_rank():
    result = distribution(np.arange(1, 101))
    assert result == dict(p50_us=50., p95_us=95., p99_us=99.,
                          p999_us_diagnostic=100., worst_us=100.)
    with pytest.raises(ValueError, match='Invalid latency'):
        distribution([1, np.nan])


def test_drops_are_retained_without_becoming_latency_samples(tmp_path):
    path = tmp_path / 'run.csv'
    path.write_text('event,intended_ns,enqueue_ns,submit_ns,complete_ns,score,dropped\n'
                    '0,10000,11000,12000,16000,1,0\n'
                    '1,12000,13000,0,0,0,1\n'
                    '2,14000,15000,18000,24000,2,0\n')
    result, kept = samples(path)
    assert result['offered'] == 3 and result['completed'] == 2 and result['dropped'] == 1
    assert result['service']['p99_us'] == 6
    assert result['arrival']['p99_us'] == 10
    assert result['queueing']['p99_us'] == 3
    assert result['injection']['p99_us'] == 1
    assert len(kept) == 2 and result['drop_fraction'] == 1 / 3
    path.write_text(path.read_text().replace('2,14000', '3,14000'))
    with pytest.raises(ValueError, match='every offered event'):
        samples(path)


def test_telemetry_excludes_startup_and_preserves_unavailable_metrics(tmp_path):
    prefix = tmp_path / 'run'
    clock = dict(start_wall_ns=1_000_000_000, start_monotonic_ns=10_000_000_000,
                 end_wall_ns=4_000_000_000, end_monotonic_ns=13_000_000_000,
                 utc_offset_seconds=0, telemetry_interval_ms=200)
    (tmp_path / 'run.clock.json').write_text(json.dumps(clock))
    (tmp_path / 'run.telemetry.csv').write_text(
        'timestamp,power.draw,clocks.current.graphics,clocks.current.memory,temperature.gpu,utilization.gpu\n'
        '1970/01/01 00:00:01.000,300 W,2000 MHz,1000 MHz,60,100 %\n'
        '1970/01/01 00:00:02.000,40 W,400 MHz,1000 MHz,40,10 %\n'
        '1970/01/01 00:00:03.000,[N/A],600 MHz,1000 MHz,42,20 %\n')
    result = telemetry(prefix, 10_500_000_000, 12_500_000_000)
    assert result['sample_count'] == 2
    assert result['power_w']['sample_count'] == 1
    assert result['power_w']['sample_mean'] == 40
    assert result['graphics_mhz']['sample_mean'] == 500
    clock['end_wall_ns'] += 100_000_000
    (tmp_path / 'run.clock.json').write_text(json.dumps(clock))
    assert telemetry(prefix, 10_500_000_000, 12_500_000_000)['status'] == 'clock-correlation-invalid'


def test_transport_warmup_and_directional_boundaries_are_explicit(tmp_path):
    prefix = tmp_path / 'transport'
    (tmp_path / 'transport.json').write_text(json.dumps(dict(
        mode='gpu-ping', failed=0, events=4, boundary='host wait and acknowledgement',
        provenance={'git_commit': 'synthetic'}, poll_policy='tight', cpu_seconds=1., wall_seconds=2.)))
    raw = tmp_path / 'transport.csv'
    raw.write_text('sequence,host_observation_us,device_round_trip_or_wait_cycles\n'
                   '1,1000,9000\n2,2,400\n3,3,500\n4,4,600\n')
    result = analyze_transport(prefix, warmup=1)
    assert result['host_observation']['p99_us'] == 4
    assert result['device_cycles']['p99_cycles'] == 600
    assert result['device_cycle_boundary'] == 'GPU full round trip'
    assert result['measured_events'] == 3
    assert not result['final_sample_count']
    assert result['cpu_core_equivalents_including_warmup'] == .5
    with pytest.raises(ValueError, match='warmup discard'):
        analyze_transport(prefix, warmup=4)
    raw.write_text(raw.read_text().replace('3,3,500', '4,3,500'))
    with pytest.raises(ValueError, match='Incomplete or failed'):
        analyze_transport(prefix, warmup=1)

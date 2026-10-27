import json
import os

import pytest

from bench import gpu_guard


def test_process_ownership_accepts_descendants_only():
    assert gpu_guard.belongs_to_run(os.getpid(), os.getpid())
    assert gpu_guard.belongs_to_run(os.getpid(), os.getppid())
    assert not gpu_guard.belongs_to_run(os.getpid(), -1)


def test_process_query_includes_graphics_and_compute(monkeypatch):
    xml = '''<nvidia_smi_log><gpu><uuid>GPU-test</uuid><processes>
    <process_info><pid>12</pid><type>G</type><process_name>graphics</process_name></process_info>
    <process_info><pid>34</pid><type>C</type><process_name>compute</process_name></process_info>
    </processes></gpu></nvidia_smi_log>'''
    monkeypatch.setattr(gpu_guard.subprocess, 'check_output', lambda *a, **k: xml)
    assert [p['pid'] for p in gpu_guard.gpu_processes()] == [12, 34]


def test_missing_process_visibility_fails(monkeypatch):
    monkeypatch.setattr(gpu_guard.subprocess, 'check_output',
                        lambda *a, **k: '<nvidia_smi_log><gpu/></nvidia_smi_log>')
    with pytest.raises(RuntimeError, match='visibility'):
        gpu_guard.gpu_processes()


def test_guard_waits_for_idle_and_records_competitor(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    readings = iter([[dict(pid=42)], [], [dict(pid=99)]])
    monkeypatch.setattr(gpu_guard, 'gpu_processes', lambda: next(readings))
    sleeps = []
    monkeypatch.setattr(gpu_guard.time, 'sleep', sleeps.append)
    evidence = tmp_path / 'guard.json'
    with pytest.raises(RuntimeError, match='GPU run invalid'):
        with gpu_guard.GpuGuard(evidence) as guard:
            guard.pid = 12
    assert sleeps == [5]
    assert json.loads(evidence.read_text())['valid'] is False


def test_guard_allows_child_and_rejects_query_failure(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(gpu_guard, 'gpu_processes', lambda: [])
    evidence = tmp_path / 'guard.json'
    with gpu_guard.GpuGuard(evidence) as guard:
        guard.pid = 12
        monkeypatch.setattr(gpu_guard, 'gpu_processes', lambda: [dict(pid=12)])
    assert json.loads(evidence.read_text())['valid'] is True
    monkeypatch.setattr(gpu_guard, 'gpu_processes', lambda: [])
    with pytest.raises(RuntimeError, match='GPU run invalid'):
        with gpu_guard.GpuGuard(evidence):
            def fail():
                raise RuntimeError('query unavailable')
            monkeypatch.setattr(gpu_guard, 'gpu_processes', fail)
    assert 'query_error' in json.loads(evidence.read_text())['violations'][0]

import json

from bench.provenance import file_hash
from bench.runner import archive_attempt, cached_result


def test_guarded_cache_requires_valid_unchanged_process_evidence(tmp_path):
    prefix = tmp_path / 'guarded'
    identity = dict(command=['bench'], gpu_exclusivity_policy='xml-process-watch-v1')
    for suffix, content in [('.json', '{"completed": 100000}'), ('.csv', 'raw'),
                            ('.gpu-processes.json', '{"valid": true}')]:
        tmp_path.joinpath('guarded' + suffix).write_text(content)
    artifacts = {suffix: file_hash(tmp_path / ('guarded' + suffix))
                 for suffix in ['.json', '.csv', '.gpu-processes.json']}
    (tmp_path / 'guarded.stamp.json').write_text(json.dumps(dict(identity, artifacts=artifacts)))
    assert cached_result(prefix, identity) == {'completed': 100000}
    (tmp_path / 'guarded.gpu-processes.json').write_text('{"valid": false}')
    assert cached_result(prefix, identity) is None
    artifacts['.gpu-processes.json'] = file_hash(tmp_path / 'guarded.gpu-processes.json')
    (tmp_path / 'guarded.stamp.json').write_text(json.dumps(dict(identity, artifacts=artifacts)))
    assert cached_result(prefix, identity) is None


def test_resume_requires_unchanged_complete_raw_artifacts(tmp_path):
    prefix = tmp_path / 'condition'
    identity = dict(command=['bench', '--events', '100000'], binary='test-hash')
    result = tmp_path / 'condition.json'
    raw = tmp_path / 'condition.csv'
    result.write_text('{"completed": 100000}')
    raw.write_text('synthetic raw artifact\n')
    stamp = dict(identity, artifacts={'.json': file_hash(result), '.csv': file_hash(raw)})
    (tmp_path / 'condition.stamp.json').write_text(json.dumps(stamp))
    assert cached_result(prefix, identity) == {'completed': 100000}
    assert cached_result(prefix, dict(identity, binary='changed')) is None
    raw.write_text('truncated')
    assert cached_result(prefix, identity) is None
    raw.unlink()
    assert cached_result(prefix, identity) is None


def test_retry_archives_failed_artifacts_and_retains_attempt_identity(tmp_path):
    prefix = tmp_path / 'condition'
    (tmp_path / 'condition.log').write_text('original failure')
    (tmp_path / 'condition.json').write_text('{"completed": 5}')
    (tmp_path / 'condition-other.log').write_text('unrelated')
    records = [dict(label='condition', returncode=2)]
    archive = archive_attempt(prefix, records)
    assert (archive / 'condition.log').read_text() == 'original failure'
    assert records[0]['archive'] == str(archive)
    assert (tmp_path / 'condition-other.log').read_text() == 'unrelated'
    assert not (tmp_path / 'condition.json').exists()
    (tmp_path / 'condition.log').write_text('second failure')
    records.append(dict(label='condition', returncode=1))
    second = archive_attempt(prefix, records)
    assert second != archive
    assert records[0]['archive'] == str(archive)
    assert records[1]['archive'] == str(second)

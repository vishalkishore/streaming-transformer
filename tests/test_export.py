import json

import pytest

from reference.export import export
from scripts.fixtures import verify


def test_fixture_regeneration_and_corruption(tmp_path):
    config = tmp_path / 'config.json'
    config.write_text(json.dumps(dict(layers=2, width=8, heads=2, ffn=16, window=3, features=4)))
    first, second = tmp_path / 'first', tmp_path / 'second'
    export(config, first, mode='mixed')
    export(config, second, mode='mixed')
    assert verify(first) == verify(second)
    assert json.loads((first / 'resets.json').read_text()) == [9]
    # Detect transfer damage in both the trace and the independently hashed state archive.
    trace = (first / 'trace.bin').read_bytes()
    (first / 'trace.bin').write_bytes(bytes([trace[0] ^ 1]) + trace[1:])
    with pytest.raises(ValueError, match='Trace mismatch'):
        verify(first)
    (first / 'trace.bin').write_bytes(trace)
    archive = first / 'expected/tensors.bin'
    raw = archive.read_bytes()
    archive.write_bytes(raw[:-1])
    with pytest.raises(ValueError, match='Checksum mismatch'):
        verify(first)

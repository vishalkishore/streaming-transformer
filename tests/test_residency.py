import pytest

from bench.residency import tile_bytes


def test_weight_partition_accounts_for_warp_rows_and_norm_channels():
    manifest = dict(tensors=[dict(shape=[10, 8], dtype='float16'), dict(shape=[256], dtype='float32')])
    # Two blocks with four warps each: block0 owns rows0..3 and8..9;
    # block1 owns rows4..7. Each block owns128 normalization channels.
    assert tile_bytes(manifest, 2) == [6 * 8 * 2 + 128 * 4, 4 * 8 * 2 + 128 * 4]
    assert tile_bytes(manifest, 1) == [10 * 8 * 2 + 256 * 4]
    assert sum(tile_bytes(manifest, 110)) == 10 * 8 * 2 + 256 * 4
    with pytest.raises(ValueError, match='whole warps'):
        tile_bytes(manifest, 1, threads=33)

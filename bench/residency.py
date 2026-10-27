"""Account for weight tiles under the existing warp-row/block ownership.

This is a resource feasibility calculation, not evidence of weight residency,
physical block placement, actual occupancy or measured memory traffic.
"""
import argparse
import json
from pathlib import Path


def tile_bytes(manifest, blocks, threads=128):
    if blocks <= 0 or threads <= 0 or threads % 32:
        raise ValueError('Positive grid and whole warps are required')
    sizes = [0] * blocks
    for tensor in manifest['tensors']:
        shape = tensor['shape']
        element_bytes = {'float16': 2, 'float32': 4}[tensor['dtype']]
        if len(shape) == 2:
            # matvec(): a warp owns a complete output row, cycling across grid warps.
            rows, columns = shape
            for row in range(rows):
                sizes[(row // (threads // 32)) % blocks] += columns * element_bytes
        elif len(shape) == 1:
            # rms_norm(): output channels cycle across all grid threads.
            for channel in range(shape[0]):
                sizes[(channel // threads) % blocks] += element_bytes
        else:
            raise ValueError('Unexpected model weight rank')
    return sizes


def footprint(manifest, hardware):
    config = manifest['config']
    width, layers, window = config['width'], config['layers'], config['window']
    dtype_bytes = 2 if manifest['mode'] == 'mixed' else 4
    state_bytes = 2 * layers * width * window * dtype_bytes
    scratch_bytes = 4 * (9 * width + 2 * config['ffn'] + config['heads'] * window
                         + layers * width + config['features'] + 1)
    total = sum(tensor['nbytes'] for tensor in manifest['tensors'])
    grids = sorted(set([1, 4, 8, 16, hardware['sm_count']]))
    rows = []
    for blocks in grids:
        tiles = tile_bytes(manifest, blocks)
        if sum(tiles) != total:
            raise ValueError('Weight tile accounting differs from checkpoint bytes')
        maximum = max(tiles)
        rows.append(dict(blocks=blocks, weight_tile_bytes_by_block=tiles,
                         max_weight_tile_bytes=maximum, min_weight_tile_bytes=min(tiles),
                         fits_default_block_shared_limit=maximum <= hardware['shared_bytes_per_block'],
                         shared_memory_only_blocks_per_sm_upper_bound=hardware['shared_bytes_per_sm'] // maximum))
    return dict(config=config, precision=manifest['mode'], checkpoint_sha256=manifest['sha256'],
                total_weight_bytes=total, kv_bytes_global=state_bytes, scratch_bytes_global=scratch_bytes,
                grids=rows, shared_optin_limit='Not recorded in this probe; must query before an opt-in experiment',
                interpretation='Weight-only per-block staging estimate; KV/scratch remain global. Actual kernel resource allocation and legal occupancy must be measured.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--hardware', type=Path, default=Path('results/blackwell-hardware.json'))
    parser.add_argument('--output', type=Path, default=Path('results/weight-footprints.json'))
    args = parser.parse_args()
    hardware = json.loads(args.hardware.read_text())
    report = [footprint(json.loads(Path(f'fixtures/{size}-{mode}/manifest.json').read_text()), hardware)
              for size in ['small', 'main', 'large'] for mode in ['fp32', 'mixed']]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()

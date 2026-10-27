"""Cold-process regression for persistent snapshots under CUDA loading modes."""
import argparse
import itertools
import json
import os
import subprocess
from pathlib import Path

from bench.gpu_guard import run_exclusive


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary', default='build-blackwell/stream-bench')
    parser.add_argument('--output', type=Path, default=Path('results/cuda-loading-regression'))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    rows = []
    for loading, precision, attention, blocks in itertools.product(
            ['LAZY', 'EAGER'], ['fp32', 'mixed'], ['staged', 'fused'], [1, 110]):
        prefix = args.output / f'{loading.lower()}-{precision}-{attention}-{blocks}'
        # Two RUNs detect an idle exit hidden by a blocking first snapshot.
        command = [args.binary, '--config', 'configs/small.json', '--weights', f'fixtures/small-{precision}',
                   '--backend', 'persistent', '--blocks', str(blocks), '--events', '2', '--warmup', '0',
                   '--mode', 'correctness', '--idle-seconds', '1', '--attention', attention, '--output', str(prefix)]
        with Path(str(prefix) + '.log').open('w') as out:
            result = run_exclusive(command, evidence=str(prefix) + '.gpu-processes.json',
                                   env=dict(os.environ, CUDA_MODULE_LOADING=loading),
                                   stdout=out, stderr=subprocess.STDOUT)
        rows.append(dict(command=command, cuda_module_loading=loading, returncode=result.returncode))
        (args.output / 'summary.json').write_text(json.dumps(rows, indent=2))
        print(prefix.name, 'PASS' if result.returncode == 0 else 'FAIL', flush=True)
        if result.returncode:
            raise RuntimeError(f'Cold-process loading regression failed: {prefix}')


if __name__ == '__main__':
    main()

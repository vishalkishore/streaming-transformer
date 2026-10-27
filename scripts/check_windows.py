"""Validate cache-shape variants before collecting their performance samples."""
from bench.gpu_guard import run_exclusive

import itertools
import json
import subprocess
from pathlib import Path


def main():
    output = Path('results/window-validation')
    output.mkdir(parents=True, exist_ok=True)
    report = []
    for size, window, mode in itertools.product(['small', 'main', 'large'], [32, 64, 128, 256, 512], ['fp32', 'mixed']):
        variant = f'{size}-w{window}'
        # Full oracle snapshots exercise partial/full/wrapped caches and reset.
        for backend, blocks in [('cuda', 4), ('cublas-graph', 4), ('persistent', 1), ('persistent', 4)]:
            prefix = output / f'{variant}-{mode}-{backend}-{blocks}'
            command = ['build-blackwell/stream-bench', '--config', f'configs/{variant}.json',
                       '--weights', f'fixtures/{variant}-{mode}', '--backend', backend,
                       '--blocks', str(blocks), '--mode', 'correctness', '--events', str(3 * window + 9),
                       '--output', str(prefix)]
            with Path(str(prefix) + '.log').open('w') as out:
                rc = run_exclusive(command, evidence=str(prefix) + '.gpu-processes.json', stdout=out, stderr=subprocess.STDOUT).returncode
            report.append(dict(name=prefix.name, command=command, returncode=rc))
            (output / 'summary.json').write_text(json.dumps(report, indent=2))
            print(prefix.name, rc, flush=True)
            if rc:
                raise RuntimeError(f'Cache-shape validation failed: {prefix}')


if __name__ == '__main__':
    main()

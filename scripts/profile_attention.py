"""Finite device-trace profiles of staged/fused attention; not latency evidence."""
import argparse
import itertools
import json
import subprocess
from pathlib import Path

from bench.gpu_guard import run_exclusive


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('results/attention-profiles'))
    parser.add_argument('--events', type=int, default=512)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    rows = []
    for backend, attention in itertools.product(['graph', 'persistent'], ['staged', 'fused']):
        prefix = args.output / f'{backend}-{attention}'
        command = ['nsys', 'profile', '--trace=cuda', '--cuda-graph-trace=node', '--sample=none',
                   '--cpuctxsw=none', '--force-overwrite=true', '--output', str(prefix),
                   'build-blackwell/stream-bench', '--config', 'configs/main.json',
                   '--weights', 'fixtures/main-fp32', '--backend', backend, '--blocks', '110',
                   '--attention', attention, '--matvec', 'scalar', '--mode', 'device',
                   '--events', str(args.events), '--output', str(prefix)]
        Path(str(prefix) + '.command.json').write_text(json.dumps(command, indent=2))
        with Path(str(prefix) + '.log').open('w') as out:
            result = run_exclusive(command, evidence=str(prefix) + '.gpu-processes.json',
                                   stdout=out, stderr=subprocess.STDOUT)
        rows.append(dict(command=command, returncode=result.returncode,
                         scope='Preloaded finite device trace, includes partial and full histories; profiling overhead applies'))
        (args.output / 'summary.json').write_text(json.dumps(rows, indent=2))
        if result.returncode:
            raise RuntimeError(f'Profile failed: {prefix}')
        with Path(str(prefix) + '.stats.txt').open('w') as out:
            subprocess.run(['nsys', 'stats', '--report', 'cuda_gpu_kern_sum', '--format', 'csv',
                            '--force-export=true', str(prefix) + '.nsys-rep'],
                           stdout=out, stderr=subprocess.STDOUT, check=True)
        print(prefix, 'PASS', flush=True)


if __name__ == '__main__':
    main()

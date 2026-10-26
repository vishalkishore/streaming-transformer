"""Run numerical gates, preserving stdout, errors and exact command lines."""
import argparse
import json
import subprocess
import sys
from pathlib import Path


def run(command, log):
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open('w') as out:
        out.write(json.dumps(command) + '\n')
        out.flush()
        result = subprocess.run(command, stdout=out, stderr=subprocess.STDOUT)
    return result.returncode


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--binary', default='build/stream-bench')
    p.add_argument('--sizes', nargs='+', default=['small', 'main'])
    p.add_argument('--output', default='results/validation')
    p.add_argument('--stress', type=int, default=0)
    p.add_argument('--backends', nargs='+', choices=['cpu', 'cuda', 'cublas', 'graph', 'cublas-graph', 'fused', 'persistent'],
                   default=['cpu', 'cuda', 'cublas', 'graph', 'cublas-graph', 'fused', 'persistent'])
    a = p.parse_args()
    report = []
    output = Path(a.output)
    for size in a.sizes:
        for mode in ['fp32', 'mixed']:
            fixture = Path(f'fixtures/{size}-{mode}')
            if not (fixture / 'expected/manifest.json').exists():
                subprocess.run([sys.executable, '-m', 'reference.export', '--config', f'configs/{size}.json',
                                '--mode', mode, '--output', str(fixture)], check=True)
            config = json.loads(Path(f'configs/{size}.json').read_text())
            count = 3 * config['window'] + 9
            for backend, blocks in [('cpu', 1), ('cuda', 4), ('cublas', 4), ('graph', 4),
                                    ('cublas-graph', 4), ('fused', 1), ('fused', 4),
                                    ('persistent', 1), ('persistent', 4)]:
                if backend not in a.backends:
                    continue
                name = f'{size}-{mode}-{backend}-{blocks}'
                command = [a.binary, '--config', f'configs/{size}.json', '--weights', str(fixture),
                           '--backend', backend, '--blocks', str(blocks), '--mode', 'correctness',
                           '--events', str(count), '--output', str(output / name)]
                rc = run(command, output / f'{name}.log')
                report.append(dict(name=name, returncode=rc, command=command))
                (output / 'summary.json').write_text(json.dumps(report, indent=2))
                print(name, 'PASS' if rc == 0 else f'FAIL({rc})', flush=True)
    if a.stress and any(row['returncode'] for row in report):
        print('Stress skipped: fixture correctness gate failed.', flush=True)
    elif a.stress:
        for mode in ['fp32', 'mixed']:
            for blocks in [1, 4]:
                name = f'main-{mode}-stress-{blocks}'
                command = [a.binary, '--config', 'configs/main.json', '--weights', f'fixtures/main-{mode}',
                           '--backend', 'persistent', '--blocks', str(blocks), '--mode', 'correctness',
                           '--expected', 'cpu', '--events', str(a.stress), '--reset-every', '10007',
                           '--output', str(output / name)]
                rc = run(command, output / f'{name}.log')
                report.append(dict(name=name, returncode=rc, command=command))
                (output / 'summary.json').write_text(json.dumps(report, indent=2))
                print(name, 'PASS' if rc == 0 else f'FAIL({rc})', flush=True)
    raise SystemExit(any(row['returncode'] for row in report))


if __name__ == '__main__':
    main()

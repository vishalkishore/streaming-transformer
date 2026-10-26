"""Run eager/compiled/explicit-graph correctness serially on the final GPU."""
import argparse
import itertools
import json
import subprocess
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--python', default='.venv-gpu/bin/python')
    p.add_argument('--output', type=Path, default=Path('results/torch-validation-final'))
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    report = []
    for size, mode, backend in itertools.product(['small', 'main', 'large'], ['fp32', 'mixed'], ['eager', 'compile', 'graph']):
        prefix = a.output / f'{size}-{mode}-{backend}'
        command = [a.python, '-m', 'bench.torch_bench', '--fixture', f'fixtures/{size}-{mode}',
                   '--backend', backend, '--correctness', '--warmup', '4', '--output', str(prefix)]
        with Path(str(prefix) + '.log').open('w') as out:
            rc = subprocess.run(command, stdout=out, stderr=subprocess.STDOUT).returncode
        report.append(dict(name=prefix.name, returncode=rc, command=command))
        (a.output / 'summary.json').write_text(json.dumps(report, indent=2))
        print(prefix.name, rc, flush=True)
    raise SystemExit(any(r['returncode'] for r in report))


if __name__ == '__main__':
    main()

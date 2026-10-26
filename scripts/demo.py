"""A bounded demonstration of independent CPU/graph/persistent stateful execution."""
import argparse
import json
import subprocess
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--binary', default='build-blackwell/stream-bench')
    p.add_argument('--output', type=Path, default=Path('results/demo'))
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    for backend, blocks in [('cpu', 1), ('cublas-graph', 4), ('persistent', 1), ('persistent', 4)]:
        prefix = a.output / f'{backend}-{blocks}'
        command = [a.binary, '--config', 'configs/small.json', '--weights', 'fixtures/small-fp32',
                   '--backend', backend, '--blocks', str(blocks), '--mode', 'correctness',
                   '--events', '201', '--output', str(prefix)]
        with Path(str(prefix) + '.log').open('w') as out:
            subprocess.run(command, stdout=out, stderr=subprocess.STDOUT, check=True)
        data = json.loads(Path(str(prefix) + '.json').read_text())
        print(f'{backend} blocks={blocks}: {data["events"]} events, wraparound and reset checked; '
              f'max score error={data["scores"]["max_abs"]:.3g}')
    print('Demonstration passed. Correctness timings are not performance comparisons.')


if __name__ == '__main__':
    main()

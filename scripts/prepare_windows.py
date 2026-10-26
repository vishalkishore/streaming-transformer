"""Prepare fixed-seed window variants before freezing performance-study sources."""
import itertools
import json
from pathlib import Path

from reference.export import export


def main():
    for size, window in itertools.product(['small', 'main', 'large'], [32, 64, 128, 256, 512]):
        variant = f'{size}-w{window}'
        config = json.loads(Path(f'configs/{size}.json').read_text())
        config['window'] = window
        path = Path(f'configs/{variant}.json')
        path.write_text(json.dumps(config, indent=2) + '\n')
        for mode in ['fp32', 'mixed']:
            output = Path(f'fixtures/{variant}-{mode}')
            if not (output / 'expected/manifest.json').exists():
                export(path, output, mode)
            baseline = json.loads(Path(f'fixtures/{size}-{mode}/manifest.json').read_text())
            actual = json.loads((output / 'manifest.json').read_text())
            if actual['sha256'] != baseline['sha256']:
                raise ValueError('Window sweep changed frozen model weights')
            print(variant, mode, 'prepared', flush=True)


if __name__ == '__main__':
    main()

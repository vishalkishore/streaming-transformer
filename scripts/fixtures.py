"""Regenerate the frozen fixture matrix after transferring a source-only archive."""
import argparse
import hashlib
import json
from pathlib import Path

from reference.export import export


def verify(path):
    checkpoint = json.loads((path / 'manifest.json').read_text())
    for directory in [path, path / 'expected']:
        manifest = json.loads((directory / 'manifest.json').read_text())
        raw = (directory / 'tensors.bin').read_bytes()
        if hashlib.sha256(raw).hexdigest() != manifest['sha256']:
            raise ValueError(f'Checksum mismatch: {directory}')
        for tensor in manifest['tensors']:
            part = raw[tensor['offset']:tensor['offset'] + tensor['nbytes']]
            if len(part) != tensor['nbytes'] or hashlib.sha256(part).hexdigest() != tensor['sha256']:
                raise ValueError(f'Tensor checksum mismatch: {directory}/{tensor["name"]}')
    expected = json.loads((path / 'expected/manifest.json').read_text())
    tensor = next(t for t in expected['tensors'] if t['name'] == 'events')
    trace = (path / 'trace.bin').read_bytes()
    if hashlib.sha256(trace).hexdigest() != tensor['sha256']:
        raise ValueError(f'Trace mismatch: {path}')
    if json.loads((path / 'resets.json').read_text()) != expected['resets']:
        raise ValueError(f'Reset schedule mismatch: {path}')
    return dict(checkpoint_sha256=checkpoint['sha256'], expected_sha256=expected['sha256'],
                trace_sha256=hashlib.sha256(trace).hexdigest())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sizes', choices=['small', 'main', 'large'], nargs='+', default=['small', 'main'])
    parser.add_argument('--modes', choices=['fp32', 'mixed'], nargs='+', default=['fp32', 'mixed'])
    parser.add_argument('--output', type=Path, default=Path('fixtures'))
    parser.add_argument('--verify-only', action='store_true')
    args = parser.parse_args()
    for size in args.sizes:
        for mode in args.modes:
            path = args.output / f'{size}-{mode}'
            if not args.verify_only:
                export(Path('configs') / f'{size}.json', path, mode)
            hashes = verify(path)
            print(path, json.dumps(hashes), flush=True)


if __name__ == '__main__':
    main()

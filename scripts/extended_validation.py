"""Independent full-sequence checks and bounded native adversarial input families."""
import argparse
import itertools
import json
import subprocess
from pathlib import Path

import numpy as np
import torch

from reference.export import traces
from reference.model import Config, Streaming, FullSequence, make_weights
from reference.numerical import POLICY, score_errors


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--binary', default='build-blackwell/stream-bench')
    p.add_argument('--output', type=Path, default=Path('results/extended-validation'))
    p.add_argument('--sizes', nargs='+', default=['small', 'main', 'large'])
    p.add_argument('--reference-only', action='store_true',
                   help='Run independent CPU references without starting a GPU client')
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    report = []
    for size, mode, kind in itertools.product(a.sizes, ['fp32', 'mixed'],
                                              ['random', 'zeros', 'constant', 'impulses', 'alternating', 'repeated']):
        c = Config(**json.loads(Path(f'configs/{size}.json').read_text()))
        events = traces(3 * c.window + 9, c.features, kind)
        w = make_weights(c, mode=mode)
        oracle = Streaming(c, w, mode)
        reset = 2 * c.window + 3
        scores = []
        for i, event in enumerate(events):
            if i == reset:
                oracle.reset()
            scores.append(oracle.step(torch.from_numpy(event)))
        stream = torch.stack(scores).numpy()
        # Independent masked sequences restart absolute positions at the same reset.
        full = torch.cat([FullSequence(c, w, mode).forward(torch.from_numpy(segment))
                          for segment in (events[:reset], events[reset:])]).numpy()
        diff = np.abs(stream - full)
        if mode == 'mixed':
            acceptance = score_errors(stream, full, c.layers)
        else:
            failed = int(np.count_nonzero(~np.isfinite(stream) | ~np.isfinite(full)
                                         | (diff > 1e-5 + 1e-4 * np.abs(full))))
            acceptance = dict(failed=failed, passed=failed == 0)
        bad = acceptance['failed']
        relative = float(np.linalg.norm(diff) / max(1e-30, np.linalg.norm(full)))
        row = dict(size=size, mode=mode, kind=kind, independent_reference_failures=bad,
                   max_score_abs=float(diff.max()), relative_l2=relative, native=[],
                   acceptance_policy=POLICY if mode == 'mixed' else 'fp32-scalar',
                   acceptance=acceptance, reset_events=[reset], reference_only=a.reference_only)
        if not acceptance['passed']:
            report.append(row)
            (a.output / 'summary.json').write_text(json.dumps(report, indent=2))
            raise RuntimeError(f'Independent reference failure: {size} {mode} {kind}')
        trace = a.output / f'{size}-{kind}.bin'
        events.tofile(trace)
        # Random fixture runs already test every backend and all state snapshots.
        if kind != 'random' and not a.reference_only:
            for backend, blocks in [('cuda', 4), ('cublas', 4), ('persistent', 1), ('persistent', 4)]:
                prefix = a.output / f'{size}-{mode}-{kind}-{backend}-{blocks}'
                command = [a.binary, '--config', f'configs/{size}.json', '--weights', f'fixtures/{size}-{mode}',
                           '--trace', str(trace), '--backend', backend, '--blocks', str(blocks),
                           '--mode', 'correctness', '--expected', 'cpu', '--events', str(len(events)),
                           '--reset-every', str(reset), '--output', str(prefix)]
                with Path(str(prefix) + '.log').open('w') as out:
                    rc = subprocess.run(command, stdout=out, stderr=subprocess.STDOUT).returncode
                row['native'].append(dict(command=command, returncode=rc))
                if rc:
                    report.append(row)
                    (a.output / 'summary.json').write_text(json.dumps(report, indent=2))
                    raise RuntimeError(f'Native adversarial failure: {prefix}')
        report.append(row)
        (a.output / 'summary.json').write_text(json.dumps(report, indent=2))
        print(size, mode, kind, 'PASS', flush=True)


if __name__ == '__main__':
    main()

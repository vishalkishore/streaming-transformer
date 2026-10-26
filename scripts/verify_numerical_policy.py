"""Test the frozen mixed policy against independent FP32/FP64 arithmetic and held-out seeds."""
import argparse
import itertools
import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from reference.export import traces
from reference.model import Config, Streaming, make_weights
from reference.numerical import POLICY, state_errors, score_errors


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, default=Path('results/numerical-policy-validation.json'))
    p.add_argument('--sizes', nargs='+', default=['small', 'main', 'large'])
    p.add_argument('--seeds', type=int, nargs='+', default=[42, 17, 314])
    p.add_argument('--kinds', nargs='+', default=['random', 'constant'])
    a = p.parse_args()
    torch.set_num_threads(1)
    report = dict(policy=POLICY, torch=torch.__version__, cases=[], status='running')
    a.output.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    for size, seed, kind in itertools.product(a.sizes, a.seeds, a.kinds):
        c = Config(**json.loads(Path(f'configs/{size}.json').read_text()))
        weights = make_weights(c, seed=seed, mode='mixed')
        fp32 = Streaming(c, weights, 'mixed', torch.float32)
        fp64 = Streaming(c, weights, 'mixed', torch.float64)
        events = traces(3 * c.window + 9, c.features, kind, seed=123 if seed == 42 else 987)
        summaries = {}
        scores32, scores64 = [], []
        for i, event in enumerate(events):
            if i == 2 * c.window + 3:
                fp32.reset()
                fp64.reset()
            scores32.append(float(fp32.step(torch.from_numpy(event))))
            scores64.append(float(fp64.step(torch.from_numpy(event).double())))
            for layer in range(c.layers):
                hidden = f'layers.{layer}.hidden'
                for name, actual, reference in [('hidden', fp32.intermediates[hidden], fp64.intermediates[hidden]),
                                                 ('keys', fp32.k[layer], fp64.k[layer]),
                                                 ('values', fp32.v[layer], fp64.v[layer])]:
                    result = state_errors(actual.numpy(), reference.numpy(), layer)
                    entry = summaries.setdefault(f'layers.{layer}.{name}', dict(max_abs=0., max_relative_l2=0., raw_failed=0,
                                                                               failed=0, failing_events=[]))
                    entry['max_abs'] = max(entry['max_abs'], result['max_abs'])
                    entry['max_relative_l2'] = max(entry['max_relative_l2'], result['relative_l2'])
                    entry['raw_failed'] += result['raw_failed']
                    entry['failed'] += result['failed']
                    if not result['passed'] and len(entry['failing_events']) < 8:
                        entry['failing_events'].append(dict(event=i, **result))
        s32, s64 = np.array(scores32), np.array(scores64)
        diff = np.abs(s32 - s64)
        score = score_errors(s32, s64, c.layers)
        passed = score['passed'] and all(not e['failing_events'] for e in summaries.values())
        report['cases'].append(dict(size=size, config=asdict(c), seed=seed, trace_seed=123 if seed == 42 else 987,
                                    kind=kind, events=len(events), states=summaries, scores=score, passed=passed))
        a.output.write_text(json.dumps(report, indent=2) + '\n')
        print(size, seed, kind, 'PASS' if passed else 'FAIL', flush=True)
    report['status'] = 'passed' if all(case['passed'] for case in report['cases']) else 'failed'
    report['wall_seconds'] = time.monotonic() - started
    a.output.write_text(json.dumps(report, indent=2) + '\n')
    raise SystemExit(report['status'] != 'passed')


if __name__ == '__main__':
    main()

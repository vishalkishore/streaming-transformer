"""Bounded regression for CPU-once export and CPU-free checked GPU replay."""
import argparse
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import torch

from reference.model import Config, Streaming, make_weights
from reference.numerical import score_errors, state_errors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('results/stored-oracle-regression'))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    report = []

    def run(command, label, should_pass=True):
        with (args.output / f'{label}.log').open('w') as log:
            rc = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT).returncode
        report.append(dict(name=label, command=command, returncode=rc, expected_success=should_pass))
        (args.output / 'summary.json').write_text(json.dumps(report, indent=2))
        if (rc == 0) != should_pass:
            raise RuntimeError(f'Unexpected regression outcome: {label}')

    events, reset, interval = 401, 137, 17
    for mode in ['fp32', 'mixed']:
        oracle = args.output / f'small-{mode}-oracle'
        run(['build-blackwell/stream-oracle', '--weights', f'fixtures/small-{mode}', '--events', str(events),
             '--reset-every', str(reset), '--snapshot-every', str(interval), '--output', str(oracle)], f'export-{mode}')
        manifest = json.loads((oracle / 'manifest.json').read_text())
        data = (oracle / 'tensors.bin').read_bytes()
        tensors = {t['name']: np.frombuffer(data, dtype='<f4', count=t['nbytes'] // 4, offset=t['offset'])
                   for t in manifest['tensors']}
        config = Config(**manifest['config'])
        independent = Streaming(config, make_weights(config, mode=mode), mode)
        trace = np.fromfile(f'fixtures/small-{mode}/trace.bin', dtype='<f4').reshape(-1, config.features)
        scores = []
        rows = {event: index for index, event in enumerate(manifest['state_events'])}
        for event in range(events):
            if event and event % reset == 0:
                independent.reset()
            scores.append(float(independent.step(torch.from_numpy(trace[event % len(trace)]))))
            if event in rows:
                row = rows[event]
                for name, reference in [('keys', independent.k), ('values', independent.v)]:
                    actual = tensors[name].reshape(len(rows), config.layers, config.heads, config.window, -1)[row]
                    for layer in range(config.layers):
                        assert state_errors(actual[layer], reference[layer].numpy(), layer, mode == 'mixed')['passed']
                actual = tensors['hidden'].reshape(len(rows), config.layers, config.width)[row]
                for layer in range(config.layers):
                    reference = independent.intermediates[f'layers.{layer}.hidden'].numpy()
                    assert state_errors(actual[layer], reference, layer, mode == 'mixed')['passed']
        if mode == 'mixed':
            assert score_errors(tensors['scores'], scores, config.layers)['passed']
        else:
            np.testing.assert_allclose(tensors['scores'], scores, atol=1e-5, rtol=1e-4)
        for backend, blocks in [('cublas-graph', 4), ('persistent', 1), ('persistent', 4)]:
            prefix = args.output / f'{mode}-{backend}-{blocks}'
            command = ['build-blackwell/stream-bench', '--config', 'configs/small.json', '--weights', f'fixtures/small-{mode}',
                       '--backend', backend, '--blocks', str(blocks), '--mode', 'correctness', '--expected', str(oracle),
                       '--events', str(events), '--reset-every', str(reset), '--output', str(prefix)]
            run(command, prefix.name)
            result = json.loads(Path(str(prefix) + '.json').read_text())
            assert not result['cpu_oracle_active'] and result['cpu_threads_actual'] == 0
            assert result['scores']['elements'] == events
            assert result['keys']['elements'] == len(rows) * config.layers * config.width * config.window
        for field in ['events', 'reset_every', 'state_events', 'sha256']:
            bad = args.output / f'bad-{mode}-{field}'
            bad.mkdir(exist_ok=True)
            shutil.copyfile(oracle / 'tensors.bin', bad / 'tensors.bin')
            changed = dict(manifest)
            if field == 'state_events':
                changed[field] = changed[field][:-1]
            elif field == 'sha256':
                changed[field] = '0' * 64
            else:
                changed[field] += 1
            (bad / 'manifest.json').write_text(json.dumps(changed))
            rejected = command.copy()
            rejected[rejected.index('--expected') + 1] = str(bad)
            rejected[rejected.index('--output') + 1] = str(args.output / f'rejected-{mode}-{field}')
            run(rejected, f'reject-{mode}-{field}', should_pass=False)
        print(mode, 'independent CPU agreement, three GPU replays and four rejection checks PASS', flush=True)


if __name__ == '__main__':
    main()

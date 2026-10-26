"""Compare native CPU operation snapshots to FP32/FP64 reference arithmetic.

This is a diagnostic run, not a benchmark or a replacement acceptance policy.
Both reference streams use the checkpoint's cast boundaries and reset schedule.
"""
import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
import torch

from reference.model import Config, Streaming


def load_weights(fixture, manifest):
    raw = (fixture / 'tensors.bin').read_bytes()
    if hashlib.sha256(raw).hexdigest() != manifest['sha256']:
        raise ValueError('Checkpoint checksum mismatch')
    return {
        t['name']: torch.from_numpy(np.frombuffer(
            raw, dtype=t['dtype'], count=t['nbytes'] // np.dtype(t['dtype']).itemsize,
            offset=t['offset']).copy().reshape(t['shape']))
        for t in manifest['tensors']
    }


def summarize(actual, expected, mixed):
    diff = np.abs(actual.astype(np.float64) - expected)
    bound = (2e-3 if mixed else 1e-5) + (2e-2 if mixed else 1e-4) * np.abs(expected)
    finite = np.isfinite(actual) & np.isfinite(expected)
    bad = ~finite | (diff > bound)
    examples = [dict(event=int(i), coordinate=int(j), actual=float(actual[i, j]),
                     expected=float(expected[i, j]), difference=float(diff[i, j]))
                for i, j in np.argwhere(bad)[:8]]
    return dict(elements=diff.size, failed=int(bad.sum()), nonfinite=int((~finite).sum()),
                max_abs=float(diff.max()), relative_l2=float(np.linalg.norm(diff) /
                max(1e-30, np.linalg.norm(expected))),
                absolute_quantiles=np.quantile(diff, [.5, .95, .99, .999, 1]).tolist(), examples=examples)


def diagnose(fixture, binary, output, backend='cpu'):
    torch.set_num_threads(1)
    manifest = json.loads((fixture / 'manifest.json').read_text())
    config = Config(**manifest['config'])
    weights = load_weights(fixture, manifest)
    trace_bytes = (fixture / 'trace.bin').read_bytes()
    events = np.frombuffer(trace_bytes, dtype='<f4').copy().reshape(-1, config.features)
    resets = set(json.loads((fixture / 'resets.json').read_text()))
    native = output / 'native'
    command = [str(binary), str(fixture), str(native), backend]
    subprocess.run(command, check=True)
    layout = json.loads((native / 'manifest.json').read_text())
    raw = (native / 'stages.bin').read_bytes()
    if hashlib.sha256(raw).hexdigest() != layout['sha256']:
        raise ValueError('Native snapshot checksum mismatch')
    if layout['checkpoint_sha256'] != manifest['sha256'] or layout['trace_sha256'] != hashlib.sha256(trace_bytes).hexdigest():
        raise ValueError('Native snapshot inputs differ from reference')
    actual = np.frombuffer(raw, dtype='<f4').reshape(len(events), layout['event_elements'])
    native_stages = {s['name']: actual[:, s['offset_elements']:s['offset_elements'] + s['elements']]
                     for s in layout['stages']}
    report = dict(command=command, backend=backend, checkpoint_sha256=manifest['sha256'],
                  trace_sha256=layout['trace_sha256'], compiler=layout['compiler'],
                  config=manifest['config'], mode=manifest['mode'], thresholds_changed=False,
                  timing='diagnostic operation snapshots; excluded from performance evidence', comparisons={})
    references = {}
    order = None
    for dtype in [torch.float32, torch.float64]:
        oracle = Streaming(config, weights, manifest['mode'], dtype)
        captured = {name: [] for name in native_stages}
        for i, event in enumerate(events):
            if i in resets:
                oracle.reset()
            oracle.step(torch.from_numpy(event).to(dtype))
            if order is None:
                order = list(oracle.intermediates)
            for name in captured:
                captured[name].append(oracle.intermediates[name].flatten().double().numpy())
        references[str(dtype)] = {name: np.stack(values) for name, values in captured.items()}
    mixed = manifest['mode'] == 'mixed'
    forced = {}
    for name, weight in weights.items():
        if weight.ndim != 2:
            continue
        operand = torch.from_numpy(native_stages[name + '.operand'].copy()).double()
        exact = (operand @ weight.double().T).numpy()
        actual_pre = native_stages[name + '.pre_cast'].astype(np.float64)
        # Standard conservative dot-product bound with separate multiply/add rounding.
        # The common native operands remove upstream rounding and cache drift.
        operations = 2 * operand.shape[1]
        unit_roundoff = np.finfo(np.float32).eps / 2
        gamma = operations * unit_roundoff / (1 - operations * unit_roundoff)
        bound = gamma * (operand.abs() @ weight.double().abs().T).numpy()
        diff = np.abs(actual_pre - exact)
        expected_cast = exact if name == 'score' or not mixed else exact.astype(np.float16).astype(np.float64)
        forced[name] = dict(max_pre_cast_abs=float(diff.max()),
                            fp32_dot_bound_violations=int((diff > bound).sum()),
                            max_fraction_of_dot_bound=float(np.max(diff / np.maximum(bound, 1e-30))),
                            output_cast_mismatches=int(np.count_nonzero(native_stages[name + '.linear'] != expected_cast)))
    report['isolated_native_linear_operations'] = forced
    nonlinear = {}
    for name, weight in weights.items():
        if weight.ndim != 1:
            continue
        operand = native_stages[name + '.operand'].astype(np.float64)
        exact = operand / np.sqrt(np.mean(operand * operand, axis=1, keepdims=True) + config.epsilon) * weight.double().numpy()
        nonlinear[name] = summarize(native_stages[name + '.pre_cast'], exact, False)
    for layer in range(config.layers):
        p = f'layers.{layer}.'
        gate, up = native_stages[p + 'gate.linear'].astype(np.float64), native_stages[p + 'up.linear'].astype(np.float64)
        activation = gate / (1 + np.exp(-gate)) * up
        if mixed:
            activation = activation.astype(np.float16).astype(np.float64)
        nonlinear[p + 'activation'] = summarize(native_stages[p + 'activation'], activation, mixed)
        # FP16 operands add exactly in FP32; residual output must round identically.
        prior = native_stages['input'] if layer == 0 else native_stages[f'layers.{layer-1}.hidden']
        for target, lhs, rhs in [('attention_residual', prior, native_stages[p + 'out.linear']),
                                  ('hidden', native_stages[p + 'attention_residual'], native_stages[p + 'down.linear'])]:
            exact = lhs.astype(np.float64) + rhs
            if mixed:
                exact = exact.astype(np.float16).astype(np.float64)
            nonlinear[p + target] = summarize(native_stages[p + target], exact, mixed)
    report['isolated_native_nonlinear_operations'] = nonlinear
    for label, lhs, rhs in [('native_vs_fp32', native_stages, references['torch.float32']),
                            ('native_vs_fp64', native_stages, references['torch.float64']),
                            ('fp32_vs_fp64', references['torch.float32'], references['torch.float64'])]:
        stages = {name: summarize(lhs[name], rhs[name], mixed) for name in order if name in lhs}
        first = None
        first_cast = None
        # Report execution order within the earliest divergent event, rather than map order.
        for i in range(len(events)):
            for name in stages:
                changed = np.flatnonzero(lhs[name][i] != rhs[name][i])
                if len(changed):
                    j = int(changed[0])
                    first = dict(event=i, tensor=name, coordinate=j,
                                 actual=float(lhs[name][i, j]), expected=float(rhs[name][i, j]))
                    break
            if first is not None:
                break
        for i in range(len(events)):
            for name in stages:
                if name.endswith('.pre_cast') or name.startswith('score.'):
                    continue
                changed = np.flatnonzero(lhs[name][i] != rhs[name][i])
                if len(changed):
                    j = int(changed[0])
                    first_cast = dict(event=i, tensor=name, coordinate=j,
                                      actual=float(lhs[name][i, j]), expected=float(rhs[name][i, j]))
                    break
            if first_cast is not None:
                break
        input_examples = []
        for i, j in np.argwhere(lhs['input.linear'] != rhs['input.linear'])[:8]:
            # Identical input and weights isolate this projection from cache/history drift.
            exact = float(events[i].astype(np.float64) @ weights['input'][j].double().numpy())
            rounded = np.float16(exact)
            low = rounded if float(rounded) <= exact else np.nextafter(rounded, np.float16(-np.inf))
            high = np.nextafter(low, np.float16(np.inf))
            midpoint = (float(low) + float(high)) / 2
            input_examples.append(dict(event=int(i), coordinate=int(j), fp64_dot=exact,
                                       fp16_midpoint=midpoint, distance_to_midpoint=abs(exact - midpoint),
                                       lhs_pre_cast=float(lhs['input.pre_cast'][i, j]),
                                       rhs_pre_cast=float(rhs['input.pre_cast'][i, j]),
                                       lhs_cast=float(lhs['input.linear'][i, j]),
                                       rhs_cast=float(rhs['input.linear'][i, j])))
        report['comparisons'][label] = dict(first_divergence=first, first_cast_divergence=first_cast,
                                           isolated_input_cast_examples=input_examples, stages=stages)
    (output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fixture', type=Path, default=Path('fixtures/main-mixed'))
    parser.add_argument('--binary', type=Path, default=Path('build-blackwell/stream-diagnose'))
    parser.add_argument('--output', type=Path, default=Path('results/blackwell-mixed-diagnosis'))
    parser.add_argument('--backend', choices=['cpu', 'cuda'], default='cpu')
    args = parser.parse_args()
    report = diagnose(args.fixture, args.binary, args.output, args.backend)
    for label, comparison in report['comparisons'].items():
        print(label, 'first divergence:', comparison['first_divergence'])
        for name, stats in comparison['stages'].items():
            if name.endswith('hidden'):
                print(name, 'failures:', stats['failed'], 'max absolute error:', stats['max_abs'])

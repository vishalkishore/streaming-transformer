"""Export reproducible weights and state-rich execution fixtures."""
import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path
import numpy as np
import torch
from reference.model import Config, Streaming, make_weights


def traces(count, features, kind, seed=123):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((count, features)).astype('<f4')
    if kind == 'zeros':
        x.fill(0)
    elif kind == 'constant':
        x.fill(0.25)
    elif kind == 'impulses':
        x.fill(0)
        x[::17, 0] = 1
    elif kind == 'alternating':
        x[:] = ((-1.) ** np.arange(count))[:, None]
    elif kind == 'repeated':
        x[:] = x[np.arange(count) % min(count, 7)]
    elif kind != 'random':
        raise ValueError(kind)
    return x


def write_tensors(path, tensors, extra):
    path.mkdir(parents=True, exist_ok=True)
    data = bytearray()
    manifest = dict(version=1, byte_order='little', orientation='out_in', **extra, tensors=[])
    for name, tensor in tensors.items():
        a = np.ascontiguousarray(tensor.detach().cpu().numpy())
        raw = a.tobytes()
        manifest['tensors'].append(dict(name=name, shape=list(a.shape), dtype=str(a.dtype),
                                        offset=len(data), nbytes=len(raw), sha256=hashlib.sha256(raw).hexdigest()))
        data.extend(raw)
    manifest['sha256'] = hashlib.sha256(data).hexdigest()
    (path / 'tensors.bin').write_bytes(data)
    (path / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')


def export(config, output, mode='fp32', count=None, kind='random', seed=42):
    c = Config(**json.loads(Path(config).read_text()))
    torch.set_num_threads(1)
    weights = make_weights(c, seed, mode)
    output = Path(output)
    write_tensors(output, weights, dict(config=asdict(c), mode=mode, seed=seed))
    events = traces(count or (3 * c.window + 9), c.features, kind)
    events.tofile(output / 'trace.bin')
    oracle = Streaming(c, weights, mode)
    tensors = {'events': torch.from_numpy(events)}
    scores, hidden, keys, values = [], [], [], []
    # Reset before the event at these indices, also covering a warm restart.
    resets = [2 * c.window + 3] if len(events) > 2 * c.window + 3 else []
    for i, event in enumerate(events):
        if i in resets:
            oracle.reset()
        scores.append(oracle.step(torch.from_numpy(event)))
        hidden.append(torch.stack([oracle.intermediates[f'layers.{l}.hidden'] for l in range(c.layers)]))
        keys.append(oracle.k.clone())
        values.append(oracle.v.clone())
    tensors.update(scores=torch.stack(scores), hidden=torch.stack(hidden),
                   keys=torch.stack(keys), values=torch.stack(values))
    write_tensors(output / 'expected', tensors, dict(config=asdict(c), mode=mode, resets=resets))
    (output / 'resets.json').write_text(json.dumps(resets))
    return output


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--config', default='configs/small.json')
    p.add_argument('--output', default='fixtures/small-fp32')
    p.add_argument('--mode', choices=['fp32', 'mixed'], default='fp32')
    p.add_argument('--events', type=int)
    p.add_argument('--kind', default='random')
    a = p.parse_args()
    export(a.config, a.output, a.mode, a.events, a.kind)

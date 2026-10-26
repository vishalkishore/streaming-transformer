"""Quantify reduction/cast sensitivity without changing acceptance thresholds."""
import json
from pathlib import Path
import numpy as np
import torch
from reference.model import Config, Streaming, make_weights
from reference.export import traces


def main():
    torch.set_num_threads(1)
    c = Config(**json.loads(Path('configs/main.json').read_text()))
    w = make_weights(c, mode='mixed')
    fp32, fp64 = Streaming(c, w, 'mixed'), Streaming(c, w, 'mixed', torch.float64)
    events = traces(3 * c.window + 9, c.features, 'random')
    first = None
    errors = []
    examples = []
    for event, x in enumerate(events):
        if event == 2 * c.window + 3:
            fp32.reset()
            fp64.reset()
        fp32.step(torch.from_numpy(x))
        fp64.step(torch.from_numpy(x).double())
        for name, actual in fp32.intermediates.items():
            ref = fp64.intermediates[name]
            diff = (actual.double() - ref).abs()
            if first is None and diff.max() > 0:
                first = dict(event=event, tensor=name, max_abs=float(diff.max()),
                             explanation='Same weights and cast boundaries; FP32 versus FP64 reduction/nonlinearity arithmetic')
            if name.endswith('hidden'):
                bound = 2e-3 + 2e-2 * ref.abs()
                rms = float(torch.sqrt((ref * ref).mean()))
                for index in torch.where(diff > bound)[0].tolist():
                    if len(examples) < 20:
                        examples.append(dict(event=event, tensor=name, index=index, actual=float(actual[index]),
                                             expected=float(ref[index]), diff=float(diff[index]), rms=rms))
                errors.extend(diff.tolist())
    data = dict(first_divergence=first, raw_hidden_failures=examples,
                max_hidden_abs=float(np.max(errors)), hidden_abs_quantiles=np.quantile(errors, [.5,.95,.99,.999,1]).tolist(),
                thresholds_changed=False)
    Path('results/mixed-reduction-diagnosis.json').write_text(json.dumps(data, indent=2))
    print(json.dumps(data, indent=2))


if __name__ == '__main__':
    main()

"""Check installed Inductor preserves explicit FP16 rounding on a CPU witness.

This exercises compiler configuration without initializing a GPU. The full CUDA
model matrix remains necessary; this witness does not establish GPU correctness.
"""
import argparse
import json
import os
from pathlib import Path

# Inductor may initialize CUDA while choosing compiler capabilities even for CPU
# tensors. Hide GPUs in this standalone witness before importing the framework.
os.environ['CUDA_VISIBLE_DEVICES'] = ''

import torch

from bench.torch_bench import configure_numerics


def rounded_difference(x):
    rounded = (x + .0003).half().float()
    return rounded - x


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('results/torch-cast-witness.json'))
    args = parser.parse_args()
    configure_numerics()
    torch.set_num_threads(1)
    x = torch.tensor([1., -1., .5, -.5, 2., -2.])
    expected = rounded_difference(x)
    unrounded = (x + .0003) - x
    actual = torch.compile(rounded_difference, fullgraph=True, backend='inductor')(x)
    arithmetic_passed = torch.equal(actual, expected) and not torch.equal(expected, unrounded)
    passed = arithmetic_passed and not torch.cuda.is_initialized()
    report = dict(torch=torch.__version__, device='cpu', passed=passed,
                  arithmetic_passed=arithmetic_passed, cuda_visible_devices=os.environ['CUDA_VISIBLE_DEVICES'],
                  emulate_precision_casts=torch._inductor.config.emulate_precision_casts,
                  expected=expected.tolist(), actual=actual.tolist(), unrounded=unrounded.tolist(),
                  gpu_initialized=torch.cuda.is_initialized(),
                  scope='CPU compiler cast witness only; CUDA model validation is separate')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report))
    raise SystemExit(not passed or report['gpu_initialized'])


if __name__ == '__main__':
    main()

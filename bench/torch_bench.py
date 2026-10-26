"""Matched stateful PyTorch eager/Inductor/explicit-graph host-service baseline."""
import argparse
import csv
import hashlib
import json
import os
import resource
import time
from pathlib import Path

import numpy as np
import torch

from reference.numerical import POLICY
from bench.provenance import python_provenance


class Model:
    def __init__(self, fixture):
        self.manifest = json.loads((fixture / 'manifest.json').read_text())
        self.c = self.manifest['config']
        self.mixed = self.manifest['mode'] == 'mixed'
        raw = (fixture / 'tensors.bin').read_bytes()
        if hashlib.sha256(raw).hexdigest() != self.manifest['sha256']:
            raise ValueError('Checkpoint checksum mismatch')
        self.w = {}
        for t in self.manifest['tensors']:
            a = np.frombuffer(raw, dtype=t['dtype'], offset=t['offset'],
                              count=int(np.prod(t['shape']))).copy().reshape(t['shape'])
            self.w[t['name']] = torch.from_numpy(a).to('cuda')
        # Input remains FP32. Native cuBLAS has the same additional input-weight view.
        self.input_weight = self.w['input'].float()
        c = self.c
        shape = (c['layers'], c['heads'], c['window'], c['width'] // c['heads'])
        self.k = torch.zeros(shape, device='cuda', dtype=torch.float16 if self.mixed else torch.float32)
        self.v = torch.zeros_like(self.k)
        self.hidden = torch.zeros((c['layers'], c['width']), device='cuda')
        self.position = torch.zeros((), dtype=torch.int64, device='cuda')
        self.slots = torch.arange(c['window'], device='cuda')
        d = c['width'] // c['heads']
        # Frequencies follow the oracle's FP32 CPU construction, then remain fixed.
        self.freq = (10000. ** (-torch.arange(0, d, 2, dtype=torch.float32) / d)).cuda()

    def reset(self):
        self.k.zero_()
        self.v.zero_()
        self.hidden.zero_()
        self.position.zero_()

    def cast(self, x):
        return x.half().float() if self.mixed else x

    def linear(self, x, name, final=False):
        if name == 'input':
            result = torch.mv(self.input_weight, x)
        elif self.mixed:
            result = torch.mm(self.w[name], x.half()[:, None], out_dtype=torch.float32).flatten()
        else:
            result = torch.mv(self.w[name], x)
        return result if final else self.cast(result)

    def norm(self, x, name):
        return self.cast(x * torch.rsqrt((x * x).mean() + self.c['epsilon']) * self.w[name])

    def rope(self, x):
        c = self.c
        x = x.reshape(c['heads'], -1)
        angle = self.position.float() * self.freq
        a, b = x[:, 0::2], x[:, 1::2]
        return self.cast(torch.stack((a * angle.cos() - b * angle.sin(),
                                      a * angle.sin() + b * angle.cos()), -1).flatten(-2))

    def step(self, event):
        c = self.c
        slot = (self.position % c['window']).reshape(1)
        valid = torch.minimum(self.position + 1, torch.full_like(self.position, c['window']))
        x = self.linear(event, 'input')
        for layer in range(c['layers']):
            p = f'layers.{layer}.'
            z = self.norm(x, p + 'norm1')
            q = self.rope(self.linear(z, p + 'q'))
            k = self.rope(self.linear(z, p + 'k'))
            v = self.linear(z, p + 'v').reshape(c['heads'], -1)
            self.k[layer].index_copy_(1, slot, k[:, None].to(self.k.dtype))
            self.v[layer].index_copy_(1, slot, v[:, None].to(self.v.dtype))
            logits = (self.k[layer].float() * q[:, None]).sum(-1) / (c['width'] // c['heads']) ** .5
            probabilities = logits.masked_fill(self.slots >= valid, -torch.inf).softmax(-1)
            attention = self.cast((probabilities[..., None] * self.v[layer].float()).sum(1).flatten())
            x = self.cast(x + self.linear(attention, p + 'out'))
            z = self.norm(x, p + 'norm2')
            gate, up = self.linear(z, p + 'gate'), self.linear(z, p + 'up')
            activation = self.cast((gate / (1 + torch.exp(-gate))) * up)
            x = self.cast(x + self.linear(activation, p + 'down'))
            self.hidden[layer].copy_(x)
        score = self.linear(self.norm(x, 'final_norm'), 'score', final=True)
        self.position.add_(1)
        return score


def distribution(values):
    values = np.sort(values)
    return {name: float(values[min(len(values) - 1, int(np.ceil(q * len(values))) - 1)])
            for name, q in [('p50_us', .5), ('p95_us', .95), ('p99_us', .99),
                            ('p999_us_diagnostic', .999), ('worst_us', 1)]}


def configure_numerics():
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    torch.backends.cuda.matmul.allow_fp16_accumulation = False
    torch.set_float32_matmul_precision('highest')
    torch._inductor.config.triton.cudagraphs = False
    torch._inductor.config.use_fast_math = False
    # Inductor otherwise elides half->float round trips during fusion. Those
    # round trips are model cast boundaries, so their rounding must be retained.
    torch._inductor.config.emulate_precision_casts = True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fixture', type=Path, default=Path('fixtures/main-fp32'))
    parser.add_argument('--backend', choices=['eager', 'compile', 'graph'], default='graph')
    parser.add_argument('--events', type=int, default=100000)
    parser.add_argument('--warmup', type=int, default=1000)
    parser.add_argument('--output', type=Path, default=Path('results/torch'))
    parser.add_argument('--worker-core', type=int, default=-1)
    parser.add_argument('--correctness', action='store_true')
    parser.add_argument('--profile', action='store_true')
    args = parser.parse_args()
    if args.worker_core >= 0:
        os.sched_setaffinity(0, {args.worker_core})
    torch.set_num_threads(1)
    configure_numerics()
    startup = time.perf_counter()
    model = Model(args.fixture)
    trace_bytes = (args.fixture / 'trace.bin').read_bytes()
    events = torch.from_numpy(np.frombuffer(trace_bytes, dtype='<f4').copy().reshape(-1, model.c['features'])).pin_memory()
    d_input = torch.zeros(model.c['features'], device='cuda')
    h_output = torch.empty(1, pin_memory=True)
    output_view = h_output.numpy()
    step = torch.compile(model.step, fullgraph=True, backend='inductor') if args.backend == 'compile' else model.step
    with torch.inference_mode():
        step(d_input)
        torch.cuda.synchronize()
        if args.backend == 'graph':
            side = torch.cuda.Stream()
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):
                for _ in range(3):
                    step(d_input)
            torch.cuda.current_stream().wait_stream(side)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                result = step(d_input)
            def compute():
                graph.replay()
                return result
        else:
            def compute():
                return step(d_input)
        for _ in range(args.warmup):
            compute()
        model.reset()
        torch.cuda.synchronize()
        startup_ms = (time.perf_counter() - startup) * 1000
        if args.profile:
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]) as profile:
                compute()
                torch.cuda.synchronize()
            args.output.parent.mkdir(parents=True, exist_ok=True)
            profile.export_chrome_trace(str(args.output) + '.profile.json')
            model.reset()
            torch.cuda.synchronize()
        resets = set(json.loads((args.fixture / 'resets.json').read_text())) if args.correctness else set()
        expected = {}
        if args.correctness:
            manifest = json.loads((args.fixture / 'expected/manifest.json').read_text())
            for t in manifest['tensors']:
                expected[t['name']] = np.memmap(args.fixture / 'expected/tensors.bin', dtype='<f4', mode='r',
                                               offset=t['offset'], shape=tuple(t['shape']))
            args.events = len(events)
        failures = dict(scores=0, hidden=0, hidden_raw=0, keys=0, values=0, normwise=0)
        score_squared_error, score_squared_reference = 0., 0.
        samples = []
        start_wall, start_cpu = time.perf_counter(), time.process_time()
        for i in range(args.events):
            if i in resets:
                model.reset()
                torch.cuda.synchronize()
            event = events[i % len(events)]
            submit = time.perf_counter_ns()
            d_input.copy_(event, non_blocking=True)
            score = compute()
            h_output.copy_(score, non_blocking=True)
            torch.cuda.current_stream().synchronize()
            complete = time.perf_counter_ns()
            value = float(output_view[0])
            samples.append((i, submit, complete, value))
            if args.correctness:
                for name, actual in [('scores', np.array(value)), ('hidden', model.hidden.cpu().numpy()),
                                     ('keys', model.k.float().cpu().numpy()), ('values', model.v.float().cpu().numpy())]:
                    reference = expected[name][i]
                    diff = np.abs(actual - reference)
                    atol, rtol = (2e-3, 2e-2) if model.mixed else (1e-5, 1e-4)
                    scale = model.c['layers'] if name == 'scores' else 1
                    if name in ['hidden', 'keys', 'values'] and model.mixed:
                        if name == 'hidden':
                            failures['hidden_raw'] += int(np.count_nonzero(diff > atol + rtol * np.abs(reference)))
                        depth_shape = (model.c['layers'],) + (1,) * (reference.ndim - 1)
                        depth = np.arange(1, model.c['layers'] + 1).reshape(depth_shape)
                        scale = depth * np.maximum(1, np.sqrt(np.mean(reference.astype('f8') ** 2, axis=-1, keepdims=True)))
                        # The norm gate covers a complete layer tensor. Short heads
                        # with small norms are still retained in native diagnostics.
                        axes = tuple(range(1, reference.ndim))
                        relative = np.sqrt(np.sum(diff.astype('f8') ** 2, axis=axes)) / np.maximum(
                            1e-30, np.sqrt(np.sum(reference.astype('f8') ** 2, axis=axes)))
                        failures['normwise'] += int(np.count_nonzero(relative > .002))
                    if name == 'scores':
                        score_squared_error += float(diff * diff)
                        score_squared_reference += float(reference * reference)
                    failures[name] += int(np.count_nonzero(~np.isfinite(actual) | (diff > atol * scale + rtol * np.abs(reference))))
        wall, cpu = time.perf_counter() - start_wall, time.process_time() - start_cpu
        normalized_score_rmse = np.sqrt(score_squared_error / max(1, args.events)) / max(
            1., np.sqrt(score_squared_reference / max(1, args.events)))
        if model.mixed and normalized_score_rmse > .002:
            failures['normwise'] += 1
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with Path(str(args.output) + '.csv').open('w') as file:
        writer = csv.writer(file)
        writer.writerow(['event', 'submit_ns', 'complete_ns', 'score'])
        writer.writerows(samples)
    metadata = dict(backend='torch-' + args.backend, torch=torch.__version__, cuda=torch.version.cuda,
                    provenance=python_provenance(),
                    acceptance_policy=POLICY,
                    correctness_checked=args.correctness,
                    script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    gpu=torch.cuda.get_device_name(), math_backend='torch.mv FP32; mixed torch.mm FP16 weights/inputs with FP32 output/accumulation; Inductor compilation recorded by profiler',
                    compile_backend='inductor' if args.backend == 'compile' else None,
                    tf32=False, automatic_compile_cudagraphs=False, config=model.c,
                    fp16_reduced_precision_reduction=False, fp16_accumulation=False,
                    inductor_emulate_precision_casts=True, inductor_use_fast_math=False,
                    precision=model.manifest['mode'], checkpoint_sha256=model.manifest['sha256'],
                    trace_sha256=hashlib.sha256(trace_bytes).hexdigest(), events=args.events,
                    warmup=args.warmup, startup_ms=startup_ms, failures=failures,
                    normalized_score_rmse=normalized_score_rmse,
                    service=distribution(np.array([(s[2] - s[1]) / 1000 for s in samples])),
                    wall_seconds=wall, cpu_seconds=cpu, cpu_core_equivalents=cpu / wall,
                    service_rate_per_second=args.events / wall,
                    boundary='pinned CPU input ready to pinned CPU output visible; H2D, complete streaming step, D2H, stream synchronization',
                    scratch='PyTorch caching allocator in eager/compile; graph private pool allocated during capture',
                    torch_config=torch.__config__.show(), worker_core=args.worker_core,
                    peak_cuda_bytes=torch.cuda.max_memory_allocated())
    Path(str(args.output) + '.json').write_text(json.dumps(metadata, indent=2) + '\n')
    print(json.dumps(metadata, indent=2))
    raise SystemExit(any(v for k, v in failures.items() if k != 'hidden_raw'))


if __name__ == '__main__':
    main()

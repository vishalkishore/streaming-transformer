"""Auditable local-causal transformer. Linear weights are [output, input].

mixed: weights, projection outputs, norm outputs, RoPE, attention outputs,
SwiGLU outputs and residuals round to FP16. Reductions/dots/nonlinearities
use FP32; the final score does not round. Input remains FP32 until projection.
"""
import math
from dataclasses import dataclass, asdict
import torch


@dataclass
class Config:
    layers: int = 2
    width: int = 128
    heads: int = 4
    ffn: int = 256
    window: int = 256
    features: int = 32
    epsilon: float = 1e-5

    def validate(self):
        assert self.layers > 0 and self.window > 0 and self.ffn > 0
        assert self.width % self.heads == 0 and (self.width // self.heads) % 2 == 0


def make_weights(c, seed=42, mode="fp32"):
    c.validate()
    gen = torch.Generator().manual_seed(seed)
    weights = {}

    def matrix(name, rows, cols):
        weights[name] = torch.randn(rows, cols, generator=gen) / math.sqrt(cols)

    matrix("input", c.width, c.features)
    for layer in range(c.layers):
        prefix = f"layers.{layer}."
        for name in ("norm1", "norm2"):
            weights[prefix + name] = torch.ones(c.width)
        for name in ("q", "k", "v", "out"):
            matrix(prefix + name, c.width, c.width)
        for name in ("gate", "up"):
            matrix(prefix + name, c.ffn, c.width)
        matrix(prefix + "down", c.width, c.ffn)
    weights["final_norm"] = torch.ones(c.width)
    matrix("score", 1, c.width)
    return {k: v.half() if mode == "mixed" else v for k, v in weights.items()}


class Math:
    def __init__(self, config, weights, mode="fp32", dtype=torch.float32):
        self.c = config
        self.w = weights
        self.mode = mode
        self.dtype = dtype

    def cast(self, x):
        return x.half().to(self.dtype) if self.mode == "mixed" else x.to(self.dtype)

    def record(self, name, x):
        if hasattr(self, "intermediates"):
            self.intermediates[name] = x.clone()

    def linear(self, x, name, final=False):
        self.record(name + ".operand", x)
        result = x.to(self.dtype) @ self.w[name].to(self.dtype).T
        self.record(name + ".pre_cast", result)
        output = result if final else self.cast(result)
        self.record(name + ".linear", output)
        return output

    def norm(self, x, name):
        self.record(name + ".operand", x)
        result = (x * torch.rsqrt((x * x).mean(-1, keepdim=True) + self.c.epsilon)
                  * self.w[name].to(self.dtype))
        self.record(name + ".pre_cast", result)
        output = self.cast(result)
        self.record(name + ".norm", output)
        return output

    def rope(self, x, positions):
        d = self.c.width // self.c.heads
        x = x.reshape(*x.shape[:-1], self.c.heads, d)
        freq = 10000.0 ** (-torch.arange(0, d, 2, dtype=self.dtype) / d)
        angle = positions.to(self.dtype)[..., None, None] * freq
        a, b = x[..., 0::2], x[..., 1::2]
        return self.cast(torch.stack((a * angle.cos() - b * angle.sin(),
                                      a * angle.sin() + b * angle.cos()), -1).flatten(-2))

    def ffn(self, x, prefix):
        gate = self.linear(x, prefix + "gate")
        up = self.linear(x, prefix + "up")
        activation = self.cast((gate / (1 + torch.exp(-gate))) * up)
        self.record(prefix + "activation", activation)
        return self.linear(activation, prefix + "down")


class Streaming(Math):
    def __init__(self, config, weights, mode="fp32", dtype=torch.float32):
        super().__init__(config, weights, mode, dtype)
        self.reset()

    def reset(self):
        c = self.c
        self.position = 0
        self.k = torch.zeros(c.layers, c.heads, c.window, c.width // c.heads, dtype=self.dtype)
        self.v = torch.zeros_like(self.k)
        self.intermediates = {}

    def step(self, event):
        c, t = self.c, self.position
        slot = t % c.window
        self.intermediates = {}
        x = self.linear(event, "input")
        self.record("input", x)
        for layer in range(c.layers):
            p = f"layers.{layer}."
            z = self.norm(x, p + "norm1")
            q = self.rope(self.linear(z, p + "q"), torch.tensor(t))
            k = self.rope(self.linear(z, p + "k"), torch.tensor(t))
            v = self.linear(z, p + "v").reshape(c.heads, -1)
            for name, value in (("q", q), ("k", k), ("v", v)):
                self.record(p + name, value)
            self.k[layer, :, slot] = k
            self.v[layer, :, slot] = v
            slots = torch.arange(max(0, t - c.window + 1), t + 1) % c.window
            keys = self.k[layer, :, slots]
            values = self.v[layer, :, slots]
            logits = (keys * q[:, None]).sum(-1) / math.sqrt(c.width // c.heads)
            attention = self.cast((logits.softmax(-1)[..., None] * values).sum(1).flatten())
            self.record(p + "attention", attention)
            x = self.cast(x + self.linear(attention, p + "out"))
            self.record(p + "attention_residual", x)
            x = self.cast(x + self.ffn(self.norm(x, p + "norm2"), p))
            self.record(p + "hidden", x)
        self.position += 1
        return self.linear(self.norm(x, "final_norm"), "score", final=True).squeeze()


class FullSequence(Math):
    """Independent batched implementation; each layer uses a local causal mask."""
    def forward(self, events):
        c = self.c
        count = len(events)
        positions = torch.arange(count)
        allowed = ((positions[:, None] >= positions[None, :]) &
                   (positions[:, None] - positions[None, :] < c.window))
        x = self.linear(events, "input")
        for layer in range(c.layers):
            p = f"layers.{layer}."
            z = self.norm(x, p + "norm1")
            q = self.rope(self.linear(z, p + "q"), positions).transpose(0, 1)
            k = self.rope(self.linear(z, p + "k"), positions).transpose(0, 1)
            v = self.linear(z, p + "v").reshape(count, c.heads, -1).transpose(0, 1)
            logits = (q @ k.transpose(-1, -2)) / math.sqrt(c.width // c.heads)
            attention = self.cast((logits.masked_fill(~allowed, -torch.inf).softmax(-1) @ v)
                                  .transpose(0, 1).reshape(count, c.width))
            x = self.cast(x + self.linear(attention, p + "out"))
            x = self.cast(x + self.ffn(self.norm(x, p + "norm2"), p))
        return self.linear(self.norm(x, "final_norm"), "score", final=True).flatten()

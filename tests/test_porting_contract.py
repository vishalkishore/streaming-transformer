"""Guard model differences found while reviewing external inference kernels."""
import math

import pytest
import torch

from reference.model import Config, Math, Streaming, make_weights


def test_rope_uses_adjacent_pairs():
    config = Config(layers=1, width=8, heads=1, ffn=16, window=5)
    operations = Math(config, make_weights(config))
    vector = torch.arange(1, 9, dtype=torch.float32)
    expected = []
    for pair in range(4):
        angle = 7 * 10000.0 ** (-2 * pair / 8)
        a, b = float(vector[2 * pair]), float(vector[2 * pair + 1])
        expected.extend((a * math.cos(angle) - b * math.sin(angle),
                         a * math.sin(angle) + b * math.cos(angle)))
    torch.testing.assert_close(operations.rope(vector, torch.tensor(7)).flatten(),
                               torch.tensor(expected), atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize("mode", ["fp32", "mixed"])
@pytest.mark.parametrize("window", [1, 5])
def test_uniform_attention_consumes_all_retained_values(mode, window):
    # Zero Q/K gives an independent arithmetic-mean oracle for the visible past.
    # A decode mask exposing only the first key fails as soon as history grows.
    torch.set_num_threads(1)
    config = Config(layers=2, width=8, heads=2, ffn=16, window=window)
    weights = make_weights(config, mode=mode)
    for layer in range(config.layers):
        for name in ("q", "k"):
            weights[f"layers.{layer}.{name}"].zero_()
    stream = Streaming(config, weights, mode)
    generator = torch.Generator().manual_seed(173)
    values = [[] for _ in range(config.layers)]
    for event in range(4 * window + 3):
        if event == 2 * window + 1:
            stream.reset()
            values = [[] for _ in range(config.layers)]
        stream.step(torch.randn(config.features, generator=generator))
        for layer in range(config.layers):
            prefix = f"layers.{layer}."
            values[layer].append(stream.intermediates[prefix + "v"])
            expected = torch.stack(values[layer][-window:]).double().mean(0).flatten().float()
            if mode == "mixed":
                expected = expected.half().float()
            torch.testing.assert_close(stream.intermediates[prefix + "attention"], expected,
                                       atol=2e-3 if mode == "mixed" else 1e-6,
                                       rtol=2e-3 if mode == "mixed" else 1e-6)

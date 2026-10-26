import pytest
import torch
from reference.model import Config, Streaming, FullSequence, make_weights
from reference.export import traces


@pytest.mark.parametrize('mode', ['fp32', 'mixed'])
@pytest.mark.parametrize('kind', ['random', 'zeros', 'constant', 'impulses', 'alternating', 'repeated'])
def test_independent_references(mode, kind):
    torch.set_num_threads(1)
    c = Config(layers=2, width=16, heads=2, ffn=32, window=7)
    w = make_weights(c, mode=mode)
    events = torch.from_numpy(traces(44, c.features, kind))
    stream = Streaming(c, w, mode)
    actual = torch.stack([stream.step(e) for e in events])
    expected = FullSequence(c, w, mode).forward(events)
    torch.testing.assert_close(actual, expected, atol=2e-3 if mode == 'mixed' else 1e-5,
                               rtol=2e-2 if mode == 'mixed' else 1e-4)
    stream.reset()
    torch.testing.assert_close(stream.step(events[0]), actual[0])
    assert stream.position == 1


def test_fp64_and_cache_history():
    torch.set_num_threads(1)
    c = Config(layers=2, width=16, heads=2, ffn=32, window=5)
    w = make_weights(c)
    events = torch.from_numpy(traces(40, c.features, 'random'))
    s = Streaming(c, w)
    reference = FullSequence(c, w, dtype=torch.float64).forward(events.double())
    actual = torch.stack([s.step(e) for e in events])
    torch.testing.assert_close(actual.double(), reference, atol=1e-5, rtol=1e-4)
    # Recomputing raw windows loses higher-layer history and must not replace streaming.
    naive = FullSequence(c, w).forward(events[-c.window:])[-1]
    assert abs(naive - actual[-1]) > 1e-4

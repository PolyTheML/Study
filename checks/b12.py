"""Block 12: autoregressive generation (greedy, temperature, top-k)."""
import math

from ._util import checkpoint, expect


def _models():
    import torch

    class CycleModel(torch.nn.Module):
        """Fake LM: confidently predicts (token + 1) % V at every position. Rejects contexts longer than max_len."""

        def __init__(self, V=7, max_len=5):
            super().__init__()
            self.V, self.max_len = V, max_len

        def forward(self, idx):
            assert idx.shape[1] <= self.max_len, "generate fed the model more than model.max_len tokens"
            logits = torch.full((*idx.shape, self.V), -10.0)
            logits.scatter_(2, ((idx + 1) % self.V).unsqueeze(-1), 10.0)
            return logits

    class FixedModel(torch.nn.Module):
        """Fake LM: the last position always has logits [0, ln 3], i.e. probabilities [0.25, 0.75].
        Earlier positions point the other way, to catch reading the wrong position."""

        def __init__(self):
            super().__init__()
            self.max_len = 1000

        def forward(self, idx):
            B, T = idx.shape
            logits = torch.zeros(B, T, 2)
            logits[:, :, 0] = 5.0
            logits[:, -1, 0] = 0.0
            logits[:, -1, 1] = math.log(3.0)
            return logits

    return CycleModel(), FixedModel()


@checkpoint("12: generate")
def check_generate(generate):
    import torch

    cyc, fixed = _models()
    idx = torch.tensor([[0], [3]])
    out = generate(cyc, idx, max_new_tokens=8, greedy=True)
    expect(tuple(out.shape) == (2, 9), "returns (B, T + max_new_tokens)")
    expect(torch.equal(out[:, :1], idx), "the prompt is kept at the start")
    expect(out[0].tolist() == [0, 1, 2, 3, 4, 5, 6, 0, 1] and out[1].tolist() == [3, 4, 5, 6, 0, 1, 2, 3, 4],
           "greedy decoding follows the model; the context is cropped to model.max_len",
           "feed at most the last model.max_len tokens, and read the logits at the LAST position")

    g = torch.Generator().manual_seed(0)
    out_k1 = generate(cyc, idx, max_new_tokens=8, top_k=1, generator=g)
    expect(torch.equal(out_k1, out), "top_k = 1 sampling is the same as greedy")

    def freq_of_1(temperature, top_k=None):
        g = torch.Generator().manual_seed(1)
        x = torch.zeros(4000, 2, dtype=torch.long)
        o = generate(fixed, x, max_new_tokens=1, temperature=temperature, top_k=top_k, generator=g)
        return o[:, -1].float().mean().item()

    f1 = freq_of_1(1.0)
    expect(abs(f1 - 0.75) < 0.03, f"temperature 1 samples token 1 about 75% of the time (got {f1:.3f})",
           "sample from softmax(logits at the last position); torch.multinomial takes probabilities")
    f05 = freq_of_1(0.5)
    expect(abs(f05 - 0.9) < 0.03, f"temperature 0.5 sharpens: probabilities proportional to [1, 9], so 90% (got {f05:.3f})",
           "divide the logits by the temperature BEFORE the softmax")
    expect(freq_of_1(1.0, top_k=1) == 1.0, "top_k = 1 always keeps only the most likely token")

    seen = []

    class Probe(torch.nn.Module):
        max_len = 10

        def forward(self, idx):
            seen.append(torch.is_grad_enabled())
            return torch.zeros(*idx.shape, 3)

    generate(Probe(), torch.zeros(1, 1, dtype=torch.long), max_new_tokens=2, greedy=True)
    expect(seen and not any(seen), "runs without building a gradient graph", "wrap generation in torch.no_grad()")

"""Block 11: batching, the Eq. 3 learning-rate schedule, label smoothing (§5.4)."""
import math

from ._util import checkpoint, expect, fail


@checkpoint("11: get_batch")
def check_get_batch(get_batch):
    import torch

    data = torch.arange(100)
    g = torch.Generator().manual_seed(0)
    x, y = get_batch(data, block_size=8, batch_size=4, generator=g)
    expect(tuple(x.shape) == (4, 8) and tuple(y.shape) == (4, 8), "x and y have shape (batch_size, block_size)")
    expect(torch.equal(y, x + 1), "y is x shifted one step to the left (next-token targets)")
    starts = set()
    for _ in range(3000):
        xb, _ = get_batch(data, block_size=8, batch_size=4, generator=g)
        starts.update(xb[:, 0].tolist())
    expect(min(starts) == 0 and max(starts) == 91,
           "every valid window can be sampled: starts range over 0 .. len(data) - block_size - 1",
           "torch.randint's upper bound is exclusive; the last window's target is data[-1]")


@checkpoint("11: noam_lr (Eq. 3)")
def check_noam(noam_lr):
    d, w = 512, 4000
    expect(abs(noam_lr(4000, d, w) - 6.98771243e-4) < 1e-10,
           "peak at step = warmup_steps: about 7.0e-4 for the paper's base config")
    expect(abs(noam_lr(2000, d, w) / noam_lr(1000, d, w) - 2.0) < 1e-9, "linear increase during warmup")
    expect(abs(noam_lr(16000, d, w) / noam_lr(4000, d, w) - 0.5) < 1e-9,
           "after warmup, decays like 1/sqrt(step): 4x the steps gives half the rate")
    expect(noam_lr(3999, d, w) < noam_lr(4000, d, w) > noam_lr(4001, d, w), "the maximum sits exactly at warmup")


@checkpoint("11: label_smoothed_ce")
def check_label_smoothing(label_smoothed_ce):
    import torch
    import torch.nn.functional as F

    logits = torch.tensor([[0.0, math.log(3.0)]])
    t = torch.tensor([1])
    expect(abs(float(label_smoothed_ce(logits, t, eps=0.0)) - 0.28768207) < 1e-6,
           "eps = 0 is ordinary cross-entropy: -ln 0.75")
    val = float(label_smoothed_ce(logits, t, eps=0.2))
    if abs(val - 0.50740453) < 1e-6:
        fail("you used the eps / (V - 1) convention",
             "this repo uses q = (1 - eps) * onehot + eps / V (an assumption: §5.4 does not say which)")
    expect(abs(val - 0.39754330) < 1e-6, "eps = 0.2, V = 2: target distribution [0.1, 0.9] gives 0.3975")
    V = 7
    u = torch.zeros(5, V)
    tt = torch.randint(0, V, (5,))
    expect(abs(float(label_smoothed_ce(u, tt, eps=0.3)) - math.log(V)) < 1e-6,
           "uniform logits give ln V for any eps")
    z = torch.randn(9, V)
    ty = torch.randint(0, V, (9,))
    expect(abs(float(label_smoothed_ce(z, ty, eps=0.1)) - float(F.cross_entropy(z, ty, label_smoothing=0.1))) < 1e-5,
           "matches torch's F.cross_entropy(..., label_smoothing=0.1) (mean over rows)")

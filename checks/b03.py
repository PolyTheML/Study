"""Block 03: softmax, logsumexp, cross-entropy (NumPy, gradient by hand)."""
import numpy as np

from ._util import _fd_grad, checkpoint, close, expect, rel_err


@checkpoint("03: softmax")
def check_softmax(softmax):
    rng = np.random.default_rng(0)
    expect(close(softmax(np.array([0.0, np.log(3.0)])), [0.25, 0.75]), "softmax([0, ln 3]) = [0.25, 0.75]")
    Z = rng.normal(size=(4, 5))
    S = softmax(Z)
    expect(np.shape(S) == Z.shape, "keeps the input shape")
    expect(close(S.sum(axis=-1), np.ones(4)), "each row sums to 1",
           "normalize over the LAST axis (the vocabulary axis) and keep dims so it broadcasts")
    expect(close(softmax(Z + 100.0), S), "unchanged when a constant is added to every logit")
    big = softmax(np.array([1000.0, 1001.0, 1002.0]))
    expect(np.all(np.isfinite(big)) and close(big, softmax(np.array([0.0, 1.0, 2.0]))),
           "stable for huge logits (no overflow, no nan)",
           "use the property you just passed: shift by something that makes exp safe")
    Z3 = rng.normal(size=(2, 3, 4))
    expect(close(softmax(Z3).sum(axis=-1), np.ones((2, 3))), "works on (B, T, V) arrays")


@checkpoint("03: logsumexp")
def check_logsumexp(logsumexp):
    rng = np.random.default_rng(1)
    expect(close(logsumexp(np.array([0.0, 0.0])), np.log(2.0)), "logsumexp([0, 0]) = ln 2")
    v = logsumexp(np.array([1000.0, 1000.0]))
    expect(np.isfinite(v) and close(v, 1000.0 + np.log(2.0)), "stable for huge inputs",
           "pull the largest value out of the sum before exponentiating")
    Z = rng.normal(size=(3, 4))
    out = logsumexp(Z)
    expect(np.shape(out) == (3,) and close(out, np.log(np.exp(Z).sum(axis=-1))),
           "reduces over the last axis: (3, 4) -> (3,)")


@checkpoint("03: cross_entropy (loss and gradient)")
def check_cross_entropy(cross_entropy, softmax):
    rng = np.random.default_rng(2)
    N, V = 6, 5
    targets = rng.integers(0, V, size=N)
    loss, d = cross_entropy(np.zeros((N, V)), targets)
    expect(close(loss, np.log(V)), "uniform logits: loss = ln V")

    logits = rng.normal(size=(N, V))
    loss, d = cross_entropy(logits, targets)
    p = softmax(logits)
    expect(close(loss, -np.mean(np.log(p[np.arange(N), targets]))),
           "loss = mean over rows of -log p[row, target]", "mean over N rows, not sum")
    expect(np.shape(d) == logits.shape, "dlogits has the shape of logits")
    fd = _fd_grad(lambda L: cross_entropy(L, targets)[0], logits)
    expect(rel_err(d, fd) < 1e-5, "dlogits matches finite differences",
           "derive dL/dz for ONE row first (you did the 2-class case in P1), then account for the mean over N")
    loss2, d2 = cross_entropy(logits * 1000.0, targets)
    expect(np.isfinite(loss2) and np.all(np.isfinite(d2)), "finite loss and gradient for huge logits",
           "compute log-probabilities with logsumexp instead of log(softmax(...))")

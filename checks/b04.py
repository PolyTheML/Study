"""Block 04: MLP language model (embeddings + tanh hidden layer), backprop by hand."""
import numpy as np

from ._util import _fd_grad, checkpoint, expect

KEYS = ("E", "W1", "b1", "W2", "b2")

HINTS = {
    "E": "a token can appear several times in one batch (even in one row). Every occurrence must ADD its "
         "gradient into that row of E; plain fancy-index assignment keeps only one of them",
    "W1": "the derivative of tanh(a) is 1 - tanh(a)^2; reuse the forward value you stored in the cache",
    "b1": "the bias gradient sums the upstream gradient over the batch axis",
    "W2": "logits = h @ W2 + b2, so dW2 pairs h with dlogits; check which side the transpose goes on",
    "b2": "the bias gradient sums dlogits over the batch axis",
}


@checkpoint("04: make_dataset")
def check_make_dataset(make_dataset):
    X, Y = make_dataset([5, 6, 7, 8], 2)
    X, Y = np.asarray(X), np.asarray(Y)
    expect(X.shape == (2, 2) and Y.shape == (2,), "N = len(ids) - context examples")
    expect(np.array_equal(X, [[5, 6], [6, 7]]) and np.array_equal(Y, [7, 8]),
           "X[n] = ids[n : n + C], Y[n] = ids[n + C]")


@checkpoint("04: MLP LM forward and backward")
def check_mlp_lm(init_params, forward, backward, cross_entropy):
    rng = np.random.default_rng(0)
    V, C, d, H = 7, 3, 4, 5
    params = init_params(V, C, d, H, rng)
    shapes = {"E": (V, d), "W1": (C * d, H), "b1": (H,), "W2": (H, V), "b2": (V,)}
    expect(set(params) == set(KEYS), f"params has keys {KEYS}")
    expect(all(np.shape(params[k]) == shapes[k] for k in KEYS),
           "parameter shapes: E (V,d), W1 (C*d,H), b1 (H,), W2 (H,V), b2 (V,)")
    # The gradient test uses the check's own fixed-scale parameters, so it does not depend on
    # your init choice (a saturated tanh or tiny weights make finite differences unreliable).
    params = {
        "E": 0.5 * rng.normal(size=(V, d)),
        "W1": rng.normal(size=(C * d, H)) / np.sqrt(C * d),
        "b1": 0.1 * rng.normal(size=H),
        "W2": rng.normal(size=(H, V)) / np.sqrt(H),
        "b2": 0.1 * rng.normal(size=V),
    }

    X = np.array([[1, 1, 2], [2, 3, 1], [0, 1, 1], [6, 6, 6]])  # repeated tokens on purpose
    Y = np.array([3, 0, 1, 6])
    logits, cache = forward(params, X)
    expect(np.shape(logits) == (4, V), "logits have shape (N, V)")

    _, dlogits = cross_entropy(logits, Y)
    grads = backward(params, cache, dlogits)
    expect(set(grads) == set(KEYS), "backward returns a gradient for every parameter")
    for k in KEYS:
        expect(np.shape(grads[k]) == shapes[k], f"grad {k} has the shape of {k}")

    def loss_wrt(key):
        def f(value):
            p = dict(params)
            p[key] = value
            return cross_entropy(forward(p, X)[0], Y)[0]
        return f

    for k in KEYS:
        fd = _fd_grad(loss_wrt(k), params[k])
        err = np.max(np.abs(np.asarray(grads[k], dtype=float) - fd))
        expect(err <= 1e-4 * np.max(np.abs(fd)) + 1e-8, f"grad {k} matches finite differences", HINTS[k])

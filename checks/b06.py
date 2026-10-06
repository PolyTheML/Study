"""Block 06: scaled dot-product attention (Eq. 1) and the causal mask (§3.2.3).

Repo convention: mask is boolean, True = "this query may attend to this key".
"""
import numpy as np

from ._util import _fd_grad, checkpoint, close, expect, fail, rel_err  # noqa: F401


@checkpoint("06: causal_mask")
def check_causal_mask(causal_mask):
    M = np.asarray(causal_mask(4))
    expect(M.dtype == bool and M.shape == (4, 4), "boolean array of shape (T, T)")
    expect(all(M[i, j] == (j <= i) for i in range(4) for j in range(4)),
           "M[i, j] is True exactly when j <= i (row = query position, column = key position)",
           "position i may look at itself and the past, never the future")


@checkpoint("06: scaled_dot_product_attention")
def check_attention(attn):
    rng = np.random.default_rng(0)
    Q = np.array([[2.0, 0.0, 0.0, 0.0]])
    K = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]])
    V = np.array([[1.0, 0.0], [0.0, 1.0]])
    out, w = attn(Q, K, V)
    expect(np.shape(w) == (1, 2) and np.shape(out) == (1, 2),
           "shapes: weights (T_q, T_k), out (T_q, d_v)")
    if close(w[0], [0.88079708, 0.11920292], atol=1e-6):
        fail("weights equal the UNSCALED softmax([2, 0])",
             "Eq. 1 divides Q K^T by sqrt(d_k) before the softmax; here d_k = 4")
    expect(close(w[0], [0.73105858, 0.26894142]),
           "known value: q.k = 2, d_k = 4, so softmax([2/2, 0]) = [0.7311, 0.2689]")
    expect(close(out, w @ V), "out = weights @ V")

    B, T, dk, dv = 2, 5, 8, 3
    Q, K, V = rng.normal(size=(B, T, dk)), rng.normal(size=(B, T, dk)), rng.normal(size=(B, T, dv))
    try:
        out, w = attn(Q, K, V)
    except ValueError as e:
        fail(f"your function raised ValueError on (B, T, d) inputs: {e}",
             "transpose only the last two axes of K; K.T reverses ALL axes of a 3-D array")
    expect(np.shape(out) == (B, T, dv) and np.shape(w) == (B, T, T),
           "works with a leading batch axis",
           "transpose only the last two axes of K; K.T reverses ALL axes of a 3-D array")
    expect(close(w.sum(axis=-1), np.ones((B, T))), "each query's weights sum to 1",
           "softmax over the key axis, which is the last axis of the scores")
    out0, _ = attn(np.zeros((B, T, dk)), K, V)
    expect(close(out0, np.repeat(V.mean(axis=1, keepdims=True), T, axis=1)),
           "Q = 0 gives uniform weights, so every output is the average of the V rows")
    out, w = attn(rng.normal(size=(3, dk)), rng.normal(size=(7, dk)), rng.normal(size=(7, dv)))
    expect(np.shape(out) == (3, dv) and np.shape(w) == (3, 7),
           "T_q may differ from T_k (needed later for encoder-decoder attention)")


@checkpoint("06: masked attention")
def check_masked_attention(attn, causal_mask):
    rng = np.random.default_rng(1)
    T, dk, dv = 6, 4, 3
    Q, K, V = rng.normal(size=(T, dk)), rng.normal(size=(T, dk)), rng.normal(size=(T, dv))
    M = np.asarray(causal_mask(T))
    out, w = attn(Q, K, V, mask=M)
    expect(np.allclose(w[np.triu_indices(T, 1)], 0.0), "weights above the diagonal are exactly 0",
           "set masked SCORES to -inf (or a huge negative number) before the softmax")
    expect(close(w.sum(axis=-1), np.ones(T)), "rows still sum to 1 after masking",
           "zeroing weights AFTER the softmax breaks normalization; mask before it")

    t = 2
    Q2, K2, V2 = Q.copy(), K.copy(), V.copy()
    Q2[t + 1:] += 5 * rng.normal(size=(T - t - 1, dk))
    K2[t + 1:] += 5 * rng.normal(size=(T - t - 1, dk))
    V2[t + 1:] += 5 * rng.normal(size=(T - t - 1, dv))
    out2, _ = attn(Q2, K2, V2, mask=M)
    expect(close(out[: t + 1], out2[: t + 1]),
           "changing positions > t never changes outputs at positions <= t (no peeking)")

    only0 = np.zeros((T, T), dtype=bool)
    only0[:, 0] = True
    out3, _ = attn(Q, K, V, mask=only0)
    expect(close(out3, np.repeat(V[:1], T, axis=0)),
           "mask convention: True = allowed (only key 0 allowed, so every output equals V[0])",
           "in this repo True means 'may attend'. PyTorch's nn.MultiheadAttention uses the opposite convention")

    QB, KB, VB = rng.normal(size=(2, T, dk)), rng.normal(size=(2, T, dk)), rng.normal(size=(2, T, dv))
    outB, wB = attn(QB, KB, VB, mask=M)
    expect(np.shape(outB) == (2, T, dv) and np.allclose(wB[:, 0, 1:], 0.0),
           "a (T, T) mask broadcasts over the batch axis")


@checkpoint("06 stretch: attention backward by hand")
def check_attention_backward(forward_with_cache, backward, causal_mask):
    rng = np.random.default_rng(2)
    T, dk, dv = 4, 3, 2
    Q, K, V = rng.normal(size=(T, dk)), rng.normal(size=(T, dk)), rng.normal(size=(T, dv))
    dout = rng.normal(size=(T, dv))
    for mask, name in ((None, "no mask"), (np.asarray(causal_mask(T)), "causal mask")):
        out, cache = forward_with_cache(Q, K, V, mask)
        dQ, dK, dV = backward(dout, cache)
        loss = lambda q, k, v: float(np.sum(forward_with_cache(q, k, v, mask)[0] * dout))
        expect(rel_err(dV, _fd_grad(lambda v: loss(Q, K, v), V)) < 1e-5, f"dV matches finite differences ({name})",
               "out = W V, so dV pairs W with dout")
        expect(rel_err(dQ, _fd_grad(lambda q: loss(q, K, V), Q)) < 1e-5, f"dQ matches finite differences ({name})",
               "go back through the softmax one row at a time (each row is its own softmax), "
               "then through the 1/sqrt(d_k) scale")
        expect(rel_err(dK, _fd_grad(lambda k: loss(Q, k, V), K)) < 1e-5, f"dK matches finite differences ({name})")

"""Block 07: multi-head attention (§3.2.2), NumPy then PyTorch."""
import numpy as np

from ._util import checkpoint, close, expect


@checkpoint("07: split_heads / merge_heads")
def check_split_merge(split_heads, merge_heads):
    rng = np.random.default_rng(0)
    B, T, D, h = 2, 3, 8, 4
    dk = D // h
    X = rng.normal(size=(B, T, D))
    S = split_heads(X, h)
    expect(np.shape(S) == (B, h, T, dk), "split_heads: (B, T, d_model) -> (B, h, T, d_k)",
           "reshape to (B, T, h, d_k) first, then move the head axis")
    expect(close(S[1, 2, 0], X[1, 0, 2 * dk: 3 * dk]),
           "head i owns columns i*d_k : (i+1)*d_k of each position's vector",
           "reshaping straight to (B, h, T, d_k) mixes positions and heads")
    expect(close(merge_heads(S), X), "merge_heads(split_heads(X)) == X")


def _weights(rng, D):
    return [rng.normal(size=(D, D)) / np.sqrt(D) for _ in range(4)]


@checkpoint("07: multi_head_attention (NumPy)")
def check_mha_numpy(mha, attn, causal_mask):
    rng = np.random.default_rng(1)
    B, T, D, h = 2, 5, 8, 2
    dk = D // h
    X = rng.normal(size=(B, T, D))
    Wq, Wk, Wv, Wo = _weights(rng, D)
    out, w = mha(X, Wq, Wk, Wv, Wo, h)
    expect(np.shape(out) == (B, T, D) and np.shape(w) == (B, h, T, T),
           "shapes: out (B, T, d_model), weights (B, h, T, T)")

    I = np.eye(D)
    out1, _ = mha(X, I, I, I, I, 1)
    ref1, _ = attn(X, X, X)
    expect(close(out1, ref1), "h = 1 with identity projections equals Eq. 1 applied to X itself")

    Wo0 = Wo.copy()
    Wo0[dk:, :] = 0.0
    a, _ = mha(X, Wq, Wk, Wv, Wo0, h)
    head0, _ = attn(X @ Wq[:, :dk], X @ Wk[:, :dk], X @ Wv[:, :dk])
    expect(close(a, head0 @ Wo0[:dk]),
           "head 0 = Attention(X W0^Q, X W0^K, X W0^V) with scaling by sqrt(d_k), not sqrt(d_model) (§3.2.2)",
           "each head scales by the square root of ITS OWN key size, d_k = d_model / h")
    Wq2 = Wq.copy()
    Wq2[:, dk: 2 * dk] += rng.normal(size=(D, dk))
    b, _ = mha(X, Wq2, Wk, Wv, Wo0, h)
    expect(close(a, b), "heads are independent: head 1's query weights cannot change head 0's output",
           "check your split: columns i*d_k : (i+1)*d_k must belong to head i")

    M = np.asarray(causal_mask(T))
    o1, w1 = mha(X, Wq, Wk, Wv, Wo, h, mask=M)
    X2 = X.copy()
    X2[:, 3:] += 5.0
    o2, _ = mha(X2, Wq, Wk, Wv, Wo, h, mask=M)
    expect(close(o1[:, :3], o2[:, :3]), "with a causal (T, T) mask, outputs at t <= 2 ignore positions >= 3")
    expect(np.allclose(w1[..., 0, 1:], 0.0), "the mask broadcasts over batch and heads")


@checkpoint("07: MultiHeadAttention (PyTorch)")
def check_torch_mha(MultiHeadAttention, mha_numpy):
    import torch

    torch.manual_seed(0)
    D, h = 16, 4
    m = MultiHeadAttention(d_model=D, n_heads=h, dropout=0.0, bias=False)
    m.eval()
    expect(all(isinstance(getattr(m, n, None), torch.nn.Linear) for n in ("W_q", "W_k", "W_v", "W_o")),
           "has nn.Linear attributes W_q, W_k, W_v, W_o")
    n_params = sum(p.numel() for p in m.parameters())
    expect(n_params == 4 * D * D, f"parameter count = 4 * d_model^2 = {4 * D * D} with bias=False (got {n_params})")

    X = torch.randn(2, 5, D)
    with torch.no_grad():
        out = m(X)
    expect(tuple(out.shape) == (2, 5, D), "self-attention output has shape (B, T, d_model)")
    Ws = [getattr(m, n).weight.detach().double().numpy().T for n in ("W_q", "W_k", "W_v", "W_o")]
    ref, _ = mha_numpy(X.double().numpy(), *Ws, h)
    expect(close(out, ref, atol=1e-5),
           "matches your NumPy multi-head attention when given W = linear.weight.T",
           "nn.Linear computes x @ weight.T, so the paper's W^Q corresponds to weight.T")

    T = 6
    causal = torch.tril(torch.ones(T, T, dtype=torch.bool))
    X = torch.randn(2, T, D)
    X2 = X.clone()
    X2[:, 4:] += 3.0
    with torch.no_grad():
        o1, o2 = m(X, mask=causal), m(X2, mask=causal)
    expect(torch.allclose(o1[:, :4], o2[:, :4], atol=1e-5), "a (T, T) bool mask (True = allowed) is causal")

    C = torch.randn(2, 9, D)
    with torch.no_grad():
        oc = m(X, context=C)
    expect(tuple(oc.shape) == (2, T, D), "cross-attention: m(x, context=c) takes keys/values from c")
    pad = torch.ones(2, 1, 1, 9, dtype=torch.bool)
    pad[1, ..., 6:] = False
    C2 = C.clone()
    C2[1, 6:] += 7.0
    with torch.no_grad():
        p1, p2 = m(X, context=C, mask=pad), m(X, context=C2, mask=pad)
    expect(torch.allclose(p1, p2, atol=1e-5), "a (B, 1, 1, T_k) padding mask hides masked keys")

    md = MultiHeadAttention(d_model=D, n_heads=h, dropout=0.5, bias=False)
    md.train()
    with torch.no_grad():
        d1, d2 = md(X), md(X)
    md.eval()
    with torch.no_grad():
        e1, e2 = md(X), md(X)
    expect(not torch.allclose(d1, d2) and torch.allclose(e1, e2),
           "dropout on the attention weights is active in train mode and off in eval mode")

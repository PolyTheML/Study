"""Block 09: LayerNorm, position-wise FFN (Eq. 2), and the Transformer block (§3.1)."""
import numpy as np

from ._util import checkpoint, close, expect, fail


@checkpoint("09: layer_norm (NumPy)")
def check_layer_norm(layer_norm):
    rng = np.random.default_rng(0)
    y = layer_norm(np.array([[1.0, 2.0, 3.0, 4.0]]), np.ones(4), np.zeros(4))
    if close(y, [[-1.161895, -0.387298, 0.387298, 1.161895]], atol=1e-4):
        fail("you used the unbiased variance (divide by d - 1)",
             "LayerNorm uses the population variance (divide by d); np.var does that by default")
    expect(close(y, [[-1.3416354, -0.4472118, 0.4472118, 1.3416354]]),
           "known value: [1, 2, 3, 4] -> (x - 2.5) / sqrt(1.25 + eps)")
    y1 = layer_norm(np.array([[1.0, 2.0, 3.0, 4.0]]), np.ones(4), np.zeros(4), eps=1.0)
    expect(close(y1, [[-1.0, -1.0 / 3.0, 1.0 / 3.0, 1.0]]),
           "eps goes inside the square root: eps = 1 gives (x - 2.5) / sqrt(1.25 + 1) = (x - 2.5) / 1.5")
    x = rng.normal(3.0, 5.0, size=(2, 4, 8))
    y = layer_norm(x, np.ones(8), np.zeros(8))
    expect(close(y.mean(axis=-1), np.zeros((2, 4)), atol=1e-6) and close(y.var(axis=-1), np.ones((2, 4)), atol=1e-3),
           "each position is normalized over its feature axis: mean 0, variance 1",
           "statistics over the LAST axis only, with keepdims; not over the batch or the sequence")
    g, b = rng.normal(size=8), rng.normal(size=8)
    expect(close(layer_norm(x, g, b), g * y + b), "gamma scales and beta shifts each feature after normalizing")


@checkpoint("09: ffn (NumPy, Eq. 2)")
def check_ffn(ffn):
    rng = np.random.default_rng(1)
    y = ffn(np.array([[1.0, -1.0]]), np.eye(2), np.zeros(2), np.eye(2), np.zeros(2))
    expect(close(y, [[1.0, 0.0]]), "known value: identity weights give ReLU(x)")
    x = rng.normal(size=(2, 3, 4))
    W1, b1, W2, b2 = rng.normal(size=(4, 6)), rng.normal(size=6), rng.normal(size=(6, 4)), rng.normal(size=4)
    y = ffn(x, W1, b1, W2, b2)
    expect(np.shape(y) == (2, 3, 4), "(B, T, d_model) in, (B, T, d_model) out")
    expect(close(ffn(x, W1, np.full(6, -1e3), W2, b2), np.broadcast_to(b2, (2, 3, 4))),
           "with very negative pre-activations ReLU outputs 0, leaving only b2")
    perm = rng.permutation(3)
    expect(close(ffn(x[:, perm], W1, b1, W2, b2), y[:, perm]),
           "position-wise: shuffling positions just shuffles the outputs")


@checkpoint("09: TransformerBlock (PyTorch)")
def check_torch_block(TransformerBlock):
    import torch

    torch.manual_seed(0)
    D, T = 16, 6
    blk = TransformerBlock(d_model=D, n_heads=4, d_ff=32, dropout=0.1, norm_first=False)
    expect(all(hasattr(blk, n) for n in ("attn", "ffn", "ln1", "ln2")) and
           all(hasattr(blk.ffn, n) for n in ("lin1", "lin2")),
           "has attn, ffn (with lin1, lin2), ln1, ln2")
    x = torch.randn(2, T, D)
    blk.eval()
    with torch.no_grad():
        y1, y2 = blk(x), blk(x)
    expect(tuple(y1.shape) == (2, T, D) and torch.allclose(y1, y2), "shape preserved; deterministic in eval mode")
    expect(torch.allclose(y1.mean(-1), torch.zeros(2, T), atol=1e-5),
           "post-norm (§3.1): the block's last operation is LayerNorm, so outputs have mean 0 per position",
           "the paper computes LayerNorm(x + Sublayer(x)); norm_first=False must follow it")
    blk.train()
    with torch.no_grad():
        z1, z2 = blk(x), blk(x)
    expect(not torch.allclose(z1, z2), "dropout is active in train mode (§5.4)")
    blk.eval()

    post = TransformerBlock(d_model=D, n_heads=4, d_ff=32, dropout=0.0, norm_first=False)
    post.eval()
    with torch.no_grad():
        for lin in (post.attn.W_o, post.ffn.lin2):
            for p in lin.parameters():
                p.zero_()
        out = post(x)
    expect(torch.allclose(out, torch.nn.functional.layer_norm(x, (D,)), atol=1e-4),
           "post-norm with zeroed output projections gives LayerNorm(x): the residual x + Sublayer(x) is there",
           "each sub-layer's output is ADDED to its input before the LayerNorm (§3.1)")

    causal = torch.tril(torch.ones(T, T, dtype=torch.bool))
    x2 = x.clone()
    x2[:, 4:] += 3.0
    with torch.no_grad():
        c1, c2 = blk(x, mask=causal), blk(x2, mask=causal)
    expect(torch.allclose(c1[:, :4], c2[:, :4], atol=1e-5), "forward(x, mask) passes the causal mask to attention")

    pre = TransformerBlock(d_model=D, n_heads=4, d_ff=32, dropout=0.0, norm_first=True)
    pre.eval()
    with torch.no_grad():
        for lin in (pre.attn.W_o, pre.ffn.lin2):
            for p in lin.parameters():
                p.zero_()
        out = pre(x)
    expect(torch.allclose(out, x, atol=1e-6),
           "pre-norm (norm_first=True): with zeroed output projections the block is the identity",
           "pre-norm computes x + Sublayer(LayerNorm(x)); no LayerNorm on the residual path")

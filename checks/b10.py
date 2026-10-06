"""Block 10: the tiny decoder-only GPT."""
from ._util import checkpoint, expect


@checkpoint("10: TinyGPT")
def check_gpt(TinyGPT):
    import torch

    torch.manual_seed(0)
    V, T = 11, 10
    cfg = dict(vocab_size=V, d_model=16, n_heads=4, n_layers=2, d_ff=32, max_len=12, dropout=0.0)
    m = TinyGPT(**cfg, pos="sinusoidal", tie_weights=True, norm_first=False)
    m.eval()
    expect(hasattr(m, "tok_emb") and hasattr(m, "lm_head") and getattr(m, "max_len", None) == 12,
           "has tok_emb (nn.Embedding), lm_head (nn.Linear) and a max_len attribute")
    idx = torch.randint(0, V, (3, T))
    with torch.no_grad():
        logits = m(idx)
    expect(tuple(logits.shape) == (3, T, V), "logits have shape (B, T, vocab_size)")

    idx2 = idx.clone()
    idx2[:, 6:] = torch.randint(0, V, (3, T - 6))
    with torch.no_grad():
        logits2 = m(idx2)
    expect(torch.allclose(logits[:, :6], logits2[:, :6], atol=1e-5),
           "causal: logits at positions < 6 ignore tokens at positions >= 6",
           "every block needs the causal mask")
    expect(m.lm_head.weight.data_ptr() == m.tok_emb.weight.data_ptr(),
           "tie_weights=True: lm_head and tok_emb share one weight matrix (§3.4)")

    untied = TinyGPT(**cfg, pos="sinusoidal", tie_weights=False, norm_first=False)
    n_tied = sum(p.numel() for p in m.parameters())
    n_untied = sum(p.numel() for p in untied.parameters())
    expect(n_untied - n_tied == V * 16, "untied model has exactly vocab_size * d_model more parameters",
           "lm_head should have no bias, so the only difference is one (V, d_model) matrix")

    for pos in ("learned", "none"):
        mm = TinyGPT(**cfg, pos=pos, tie_weights=True, norm_first=True)
        mm.eval()
        with torch.no_grad():
            out = mm(idx)
        expect(tuple(out.shape) == (3, T, V), f"pos='{pos}' with norm_first=True runs")

    # one layer only: with 2+ causal layers, order leaks in through the mask (a block 13 discussion point)
    one = dict(cfg, n_layers=1)
    nopos = TinyGPT(**one, pos="none", tie_weights=True, norm_first=False)
    sinus = TinyGPT(**one, pos="sinusoidal", tie_weights=True, norm_first=False)
    nopos.eval()
    sinus.eval()
    a = torch.tensor([[1, 2, 3, 4, 5]])
    b = torch.tensor([[3, 1, 4, 2, 5]])  # same tokens before the last one, shuffled
    with torch.no_grad():
        la, lb = nopos(a)[0, -1], nopos(b)[0, -1]
        sa, sb = sinus(a)[0, -1], sinus(b)[0, -1]
    expect(torch.allclose(la, lb, atol=1e-5),
           "pos='none', one layer: the last position sees a bag of earlier tokens (order invisible)")
    expect(not torch.allclose(sa, sb, atol=1e-5), "pos='sinusoidal': order becomes visible")

    zero = TinyGPT(**dict(cfg, n_layers=0), pos="none", tie_weights=True, norm_first=False)
    zero.eval()
    with torch.no_grad():
        E = zero.tok_emb.weight
        expected = (E[idx] * 16 ** 0.5) @ E.T
        got = zero(idx)
    expect(torch.allclose(got, expected, atol=1e-4),
           "with no layers and no positions, logits = sqrt(d_model) * E[idx] @ E^T  (embedding scale, §3.4)",
           "multiply the token embeddings by sqrt(d_model) before adding positions")

    pre = TinyGPT(**cfg, pos="sinusoidal", tie_weights=True, norm_first=True)
    diff = sum(p.numel() for p in pre.parameters()) - n_tied
    expect(diff == 2 * 16, "norm_first=True adds exactly one final LayerNorm (2 * d_model parameters)")

    for model, name in ((m, "sinusoidal"), (TinyGPT(**cfg, pos="none"), "none")):
        raised = False
        try:
            with torch.no_grad():
                model(torch.randint(0, V, (1, 13)))
        except Exception:
            raised = True
        expect(raised, f"pos='{name}': more than max_len tokens raises an error instead of failing silently")

"""Block 15: the full encoder-decoder of Figure 1."""
from ._util import checkpoint, expect


@checkpoint("15: shift_right")
def check_shift_right(shift_right):
    import torch

    out = shift_right(torch.tensor([[5, 6, 7, 2], [8, 9, 2, 0]]), bos_id=1)
    expect(out.tolist() == [[1, 5, 6, 7], [1, 8, 9, 2]],
           "decoder input = BOS followed by the target without its last token (Fig. 1 'shifted right')")


@checkpoint("15: TransformerSeq2Seq")
def check_seq2seq(TransformerSeq2Seq):
    import torch

    torch.manual_seed(0)
    m = TransformerSeq2Seq(src_vocab=13, tgt_vocab=11, d_model=16, n_heads=4, n_enc_layers=2, n_dec_layers=2,
                           d_ff=32, max_len=20, dropout=0.0, pad_id=0)
    m.eval()
    src = torch.randint(1, 13, (2, 7))
    tgt = torch.randint(1, 11, (2, 5))
    with torch.no_grad():
        logits = m(src, tgt)
        memory, src_mask = m.encode(src)
        logits_b = m.decode(tgt, memory, src_mask)
    expect(tuple(logits.shape) == (2, 5, 11), "logits have shape (B, T_tgt, tgt_vocab)")
    expect(tuple(memory.shape) == (2, 7, 16), "encode returns memory of shape (B, S, d_model)")
    expect(torch.allclose(logits, logits_b, atol=1e-6), "forward(src, tgt) == decode(tgt, *encode(src))")

    tgt2 = tgt.clone()
    tgt2[:, 3:] = torch.randint(1, 11, (2, 2))
    with torch.no_grad():
        l2 = m(src, tgt2)
    expect(torch.allclose(logits[:, :3], l2[:, :3], atol=1e-5), "decoder self-attention is causal")

    src2 = src.clone()
    src2[:, 2] = (src2[:, 2] % 12) + 1
    with torch.no_grad():
        l3 = m(src2, tgt)
    expect(not torch.allclose(logits, l3, atol=1e-5), "outputs depend on the source (cross-attention is wired in)")

    padded = torch.cat([src, torch.zeros(2, 4, dtype=torch.long)], dim=1)
    with torch.no_grad():
        l4 = m(padded, tgt)
    expect(torch.allclose(logits, l4, atol=1e-5),
           "appending PAD tokens to the source changes nothing",
           "mask PAD keys in BOTH encoder self-attention and decoder cross-attention")

"""The tiny decoder-only GPT (block 10)."""
import torch.nn as nn


class TinyGPT(nn.Module):
    """Decoder-only Transformer language model.

    Fig. 1's decoder stack without the encoder-decoder attention sub-layer (that deletion is
    outside the paper: Radford et al. 2018). Each layer is a TransformerBlock with a causal mask.

    __init__(vocab_size, d_model, n_heads, n_layers, d_ff, max_len, dropout=0.1,
             pos="sinusoidal", tie_weights=True, norm_first=False)
        Attributes:
          tok_emb:  nn.Embedding(vocab_size, d_model)
          lm_head:  nn.Linear(d_model, vocab_size, bias=False)
          max_len:  int, longest context the model accepts
        pos: "sinusoidal" (§3.5, a fixed buffer), "learned" (nn.Embedding(max_len, d_model), Table 3 row E),
             or "none" (no positional information; for the ablation in block 13)
        tie_weights=True: lm_head shares tok_emb's weight matrix (§3.4)
        norm_first is passed to every TransformerBlock; norm_first=True also adds one final
             LayerNorm before lm_head (pre-norm convention, outside the paper)

    forward(idx) -> logits
        idx: (B, T) int64 token ids, T <= max_len (raise ValueError otherwise, for every pos option)
        Steps: tok_emb(idx) * sqrt(d_model) (§3.4), add positions, dropout (§5.4),
               blocks with a causal mask, [final LayerNorm if norm_first], lm_head
        returns (B, T, vocab_size)
    """

    def __init__(self, vocab_size, d_model, n_heads, n_layers, d_ff, max_len, dropout=0.1,
                 pos="sinusoidal", tie_weights=True, norm_first=False):
        super().__init__()
        raise NotImplementedError("Block 10: your turn")

    def forward(self, idx):
        raise NotImplementedError("Block 10: your turn")

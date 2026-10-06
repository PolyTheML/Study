"""Feed-forward network and the Transformer block (block 09)."""
import torch.nn as nn


class FeedForward(nn.Module):
    """Eq. 2 position-wise FFN: lin2(dropout(relu(lin1(x)))).

    __init__(d_model, d_ff, dropout=0.0)
        Attributes: lin1 = nn.Linear(d_model, d_ff), lin2 = nn.Linear(d_ff, d_model) (with biases, as in Eq. 2).
        Dropout inside the FFN is an assumption (tensor2tensor style); residual dropout lives in the block.
    forward(x): (B, T, d_model) -> (B, T, d_model)
    """

    def __init__(self, d_model, d_ff, dropout=0.0):
        super().__init__()
        raise NotImplementedError("Block 09: your turn")

    def forward(self, x):
        raise NotImplementedError("Block 09: your turn")


class TransformerBlock(nn.Module):
    """One encoder-style layer: self-attention sub-layer, then FFN sub-layer (§3.1).

    __init__(d_model, n_heads, d_ff, dropout=0.1, norm_first=False)
        Attributes: attn (MultiHeadAttention), ffn (FeedForward), ln1, ln2 (nn.LayerNorm(d_model)).
        norm_first=False is the paper's post-norm:  x = LayerNorm(x + Dropout(Sublayer(x)))
        norm_first=True is pre-norm (outside the paper, used by GPT-2 and later):
                                                    x = x + Dropout(Sublayer(LayerNorm(x)))
        Residual dropout on each sub-layer output before the add (§5.4).
    forward(x, mask=None): (B, T, d_model) -> (B, T, d_model); mask is passed to self-attention.
    """

    def __init__(self, d_model, n_heads, d_ff, dropout=0.1, norm_first=False):
        super().__init__()
        raise NotImplementedError("Block 09: your turn")

    def forward(self, x, mask=None):
        raise NotImplementedError("Block 09: your turn")

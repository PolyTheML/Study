"""PyTorch multi-head attention (block 07)."""
import torch.nn as nn


class MultiHeadAttention(nn.Module):
    """Multi-head attention, §3.2.2 and Fig. 2 (right).

    __init__(d_model, n_heads, dropout=0.0, bias=False)
        Attributes: W_q, W_k, W_v, W_o, each nn.Linear(d_model, d_model, bias=bias).
        d_k = d_model // n_heads. `dropout` is applied to the attention weights
        (an assumption: §5.4 does not list it; tensor2tensor does it).

    forward(x, context=None, mask=None) -> (B, T_q, d_model)
        x:       (B, T_q, d_model), source of the queries
        context: (B, T_k, d_model) source of keys and values; None means self-attention (context = x)
        mask:    None, or bool tensor broadcastable to (B, n_heads, T_q, T_k); True = allowed.
                 A causal mask has shape (T_q, T_k); a key-padding mask has shape (B, 1, 1, T_k).
    Trap: nn.Linear computes x @ weight.T, so the paper's W^Q is W_q.weight.T.
    """

    def __init__(self, d_model, n_heads, dropout=0.0, bias=False):
        super().__init__()
        raise NotImplementedError("Block 07: your turn")

    def forward(self, x, context=None, mask=None):
        raise NotImplementedError("Block 07: your turn")

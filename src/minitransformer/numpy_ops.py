"""NumPy versions of the core mechanisms (blocks 03, 06, 07, 08, 09).

Graduate each function here from its notebook once its checkpoint passes.
Conventions used throughout this repo:
  * row vectors: rows of Q, K, V, X are positions; a linear layer is x @ W  (paper, Eq. 2)
  * boolean masks: True = "this query may attend to this key"
  * the last axis is the feature / vocabulary axis
"""
import numpy as np


def softmax(z):
    """Softmax over the last axis.

    z: float array of any shape (..., V)
    returns: array of the same shape; every slice along the last axis is non-negative and sums to 1.
    Must stay finite for huge inputs such as [1000, 1001, 1002].
    """
    raise NotImplementedError("Block 03: your turn")


def logsumexp(z):
    """log(sum(exp(z))) over the last axis, computed stably.

    z: float array (..., V)
    returns: array (...,), the last axis reduced away. Finite for huge inputs.
    """
    raise NotImplementedError("Block 03: your turn")


def cross_entropy(logits, targets):
    """Mean cross-entropy and its gradient.

    logits:  float array (N, V), unnormalized scores
    targets: int array (N,), values in [0, V)
    returns: (loss, dlogits)
        loss    = mean over the N rows of -log softmax(logits)[row, target]  (a float)
        dlogits = d loss / d logits, shape (N, V)
    For (B, T, V) logits, reshape to (B*T, V) and targets to (B*T,) before calling.
    """
    raise NotImplementedError("Block 03: your turn")


def causal_mask(T):
    """Boolean (T, T) mask with M[i, j] = True exactly when j <= i.

    Row = query position, column = key position. True = allowed.
    """
    raise NotImplementedError("Block 06: your turn")


def scaled_dot_product_attention(Q, K, V, mask=None):
    """Eq. 1: softmax(Q K^T / sqrt(d_k)) V, with optional masking (§3.2.3).

    Q:    (..., T_q, d_k)
    K:    (..., T_k, d_k)
    V:    (..., T_k, d_v)
    mask: None, or bool array broadcastable to (..., T_q, T_k); True = allowed.
          Disallowed scores become -inf (or a very large negative number) BEFORE the softmax.
    returns: (out, weights)
        out:     (..., T_q, d_v)
        weights: (..., T_q, T_k), each row sums to 1
    Leading axes (batch, heads) are optional and must broadcast.
    """
    raise NotImplementedError("Block 06: your turn")


def split_heads(X, h):
    """(B, T, d_model) -> (B, h, T, d_k) with d_k = d_model // h.

    Head i owns columns i*d_k : (i+1)*d_k of each position's vector.
    """
    raise NotImplementedError("Block 07: your turn")


def merge_heads(X):
    """(B, h, T, d_k) -> (B, T, h * d_k). Exact inverse of split_heads."""
    raise NotImplementedError("Block 07: your turn")


def multi_head_attention(X, Wq, Wk, Wv, Wo, h, mask=None):
    """Multi-head self-attention, §3.2.2, with packed weights and no biases.

    X:  (B, T, d_model)
    Wq, Wk, Wv: (d_model, d_model). Columns i*d_k : (i+1)*d_k are head i's W_i^Q (resp. K, V).
    Wo: (d_model, d_model), the paper's W^O (rows i*d_k : (i+1)*d_k act on head i's output).
    h:  number of heads; d_model must be divisible by h.
    mask: None, or bool broadcastable to (B, h, T, T); True = allowed. A causal mask is (T, T);
          a key-padding mask is (B, 1, 1, T).
    returns: (out (B, T, d_model), weights (B, h, T, T))
    Biases: the paper gives only matrices (§3.2.2); this repo's NumPy version has none (assumption).
    """
    raise NotImplementedError("Block 07: your turn")


def sinusoidal_pe(max_len, d_model):
    """§3.5 positional encodings, shape (max_len, d_model), d_model even.

    PE[pos, 2i]   = sin(pos / 10000^(2i / d_model))
    PE[pos, 2i+1] = cos(pos / 10000^(2i / d_model))
    where i = 0, 1, ..., d_model/2 - 1 indexes dimension PAIRS.
    """
    raise NotImplementedError("Block 08: your turn")


def layer_norm(x, gamma, beta, eps=1e-5):
    """Layer normalization over the last axis (Ba et al. [1]).

    x: (..., d); gamma, beta: (d,)
    returns gamma * (x - mean) / sqrt(var + eps) + beta, with mean and population variance
    taken over the last axis of each position. eps = 1e-5 is an assumption (the paper gives none).
    """
    raise NotImplementedError("Block 09: your turn")


def ffn(x, W1, b1, W2, b2):
    """Position-wise feed-forward network, Eq. 2: max(0, x W1 + b1) W2 + b2.

    x: (..., d_model); W1: (d_model, d_ff); b1: (d_ff,); W2: (d_ff, d_model); b2: (d_model,)
    returns (..., d_model)
    """
    raise NotImplementedError("Block 09: your turn")

"""Learning-rate schedule and loss (block 11)."""


def noam_lr(step, d_model, warmup_steps):
    """Eq. 3: d_model^-0.5 * min(step^-0.5, step * warmup_steps^-1.5).

    step starts at 1 (Eq. 3 is undefined at 0). Returns a float.
    """
    raise NotImplementedError("Block 11: your turn")


def label_smoothed_ce(logits, targets, eps):
    """Cross-entropy against a smoothed target distribution (§5.4).

    logits: (N, V) float tensor; targets: (N,) int64 tensor; eps: float in [0, 1)
    Target distribution q = (1 - eps) * onehot(target) + eps / V for every class.
    (Assumption: the paper does not give the exact form; this matches Szegedy et al. [36]
    and torch's F.cross_entropy(label_smoothing=eps).)
    returns: scalar tensor, mean over the N rows of -sum_k q_k log softmax(logits)_k
    """
    raise NotImplementedError("Block 11: your turn")

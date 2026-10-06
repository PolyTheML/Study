"""Autoregressive generation for the GPT (block 12)."""


def generate(model, idx, max_new_tokens, temperature=1.0, top_k=None, greedy=False, generator=None):
    """Extend each sequence in idx by max_new_tokens tokens.

    model: callable (B, T) -> logits (B, T, V), with an int attribute model.max_len
    idx:   (B, T) int64 prompt
    Each step: feed only the last model.max_len tokens, take the logits at the LAST position,
      greedy=True       -> argmax
      otherwise         -> divide logits by temperature, keep only the top_k largest if top_k is set,
                           softmax, sample with torch.multinomial(..., generator=generator)
    Run without gradients.
    returns: (B, T + max_new_tokens) int64, the prompt followed by the new tokens
    """
    raise NotImplementedError("Block 12: your turn")

"""Batching for language-model training (block 11)."""


def get_batch(data, block_size, batch_size, generator=None):
    """Sample random training windows.

    data: 1-D int64 tensor of token ids
    returns: (x, y), both (batch_size, block_size)
        x[b] = data[s : s + block_size],  y[b] = data[s + 1 : s + 1 + block_size]
        with start s drawn uniformly from 0 .. len(data) - block_size - 1 (inclusive),
        using `generator` (a torch.Generator) when given.
    """
    raise NotImplementedError("Block 11: your turn")

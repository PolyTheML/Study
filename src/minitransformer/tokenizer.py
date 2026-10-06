"""Tokenizers: character level (block 01) and byte-level BPE (block 14)."""


def build_vocab(text):
    """Character vocabulary.

    text: str
    returns: (stoi, itos)
        stoi: dict char -> id, ids 0 .. V-1 assigned in SORTED character order
        itos: dict id -> char, the inverse of stoi
    """
    raise NotImplementedError("Block 01: your turn")


def encode(text, stoi):
    """str -> list of int ids, one id per character."""
    raise NotImplementedError("Block 01: your turn")


def decode(ids, itos):
    """list of int ids -> str. decode(encode(s, stoi), itos) == s."""
    raise NotImplementedError("Block 01: your turn")


def get_pair_counts(ids):
    """Count adjacent pairs.

    ids: list of ints
    returns: dict {(a, b): count} over every adjacent pair (ids[k], ids[k+1])
    """
    raise NotImplementedError("Block 14: your turn")


def merge(ids, pair, new_id):
    """Replace every occurrence of `pair` in `ids` by `new_id`.

    Scan left to right without overlaps: merge([1, 1, 1], (1, 1), 256) == [256, 1].
    returns: a new list
    """
    raise NotImplementedError("Block 14: your turn")


def train_bpe(text, num_merges):
    """Learn byte-level BPE merges.

    Start from the UTF-8 bytes of `text` (ids 0..255). Repeat num_merges times: find the most
    frequent adjacent pair, give it the next new id (256, 257, ...), merge it everywhere.
    Ties may be broken any way you like, but deterministically.
    Stop early only if no pair is left.
    returns: list of ((a, b), new_id) in the order learned
    """
    raise NotImplementedError("Block 14: your turn")


def bpe_encode(text, merges):
    """str -> list of ids: UTF-8 bytes, then apply `merges` in the order they were learned."""
    raise NotImplementedError("Block 14: your turn")


def bpe_decode(ids, merges):
    """list of ids -> str. Expand ids back to bytes, then decode UTF-8 (errors="replace")."""
    raise NotImplementedError("Block 14: your turn")

"""Block 14: byte-level BPE (§5.1, Sennrich et al. [31]; byte-level variant from later work)."""
from ._util import checkpoint, expect


@checkpoint("14: BPE building blocks")
def check_bpe_parts(get_pair_counts, merge):
    expect(dict(get_pair_counts([1, 2, 3, 1, 2])) == {(1, 2): 2, (2, 3): 1, (3, 1): 1},
           "get_pair_counts counts adjacent pairs")
    expect(list(merge([1, 2, 3, 1, 2], (1, 2), 256)) == [256, 3, 256], "merge replaces every occurrence of the pair")
    expect(list(merge([1, 1, 1], (1, 1), 256)) == [256, 1],
           "merge scans left to right without overlaps: [1, 1, 1] -> [256, 1]",
           "after replacing a pair, jump past both of its elements")


@checkpoint("14: train / encode / decode")
def check_bpe(train_bpe, bpe_encode, bpe_decode):
    merges = train_bpe("aaabdaaabac", 3)
    expect(len(merges) == 3, "train_bpe returns num_merges merges")
    expect(tuple(merges[0][0]) == (97, 97) and merges[0][1] == 256,
           "first merge is the most frequent byte pair ('a','a' = 97, 97) and gets id 256")
    expect([m[1] for m in merges] == [256, 257, 258], "new ids are 256, 257, 258, ... in order")

    corpus = ("the cat sat on the mat. the hat is on the cat. " * 20) + ("សួស្តី ពិភពលោក " * 20)
    merges = train_bpe(corpus, 40)
    for s in ["the cat", "សួស្តី", "naïve café 🙂", ""]:
        expect(bpe_decode(bpe_encode(s, merges), merges) == s,
               f"round trip on {s!r} (byte-level BPE can encode any string)")
    lengths = [len(bpe_encode(corpus, merges[:k])) for k in (0, 10, 20, 40)]
    expect(lengths[0] == len(corpus.encode("utf-8")), "with zero merges, one token per UTF-8 byte")
    expect(all(a >= b for a, b in zip(lengths, lengths[1:])) and lengths[-1] < lengths[0],
           "more merges never make the encoding longer",
           "apply merges in the order they were learned")

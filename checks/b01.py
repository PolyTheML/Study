"""Block 01: character tokenizer."""
import numpy as np

from ._util import checkpoint, expect


@checkpoint("01: character tokenizer")
def check_tokenizer(build_vocab, encode, decode):
    text = "hello world\nHELLO!"
    stoi, itos = build_vocab(text)
    chars = sorted(set(text))
    expect(len(stoi) == len(chars) and len(itos) == len(chars),
           f"vocab size = number of unique characters ({len(chars)})")
    expect(all(stoi.get(c) == i for i, c in enumerate(chars)),
           "ids follow sorted character order, so they are reproducible",
           "sort the unique characters before numbering them")
    expect(all(itos[i] == c for c, i in stoi.items()), "itos is the inverse of stoi")

    ids = encode(text, stoi)
    expect(all(isinstance(i, (int, np.integer)) for i in ids), "encode returns integers")
    expect(len(ids) == len(text), "one id per character")
    expect(decode(ids, itos) == text, "decode(encode(text)) == text  (round trip)")
    expect(len(encode("", stoi)) == 0, "the empty string encodes to an empty sequence")

    km = "សួស្តី ពិភពលោក"
    s2, i2 = build_vocab(km)
    expect(decode(encode(km, s2), i2) == km,
           "round trip works on Khmer text too (Python strings are Unicode code points)")

from conftest import run

from checks import b01, b14
from minitransformer import tokenizer as tok


def test_char_tokenizer():
    run(b01.check_tokenizer, tok.build_vocab, tok.encode, tok.decode)


def test_bpe_parts():
    run(b14.check_bpe_parts, tok.get_pair_counts, tok.merge)


def test_bpe():
    run(b14.check_bpe, tok.train_bpe, tok.bpe_encode, tok.bpe_decode)

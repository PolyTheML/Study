from conftest import run

from checks import b07, b09, b10, b11, b12, b15, b16
from minitransformer import numpy_ops as ops
from minitransformer.attention import MultiHeadAttention
from minitransformer.data import get_batch
from minitransformer.generate import generate
from minitransformer.layers import TransformerBlock
from minitransformer.model import TinyGPT
from minitransformer.seq2seq import (TransformerSeq2Seq, beam_search, greedy_decode, length_penalty,
                                     shift_right)
from minitransformer.train_utils import label_smoothed_ce, noam_lr


def test_mha_torch():
    run(b07.check_torch_mha, MultiHeadAttention, ops.multi_head_attention)


def test_transformer_block():
    run(b09.check_torch_block, TransformerBlock)


def test_tiny_gpt():
    run(b10.check_gpt, TinyGPT)


def test_get_batch():
    run(b11.check_get_batch, get_batch)


def test_noam_lr():
    run(b11.check_noam, noam_lr)


def test_label_smoothing():
    run(b11.check_label_smoothing, label_smoothed_ce)


def test_generate():
    run(b12.check_generate, generate)


def test_shift_right():
    run(b15.check_shift_right, shift_right)


def test_seq2seq():
    run(b15.check_seq2seq, TransformerSeq2Seq)


def test_length_penalty():
    run(b16.check_length_penalty, length_penalty)


def test_decoders():
    run(b16.check_decoders, greedy_decode, beam_search)

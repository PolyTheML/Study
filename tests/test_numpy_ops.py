from conftest import run

from checks import b03, b06, b07, b08, b09
from minitransformer import numpy_ops as ops


def test_softmax():
    run(b03.check_softmax, ops.softmax)


def test_logsumexp():
    run(b03.check_logsumexp, ops.logsumexp)


def test_cross_entropy():
    run(b03.check_cross_entropy, ops.cross_entropy, ops.softmax)


def test_causal_mask():
    run(b06.check_causal_mask, ops.causal_mask)


def test_attention():
    run(b06.check_attention, ops.scaled_dot_product_attention)


def test_masked_attention():
    run(b06.check_masked_attention, ops.scaled_dot_product_attention, ops.causal_mask)


def test_split_merge():
    run(b07.check_split_merge, ops.split_heads, ops.merge_heads)


def test_mha_numpy():
    run(b07.check_mha_numpy, ops.multi_head_attention, ops.scaled_dot_product_attention, ops.causal_mask)


def test_positional_encoding():
    run(b08.check_pe, ops.sinusoidal_pe)


def test_layer_norm():
    run(b09.check_layer_norm, ops.layer_norm)


def test_ffn():
    run(b09.check_ffn, ops.ffn)

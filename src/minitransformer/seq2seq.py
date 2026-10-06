"""The paper's full encoder-decoder (block 15) and its decoders (block 16)."""
import torch.nn as nn


def shift_right(tgt, bos_id):
    """Decoder input for teacher forcing (Fig. 1, "Outputs (shifted right)").

    tgt: (B, T) int64 target ids. returns (B, T): bos_id, then tgt[:, :-1].
    """
    raise NotImplementedError("Block 15: your turn")


class TransformerSeq2Seq(nn.Module):
    """Encoder-decoder Transformer, Fig. 1 (post-norm, §3.1).

    __init__(src_vocab, tgt_vocab, d_model, n_heads, n_enc_layers, n_dec_layers, d_ff, max_len,
             dropout=0.1, pad_id=0)
        Encoder layer: self-attention + FFN.  Decoder layer: masked self-attention,
        encoder-decoder attention (queries from the decoder, keys/values from the encoder output,
        §3.2.3), FFN. Each sub-layer wrapped in residual + LayerNorm. Sinusoidal positions.
        Embeddings scaled by sqrt(d_model). Output projection to tgt_vocab.
        Sharing embeddings across languages (§3.4) needs a shared vocabulary; optional here.

    encode(src) -> (memory, src_mask)
        src: (B, S) int64, may contain pad_id
        memory: (B, S, d_model); src_mask: bool (B, 1, 1, S), True where src != pad_id
    decode(tgt_in, memory, src_mask) -> logits (B, T, tgt_vocab)
        tgt_in: (B, T) int64 decoder input (already shifted right)
        causal mask on decoder self-attention; src_mask on encoder-decoder attention.
    encode must ALSO apply src_mask in encoder self-attention, so PAD tokens never influence
    real positions (appending PADs to a source must not change the output).
    forward(src, tgt_in) == decode(tgt_in, *encode(src))
    """

    def __init__(self, src_vocab, tgt_vocab, d_model, n_heads, n_enc_layers, n_dec_layers, d_ff, max_len,
                 dropout=0.1, pad_id=0):
        super().__init__()
        raise NotImplementedError("Block 15: your turn")

    def encode(self, src):
        raise NotImplementedError("Block 15: your turn")

    def decode(self, tgt_in, memory, src_mask):
        raise NotImplementedError("Block 15: your turn")

    def forward(self, src, tgt_in):
        raise NotImplementedError("Block 15: your turn")


def greedy_decode(model, src, bos_id, eos_id, max_len):
    """Translate one source sentence greedily.

    model: has encode(src) and decode(tgt_in, memory, src_mask) as in TransformerSeq2Seq
    src: (1, S) int64
    returns: list of generated token ids WITHOUT bos_id, ending with eos_id
             (or stopping after max_len tokens)
    """
    raise NotImplementedError("Block 16: your turn")


def length_penalty(length, alpha):
    """GNMT length penalty used by §6.1 via [38]: ((5 + length) / 6) ** alpha.

    The formula itself is from Wu et al. [38], not from this paper.
    """
    raise NotImplementedError("Block 16: your turn")


def beam_search(model, src, bos_id, eos_id, beam_size=4, alpha=0.6, max_len=50):
    """Beam search (§6.1: beam size 4, alpha = 0.6).

    Keep the beam_size best partial hypotheses by total log-probability; set finished
    hypotheses (ending in eos_id) aside; at the end pick the finished hypothesis with the best
    score = total_log_prob / length_penalty(length, alpha), where length counts the generated
    tokens including eos_id (BOS not counted). Whether finished hypotheses use up beam slots is
    your choice. If nothing finishes within max_len, return the best unfinished hypothesis.
    src: (1, S) int64
    returns: list of token ids WITHOUT bos_id, ending with eos_id (or cut at max_len)
    """
    raise NotImplementedError("Block 16: your turn")

"""Block 16: greedy decoding and beam search for the encoder-decoder (§6.1)."""
import math

from ._util import checkpoint, expect

PAD, BOS, EOS, A, B, C = 0, 1, 2, 3, 4, 5


def _fake_seq2seq(table=None):
    import torch

    # next-token probabilities given the LAST decoder token
    table = table or {
        BOS: {A: 0.5, B: 0.4, EOS: 0.1},
        A: {C: 0.4, B: 0.3, EOS: 0.3},
        B: {EOS: 0.9, C: 0.1},
        C: {EOS: 1.0},
        EOS: {EOS: 1.0},
    }

    class Fake(torch.nn.Module):
        """Greedy picks A then C then EOS (prob 0.2); the best sequence is B EOS (prob 0.36)."""

        def __init__(self):
            super().__init__()
            self.max_len = 50

        def encode(self, src):
            return torch.zeros(src.shape[0], src.shape[1], 4), torch.ones(src.shape[0], 1, 1, src.shape[1], dtype=torch.bool)

        def decode(self, tgt_in, memory, src_mask):
            Bsz, T = tgt_in.shape
            logits = torch.full((Bsz, T, 6), -1e9)
            for b in range(Bsz):
                for t in range(T):
                    for tok, p in table.get(int(tgt_in[b, t]), {EOS: 1.0}).items():
                        logits[b, t, tok] = math.log(p)
            return logits

        def forward(self, src, tgt_in):
            return self.decode(tgt_in, *self.encode(src))

    return Fake()


@checkpoint("16: length_penalty")
def check_length_penalty(length_penalty):
    expect(abs(length_penalty(1, 0.6) - 1.0) < 1e-12, "lp(1) = 1")
    expect(abs(length_penalty(7, 0.6) - 1.51571657) < 1e-6,
           "GNMT form [38]: lp(L) = ((5 + L) / 6)^alpha; lp(7, 0.6) = 2^0.6")
    expect(abs(length_penalty(10, 0.0) - 1.0) < 1e-12, "alpha = 0 switches the penalty off")


@checkpoint("16: greedy_decode and beam_search")
def check_decoders(greedy_decode, beam_search):
    import torch

    m = _fake_seq2seq()
    src = torch.tensor([[7, 8, 9]])
    g = [int(t) for t in greedy_decode(m, src, bos_id=BOS, eos_id=EOS, max_len=10)]
    expect(g == [A, C, EOS], "greedy picks the locally best token each step: A, C, EOS",
           "return the generated tokens without BOS, ending with EOS")
    b1 = [int(t) for t in beam_search(m, src, bos_id=BOS, eos_id=EOS, beam_size=1, alpha=0.0, max_len=10)]
    expect(b1 == g, "beam_size = 1 is greedy decoding")
    b2 = [int(t) for t in beam_search(m, src, bos_id=BOS, eos_id=EOS, beam_size=2, alpha=0.0, max_len=10)]
    expect(b2 == [B, EOS], "beam_size = 2 finds the higher-probability sequence B, EOS (0.36 > 0.2)",
           "keep the beam_size best partial hypotheses by TOTAL log-probability, and keep finished ones aside")

    # A EOS has prob 0.3; B C EOS has prob 0.259 but is longer, so the length penalty flips the winner
    lp_table = {BOS: {A: 0.3, B: 0.7}, A: {EOS: 1.0}, B: {C: 0.37, B: 0.63}, C: {EOS: 1.0}, EOS: {EOS: 1.0}}
    m2 = _fake_seq2seq(lp_table)
    s0 = [int(t) for t in beam_search(m2, src, bos_id=BOS, eos_id=EOS, beam_size=3, alpha=0.0, max_len=10)]
    s1 = [int(t) for t in beam_search(m2, src, bos_id=BOS, eos_id=EOS, beam_size=3, alpha=1.0, max_len=10)]
    expect(s0 == [A, EOS] and s1 == [B, C, EOS],
           "the length penalty matters: alpha = 0 picks A EOS, alpha = 1 picks the longer B C EOS",
           "rank FINISHED hypotheses by total_log_prob / length_penalty(length, alpha)")

# Roadmap

Hours are estimates for focused study, including mistakes. At 5 to 10 hours a week the whole path is roughly
12 to 24 weeks. Each milestone leaves the repo in a state worth showing.

| Block | Topic | Est. hours | Milestone |
|---|---|---|---|
| 00 | Setup, gradient checker | 2 | |
| 01 | Character tokenizer | 2 | |
| 02 | Bigram baseline | 3 | |
| 02b | Character and word n-grams | 4 | |
| 02c | Shannon guessing game, English vs Khmer | 6 | |
| 02d | Smoothing, Kneser-Ney | 6 | |
| 03 | Softmax, cross-entropy by hand | 4 | |
| 04 | MLP language model by hand | 8 | **M1: a neural LM in pure NumPy** (~35 h) |
| 05 | RNN | 3 | |
| 05b | (optional) LSTM cell | 4 | |
| 05c | RNN encoder-decoder with additive attention | 8 | |
| 06 | Scaled dot-product attention | 6 | |
| 07 | Multi-head attention | 6 | |
| 08 | Positional encoding | 3 | |
| 09 | Transformer block | 5 | **M2: every component of Fig. 1, checked** (~70 h) |
| 10 | Tiny GPT | 5 | |
| 11 | Training | 8 | |
| 12 | Decoding | 3 | **M3: a trained tiny GPT that writes Shakespeare-like text** (~86 h). Repo can go public here. |
| 13 | Ablations | 8 | **M4: pre-registered experiments with seeds** (~94 h) |
| 14 | Byte-level BPE, Khmer | 6 | |
| 15 | Encoder-decoder | 8 | |
| 16 | Khmer-English translation | 12 | **M5: the paper's own architecture on a low-resource pair** (~120 h) |

If a deadline arrives early, ship at the last completed milestone and say so in the README. A finished M3 with honest
results reads better than a half-finished M5.

## After this repo
1. Resume LMRC (Li, Chen, Long, Zhang 2024): this repo covers prerequisites P2 to P8 of its prerequisite track.
2. The planned capstone extension.

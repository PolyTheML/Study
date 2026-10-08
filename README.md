# Transformer from scratch: NumPy to a tiny GPT to Khmer-English translation

A from-scratch reimplementation of Vaswani et al. (2017), *Attention Is All You Need*, built one block at a time:
first every mechanism in NumPy with gradients derived by hand, then a tiny decoder-only GPT in PyTorch trained on
Shakespeare, then the paper's full encoder-decoder trained on a small Khmer-English translation task.

> **Status: in progress.** This README is a template. Sections marked *(fill in)* get completed as blocks are finished.

## Why this repo exists
*(fill in, 3-4 sentences in your own words: what you wanted to understand, why build instead of read, why Khmer.)*

## Progress

This repo is Phase 1 of a larger language-modeling history project, from Shannon to today: see [`ROADMAP.md`](ROADMAP.md).

| Block | What | Paper anchor | Status |
|---|---|---|---|
| 00 | Setup, numerical gradient checker | | [x] |
| 01 | Character tokenizer | §5.1 | [x] |
| 02 | Bigram language model (baseline) | | [ ] |
| 02b | Character and word n-grams, n = 1..5; bits/char vs n | outside: Shannon 1948 | [ ] |
| 02c | Shannon guessing game; entropy of English vs Khmer | outside: Shannon 1951 | [ ] |
| 02d | Smoothing: add-one, interpolation, Kneser-Ney | outside: Kneser & Ney 1995 | [ ] |
| 03 | Softmax, logsumexp, cross-entropy (by hand) | §3.4 | [ ] |
| 04 | MLP language model, backprop by hand | §3.4 | [ ] |
| 05 | Vanilla RNN, the sequential bottleneck | §1, Table 1 | [ ] |
| 05b | (optional) LSTM cell | §1 [13] | [ ] |
| 05c | RNN encoder-decoder with additive attention, toy task | §3.2.1 (additive attention [2]) | [ ] |
| 06 | Scaled dot-product attention, causal mask | Eq. 1, §3.2.1, §3.2.3 | [ ] |
| 07 | Multi-head attention | §3.2.2 | [ ] |
| 08 | Sinusoidal positional encoding | §3.5 | [ ] |
| 09 | Transformer block (FFN, residual, LayerNorm) | §3.1, Eq. 2 | [ ] |
| 10 | Tiny GPT | Fig. 1, §3.4 | [ ] |
| 11 | Training (Adam, warmup, label smoothing) | §5.3, Eq. 3, §5.4 | [ ] |
| 12 | Decoding | §6.1 | [ ] |
| 13 | Ablations | Table 3 | [ ] |
| 14 | Byte-level BPE (English and Khmer) | §5.1 | [ ] |
| 15 | Encoder-decoder | Fig. 1, §3.2.3 | [ ] |
| 16 | Khmer-English translation, beam search | §6.1 | [ ] |

## Results *(fill in)*

Character-level language modelling, tiny Shakespeare, validation cross-entropy in nats per character (lower is better):

| Model | Parameters | Val NLL | Notes |
|---|---|---|---|
| Uniform guess | 0 | ln V | reference |
| Bigram counts | | | block 02 |
| MLP, context 3 | | | block 04 |
| Tiny GPT | | | block 11 |

Ablations (block 13), mean and standard deviation over 3 seeds: *(fill in, link each row to its write-up in `experiments/`)*

Translation (block 16): *(fill in: data, size, BLEU and chrF for greedy and beam)*

## Repository layout

```
notebooks/   one workbook per block: concept prompts, specs, my implementations, experiments
src/         graduated code (package `minitransformer`): NumPy ops, tokenizers, PyTorch modules
checks/      behavior checks used by the notebooks (shapes, known values, properties)
tests/       pytest wrappers that run the same checks against src/ (skipped until implemented)
scripts/     command-line training and ablation runners
experiments/ pre-registered hypotheses and results, one file per experiment
reports/     figures and the final technical report
journal/     learning log and mistakes log
docs/        roadmap and study guide
paper/       notes and notation traps for the paper (the PDF is linked, not committed)
data/        datasets (downloaded, not committed)
```

## Reproduce

```bash
pip install -r requirements.txt
pip install -e .
pytest -q                                   # every graduated component, checked
jupyter lab notebooks/                      # the study path, block by block
python scripts/train_gpt.py                 # (after block 11)
```

## What I learned
*(fill in: 3-5 bullets.)* The full record of wrong turns and what each one taught me is in
[`journal/mistakes.md`](journal/mistakes.md).

## How this was built
I studied the paper with Claude (Anthropic) acting as a tutor. The study scaffold (block structure, function
specifications, and the behavior checks in `checks/` and `tests/`) was drafted with Claude. Every implementation in
`notebooks/` and `src/`, every experiment, and every write-up is my own work.

## References
- Vaswani et al. 2017. Attention Is All You Need. NeurIPS. arXiv:1706.03762.
- Further references are cited where used, in the notebooks and in `paper/notes.md`.

## License
MIT, see [LICENSE](LICENSE).

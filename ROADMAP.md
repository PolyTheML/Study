# Roadmap: Language Modeling from Shannon to Today

**Created:** 2026-10-06
**Status:** Phase 1 in progress (repo `transformer-from-scratch`, block 00).
**Pace:** untimed. The learning has no deadline. Applications can go out whenever, using whatever is built by then.
**Endpoints:** two forks, Céline Hudelot (Paris-Saclay) and Kehai Chen (HIT Shenzhen). Both share the trunk until about 2022. Choose the order at Phase 3.

**Accuracy note:** every date and citation below is outside knowledge, written from memory on 2026-10-06. Verify each one against the actual paper when its stop begins, per the protocol's accuracy rules.

---

## The one question

Every model on this road estimates **P(next token | previous tokens)**. The eras differ in two things:
- how much context the model can use;
- how it represents that context: counts, then vectors, then hidden states, then attention.

## Rhythm for every stop

Every stop follows the same protocol loop:
1. One paper.
2. One implementation.
3. One experiment, with the hypothesis and the falsifying result stated before running.
4. Explain it back.

Each stop either adds a row to the leaderboard or reproduces one claim.

## The leaderboard (built during Phase 1)

- **Metric:** bits per character on held-out tiny Shakespeare.
  - Bits = mean NLL in nats / ln 2.
  - The validation split is fixed once, before block 02, and never changed.
- **Columns:** model | year | context used | parameters | training time | validation bits/char.
- **Reference line:** Shannon (1951) estimated English at roughly 0.6 to 1.3 bits per letter, from human guessing (outside knowledge). His estimate used different text and a 27-symbol alphabet, so treat it as a reference point, not a competitor.
- **Trap:** once BPE arrives (block 14), per-token NLL is no longer per-character. Convert as total bits over the text / number of characters, or the comparison is meaningless.

---

## Phase 1: Shannon to Transformer (repo: `transformer-from-scratch`)

New blocks use letter suffixes, so existing block numbers and commits never move.

| Year | Paper (outside knowledge) | Build | Experiment | Block | Status |
|---|---|---|---|---|---|
| 1913 | Markov, letter statistics in *Eugene Onegin* | 2-state vowel/consonant chain | Does the chain predict better than independent letters? | 02 warm-up | optional |
| 1948 | Shannon, "A Mathematical Theory of Communication" | Character and word n-gram generators, n = 1..5; the bits/char harness | Sample quality and validation bits/char vs n. Expect: improves, then collapses as counts get sparse | 02, 02b | [ ] |
| 1951 | Shannon, "Prediction and Entropy of Printed English" | Guessing-game script; entropy bounds from guess-rank counts | English vs Khmer entropy per character, from human guessers | 02c | [ ] |
| 1995 / 1998 | Kneser & Ney; Chen & Goodman (smoothing survey) | Add-one, interpolation, Kneser-Ney | Does Kneser-Ney fix the high-n collapse from 02b? | 02d | [ ] |
| 2003 | Bengio et al., "A Neural Probabilistic Language Model" | MLP language model | First neural row: does it beat Kneser-Ney at the same context length? | 04 | [ ] |
| 2010 | Mikolov et al., RNN language model | Vanilla RNN | Context is unbounded in principle: does it beat the fixed-window MLP? | 05 (now required) | [ ] |
| 1997 | Hochreiter & Schmidhuber, LSTM | LSTM cell | Gated vs vanilla RNN on long-range dependencies | 05b | optional |
| 2014 | Bahdanau, Cho & Bengio; Sutskever et al. (seq2seq) | RNN encoder-decoder with additive attention, on a toy task (e.g. reversing a string) | With vs without attention as input length grows | 05c | [ ] |
| 2013 | Mikolov et al., word2vec | Skip-gram with negative sampling | Analogy and nearest-neighbour probes | side road | optional |
| 2017 | Vaswani et al., "Attention Is All You Need" | Blocks 06 to 16 (see sheets/transformer.md) | Table 3 ablations, footnote 4 | 06-16 | [ ] |

### Notes

- **05c leads into block 06.** AIAYN §3.2.1 compares dot-product attention against additive attention [2], which is Bahdanau. After 05c you will have built both.
- **Khmer trap in 02c: what counts as a "character"?** You can count Unicode code points or grapheme clusters (consonant + subscript + vowel sign). The entropy per unit differs between the two, so choose one, state it, and keep it fixed. Comparing against English needs the same care.
- **02c novelty:** check whether anyone has published a Khmer entropy estimate before treating it as original.

---

## Phase 2: Shared trunk, 2018 to 2022 (new repo when Phase 1 ends)

| # | Year | Paper (outside knowledge) | Build | Experiment |
|---|---|---|---|---|
| 2.1 | 2016 | Sennrich et al., BPE | Covered by block 14 | Covered by block 14 |
| 2.2 | 2018 | Radford et al. (GPT); Devlin et al. (BERT) | Masked-LM head on the block-15 encoder | Masked-LM vs next-token pretraining, probed on a small classification task. Also fills the prereq-track P8 gap. |
| 2.3 | 2020 / 2022 | Kaplan et al. (scaling laws); Hoffmann et al. (Chinchilla) | 5 tiny GPTs across sizes | Fit loss vs parameters; compute-optimal trade-off at toy scale |
| 2.4 | 2020 | Brown et al. (GPT-3, in-context learning) | Few-shot prompting harness on a small pretrained model | Accuracy vs number of shots |
| 2.5 | 2021 | Hu et al. (LoRA) | LoRA from scratch on one layer, then the library version | Shared with LMRC Rung 9 |
| 2.6 | 2022 / 2023 | Ouyang et al. (InstructGPT); Rafailov et al. (DPO) | SFT on a small instruct set; DPO as a laptop-scale stand-in for RLHF (assumption, not what InstructGPT did) | Preference win rate before vs after |
| 2.7 | 2020 | Lewis et al. (RAG) | BM25 and dense retriever | Closed-book vs retrieval-augmented accuracy on a small QA set |
| 2.8 | 2022 | Wei et al. (chain-of-thought); Wang et al. (self-consistency) | Small instruct LLM on a GSM8K subset | Direct vs CoT vs self-consistency |

**LMRC slot:** after 2.5. By then P9, P10, and LoRA are built, so LMRC Rungs 7 to 11 become reachable (see sheets/lmrc.md).

---

## Phase 3: The forks

### Fork A: Céline Hudelot (MICS, CentraleSupélec, Paris-Saclay)
Builds on 2.6 to 2.8.
1. **Dossier:** pick 2 or 3 of her recent papers on LLM training, reasoning, or retrieval. Papers not chosen yet.
2. **Reproduce:** one claim from one paper, at laptop or Colab scale.
3. **Extend:** apply the idea to a Cambodian problem.

### Fork B: Kehai Chen (HIT Shenzhen)
Builds on 2.7 and 2.8.
1. **ReAct** (Yao et al., 2022) and **Toolformer** (Schick et al., 2023): a minimal tool-use loop.
2. **InfCycle** (Kehai Chen et al., ACL Findings 2025): the cycle verifier.
3. **Capstone (your stated plan):** extend the cycle verifier to regulatory citation verification for the IFRS 17 RAG assistant.

**Optional machine-translation side road:** IBM Model 1 (Brown et al., 1993), word alignment via EM. Fits before or after block 16.

---

## Repo plan

- Phase 1 stays in `transformer-from-scratch`. No rename, no restructure.
- Copy this file into that repo as `ROADMAP.md`.
- Phases 2 and 3 go in a new repo when Phase 1 ends. A GitHub profile README links the series together.

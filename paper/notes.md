# Paper notes: Attention Is All You Need (arXiv:1706.03762v7)

PDF: https://arxiv.org/abs/1706.03762 (linked, not committed). The full Paper Understanding Sheet lives in the Claude
project as `sheets/transformer.md`; copy its final version here at the end.

## Notation traps and paper gaps (flagged by the tutor at Pass 1; add your own)
- **Row vectors.** Rows of Q, K, V are positions; a linear layer is x W (Eq. 2). PyTorch's `nn.Linear` stores the
  transpose: the paper's W^Q is `W_q.weight.T`.
- **d_k is per head.** d_k = d_v = d_model / h = 64; d_model = 512 is the total (§3.2.2).
- **Post-norm.** LayerNorm(x + Sublayer(x)) (§3.1). Pre-norm, used by GPT-2 and later, is outside the paper.
- **Mask placement.** Illegal positions are set to -inf in the softmax input (§3.2.3; Fig. 2 "Mask (opt.)").
  Libraries disagree on whether True means keep or block. This repo: True = allowed.
- **Positional index.** In §3.5, i indexes dimension pairs, 0 <= i < d_model / 2.
- **Embedding scale and sharing.** Embeddings are multiplied by sqrt(d_model) and the weight matrix is shared by both
  embedding layers and the pre-softmax linear layer (§3.4). Sharing needs one shared vocabulary (§5.1).
- **Not specified:** the exact label-smoothing distribution (§5.4), biases on Q/K/V/O projections, initialization,
  LayerNorm epsilon. Eq. 3 is undefined at step 0.
- **Inconsistencies:** EN-FR BLEU is 41.8 in the Abstract and Table 2 but 41.0 in §6.1. §5.4 announces three kinds
  of regularization and lists two. §5.1 cites BPE as [3], but the BPE paper is [31].
- **Not in this paper:** the decoder-only GPT (Radford et al. 2018), sampling-based generation, byte-level BPE.

## My notes
*(add as you go: one bullet per insight, with the section number)*

"""Block 02: count-based bigram language model."""
import numpy as np

from ._util import checkpoint, close, expect


@checkpoint("02: bigram LM")
def check_bigram(bigram_counts, bigram_probs, nll, sample_bigram):
    ids = [0, 1, 0, 1, 1]
    C = np.asarray(bigram_counts(ids, 2))
    expect(C.shape == (2, 2), "counts has shape (V, V)")
    expect(np.array_equal(C, [[0, 2], [1, 1]]),
           "counts[i, j] = number of times token i is followed by token j",
           "row = current token, column = next token")
    expect(C.sum() == len(ids) - 1, "total count = len(ids) - 1 transitions")

    P0 = np.asarray(bigram_probs(C, alpha=0.0))
    expect(close(P0, [[0.0, 1.0], [0.5, 0.5]]), "alpha = 0: each row is counts / row sum")
    P1 = np.asarray(bigram_probs(C, alpha=1.0))
    expect(close(P1, [[0.25, 0.75], [0.5, 0.5]]),
           "alpha = 1: add-one smoothing, (count + alpha) / (row sum + alpha * V)")
    expect(close(P1.sum(axis=1), [1.0, 1.0]), "rows sum to 1")

    U = np.full((5, 5), 0.2)
    expect(abs(nll(U, [0, 3, 2, 4, 1, 1]) - np.log(5)) < 1e-9, "uniform model: NLL = ln V")
    expected = -np.mean(np.log([0.75, 0.5, 0.75, 0.5]))
    expect(abs(nll(P1, ids) - expected) < 1e-9,
           "NLL = mean of -log P(next | current) over the len(ids) - 1 transitions",
           "average, do not sum; use natural log")

    s = list(sample_bigram(P1, start_id=0, n=20, rng=np.random.default_rng(0)))
    expect(len(s) == 20 and all(0 <= int(t) < 2 for t in s), "sample_bigram returns n valid ids")
    det = np.array([[0.0, 1.0], [1.0, 0.0]])
    s = [int(t) for t in sample_bigram(det, start_id=0, n=6, rng=np.random.default_rng(1))]
    expect(s == [1, 0, 1, 0, 1, 0], "follows deterministic transitions; start_id is not included in the output",
           "the next id is drawn from the row of the CURRENT token, then it becomes current")

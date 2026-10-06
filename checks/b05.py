"""Block 05: vanilla RNN forward pass."""
import numpy as np

from ._util import checkpoint, close, expect


@checkpoint("05: rnn_forward")
def check_rnn_forward(rnn_forward):
    # scalar case: h_t = tanh(1.0 * x_t + 0.5 * h_(t-1)), h_0 = 0
    Hs = rnn_forward(np.array([[1.0], [0.0], [0.0]]), np.zeros(1), np.array([[1.0]]),
                     np.array([[0.5]]), np.zeros(1))
    expect(np.shape(Hs) == (3, 1), "returns all hidden states, shape (T, H)")
    expect(close(np.ravel(Hs), [0.76159416, 0.36339948, 0.17972621]),
           "known values: the first input echoes forward through Whh, fading each step",
           "h_t uses h_(t-1), and h_0 is the state BEFORE the first input")

    rng = np.random.default_rng(0)
    T, d, H = 6, 3, 4
    X = rng.normal(size=(T, d))
    h0, Wxh, Whh, bh = rng.normal(size=H), rng.normal(size=(d, H)), rng.normal(size=(H, H)) * 0.5, rng.normal(size=H)
    out = rnn_forward(X, h0, Wxh, Whh, bh)
    X2 = X.copy()
    X2[3:] += 10.0
    out2 = rnn_forward(X2, h0, Wxh, Whh, bh)
    expect(close(out[:3], out2[:3]), "h_t depends only on inputs up to t (causal by construction)")
    expect(not close(out[3:], out2[3:]), "later states do react to later inputs")

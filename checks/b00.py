"""Block 00: numerical gradient checker."""
import numpy as np

from ._util import checkpoint, close, expect


@checkpoint("00: numerical_grad")
def check_numerical_grad(numerical_grad):
    rng = np.random.default_rng(0)
    x = rng.normal(size=(3,))
    g = numerical_grad(lambda v: float(np.sum(v**2)), x.copy())
    expect(np.shape(g) == x.shape, "returns an array with the same shape as x",
           "one gradient entry per element of x")
    expect(close(g, 2 * x, atol=1e-5), "d/dx sum(x^2) = 2x",
           "nudge ONE element up and down by eps, keep the others fixed")

    X = rng.normal(size=(2, 3))
    X0 = X.copy()
    G = numerical_grad(lambda v: float(np.sum(np.sin(v))), X)
    expect(np.shape(G) == X0.shape and close(G, np.cos(X0), atol=1e-5),
           "works on 2-D arrays: d/dx sum(sin x) = cos x",
           "loop over every index of a multi-dimensional array, not just range(len(x))")
    expect(np.array_equal(X, X0), "x is unchanged after the call",
           "after nudging an element, restore its original value")

    A = rng.normal(size=(3, 3))
    G = numerical_grad(lambda v: float(v @ A @ v), x.copy())
    expect(close(G, (A + A.T) @ x, atol=1e-4), "d/dx (x^T A x) = (A + A^T) x  (cross terms handled)")

    x3 = np.array([0.7, -1.3, 2.0])
    G = numerical_grad(lambda v: float(np.sum(v**3)), x3.copy(), eps=1e-3)
    expect(close(G, 3 * x3**2, atol=1e-5),
           "accurate even with a coarse eps = 1e-3 on sum(x^3): error ~ eps^2, not ~ eps",
           "a one-sided difference has error proportional to eps; a symmetric one cancels that term")

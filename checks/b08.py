"""Block 08: sinusoidal positional encoding (§3.5)."""
import numpy as np

from ._util import checkpoint, close, expect


@checkpoint("08: sinusoidal_pe")
def check_pe(sinusoidal_pe):
    P = np.asarray(sinusoidal_pe(50, 16))
    expect(P.shape == (50, 16), "shape (max_len, d_model)")
    expect(close(P[0, 0::2], np.zeros(8)) and close(P[0, 1::2], np.ones(8)),
           "position 0: sin(0) = 0 in even dims, cos(0) = 1 in odd dims",
           "even dims (2i) use sin, odd dims (2i+1) use cos")
    expect(close(P[1, :2], [np.sin(1.0), np.cos(1.0)]), "dims 0 and 1 use frequency 1 (wavelength 2*pi)")
    expect(close(P[3, 2:4], [0.81264890, 0.58275361]),
           "pair i = 1 at pos 3: sin(3 / 10000^(2*1/16)), cos(same)",
           "i indexes PAIRS: dims 2i and 2i+1 share the frequency 1 / 10000^(2i / d_model)")
    expect(np.all(np.abs(P) <= 1.0 + 1e-12), "all entries lie in [-1, 1]")

    k = 5
    A, Bm = P[:-k], P[k:]
    M, *_ = np.linalg.lstsq(A, Bm, rcond=None)
    expect(np.max(np.abs(A @ M - Bm)) < 1e-6,
           "§3.5 claim holds: PE[pos + k] is a linear function of PE[pos] (residual ~ 0)")
    expect(abs(P[10] @ P[13] - P[20] @ P[23]) < 1e-9, "PE[p] . PE[q] depends only on the offset q - p")

"""Shared helpers for the checkpoint checks.

Spoiler warning: the files in checks/ contain test logic. Read them only
after your block passes. They test behavior (shapes, known values,
properties), never contain a solution, and are tutor-written.
"""
import functools

import numpy as np


class CheckFailed(AssertionError):
    pass


def ok(msg):
    print(f"  PASS  {msg}")


def fail(msg, hint=None):
    text = msg if hint is None else f"{msg}\n        hint: {hint}"
    raise CheckFailed(text)


def expect(cond, msg, hint=None):
    if not bool(cond):
        fail(msg, hint)
    ok(msg)


def close(a, b, atol=1e-6, rtol=1e-5):
    a = np.asarray(_np(a), dtype=float)
    b = np.asarray(_np(b), dtype=float)
    return a.shape == b.shape and np.allclose(a, b, atol=atol, rtol=rtol)


def same_shape(a, shape):
    return tuple(np.shape(_np(a))) == tuple(shape)


def _np(x):
    """Convert a torch tensor (if any) to a NumPy array; leave everything else alone."""
    if hasattr(x, "detach") and hasattr(x, "cpu"):
        return x.detach().cpu().numpy()
    return x


def _fd_grad(f, x, eps=1e-6):
    """Tutor's own finite-difference helper, used only inside checks."""
    x = np.array(x, dtype=float)
    g = np.zeros_like(x)
    for idx in np.ndindex(x.shape):
        old = x[idx]
        x[idx] = old + eps
        fp = f(x)
        x[idx] = old - eps
        fm = f(x)
        x[idx] = old
        g[idx] = (fp - fm) / (2 * eps)
    return g


def rel_err(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return float(np.max(np.abs(a - b) / np.maximum(1e-8, np.abs(a) + np.abs(b))))


def checkpoint(title):
    """Wrap a check: print a header, turn failures into a clean message.

    Returns True (all passed), False (a check failed) or None (not implemented yet).
    Errors raised by your own code (shape mismatches and so on) are NOT caught,
    so you get the normal traceback for debugging.
    """

    def deco(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            print(f"Checkpoint {title}")
            try:
                fn(*args, **kwargs)
            except NotImplementedError as e:
                print(f"  TODO  not implemented yet ({e})")
                return None
            except CheckFailed as e:
                print(f"  FAIL  {e}")
                print("  Stopped at the first failure. Fix it, re-run your definition cell, then re-run this check.")
                return False
            print("  All checks passed.\n")
            return True

        return wrapper

    return deco

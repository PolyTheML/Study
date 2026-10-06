"""pytest glue: run the notebook checkpoints against the graduated code in src/.

A test is SKIPPED while its function still raises NotImplementedError, so CI stays green
while the repo is in progress and turns each block on as you graduate it.
"""
import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))


def run(check, *args):
    try:
        result = check(*args)
    except NotImplementedError as e:  # raised while building a model inside a check
        pytest.skip(f"not implemented yet: {e}")
    if result is None:
        pytest.skip("not implemented yet")
    assert result is True, "checkpoint failed; see the printed FAIL line above"

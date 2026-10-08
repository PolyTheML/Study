"""Where the files that link the notebooks live.

Notebooks never share variables, so each block hands its results to the next one through a file:

    block 00  downloads  data/tinyshakespeare.txt
    block 01  writes     data/train_ids.npy, data/val_ids.npy
    block 11  writes     checkpoints/            (git-ignored)
    blocks 02, 04, 11, 13  read the ids;  block 12 reads the checkpoint

Import the names below instead of typing a path. They are anchored to the repo root, so they work
whatever directory the kernel starts in (VS Code starts it in notebooks/, Colab in the repo root).
"""
import pathlib
import urllib.request

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
CHECKPOINTS = ROOT / "checkpoints"

TEXT_PATH = DATA / "tinyshakespeare.txt"
TRAIN_IDS_PATH = DATA / "train_ids.npy"
VAL_IDS_PATH = DATA / "val_ids.npy"

TEXT_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"


def ensure_text(path=TEXT_PATH, url=TEXT_URL):
    """Return the path to Tiny Shakespeare, downloading it first if it is missing."""
    path = pathlib.Path(path)
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        partial = path.with_name(path.name + ".part")
        try:
            urllib.request.urlretrieve(url, partial)
            partial.replace(path)  # a half-finished download never looks like a finished one
        finally:
            partial.unlink(missing_ok=True)
    return path


def load_text(path=TEXT_PATH):
    """The whole corpus as one string."""
    return ensure_text(path).read_text(encoding="utf-8")


def load_ids(train_path=TRAIN_IDS_PATH, val_path=VAL_IDS_PATH):
    """(train_ids, val_ids) as saved by notebook 01."""
    missing = [p for p in (pathlib.Path(train_path), pathlib.Path(val_path)) if not p.exists()]
    if missing:
        raise FileNotFoundError(
            f"{', '.join(p.name for p in missing)} not found in {missing[0].parent}. "
            "Run notebook 01 first: its 'Apply it' cell saves them."
        )
    return np.load(train_path), np.load(val_path)

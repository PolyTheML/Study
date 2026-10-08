"""The file handoff between notebooks (src/minitransformer/paths.py). No network needed."""
import numpy as np
import pytest

from minitransformer import paths


def fake_download(content=b"hello"):
    calls = []

    def urlretrieve(url, filename):
        calls.append(url)
        filename.write_bytes(content)

    return calls, urlretrieve


def test_paths_are_anchored_to_the_repo_not_the_working_directory():
    assert (paths.ROOT / "pyproject.toml").exists()
    for p in (paths.TEXT_PATH, paths.TRAIN_IDS_PATH, paths.VAL_IDS_PATH):
        assert p.parent == paths.DATA == paths.ROOT / "data"
    assert paths.CHECKPOINTS == paths.ROOT / "checkpoints"


def test_data_readme_documents_every_file_in_the_chain():
    readme = (paths.ROOT / "data" / "README.md").read_text(encoding="utf-8")
    for p in (paths.TEXT_PATH, paths.TRAIN_IDS_PATH, paths.VAL_IDS_PATH):
        assert p.name in readme


def test_ensure_text_downloads_once(tmp_path, monkeypatch):
    calls, urlretrieve = fake_download(b"to be or not to be")
    monkeypatch.setattr(paths.urllib.request, "urlretrieve", urlretrieve)
    target = tmp_path / "nested" / "text.txt"

    assert paths.ensure_text(target, url="http://example.invalid/x") == target
    assert paths.ensure_text(target, url="http://example.invalid/x") == target

    assert calls == ["http://example.invalid/x"]
    assert target.read_bytes() == b"to be or not to be"
    assert [p.name for p in target.parent.iterdir()] == ["text.txt"]


def test_failed_download_leaves_nothing_behind(tmp_path, monkeypatch):
    def urlretrieve(url, filename):
        filename.write_bytes(b"half")
        raise OSError("connection dropped")

    monkeypatch.setattr(paths.urllib.request, "urlretrieve", urlretrieve)
    target = tmp_path / "text.txt"

    with pytest.raises(OSError):
        paths.ensure_text(target)

    assert list(tmp_path.iterdir()) == []


def test_load_text_decodes_utf8(tmp_path):
    target = tmp_path / "text.txt"
    target.write_text("សួស្តី", encoding="utf-8")
    assert paths.load_text(target) == "សួស្តី"


def test_load_ids_round_trip(tmp_path):
    train, val = np.arange(9), np.arange(3)
    np.save(tmp_path / "train.npy", train)
    np.save(tmp_path / "val.npy", val)
    got_train, got_val = paths.load_ids(tmp_path / "train.npy", tmp_path / "val.npy")
    assert np.array_equal(got_train, train) and np.array_equal(got_val, val)


def test_load_ids_before_notebook_01_says_what_to_do(tmp_path):
    with pytest.raises(FileNotFoundError, match="Run notebook 01"):
        paths.load_ids(tmp_path / "train.npy", tmp_path / "val.npy")

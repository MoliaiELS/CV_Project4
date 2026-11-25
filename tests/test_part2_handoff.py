import hashlib
import json

import pytest

from scripts import part2_handoff as handoff


def test_verify_detects_same_size_corruption(tmp_path, monkeypatch):
    assets = tmp_path / "assets"
    assets.mkdir()
    checkpoint = assets / "checkpoint.pt"
    checkpoint.write_bytes(b"original")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"checkpoint_count": 1, "files": [{
        "relative_path": "checkpoint.pt", "bytes": 8,
        "sha256": hashlib.sha256(b"original").hexdigest()
    }]}))
    monkeypatch.setattr(handoff, "MANIFEST", manifest)
    handoff.verify(assets, full_hash=True)
    checkpoint.write_bytes(b"modified")
    with pytest.raises(RuntimeError, match="SHA256 differs"):
        handoff.verify(assets, full_hash=True)


def test_verify_detects_missing_checkpoint(tmp_path, monkeypatch):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"checkpoint_count": 1, "files": [{
        "relative_path": "absent.pt", "bytes": 8, "sha256": "irrelevant"
    }]}))
    monkeypatch.setattr(handoff, "MANIFEST", manifest)
    with pytest.raises(RuntimeError, match="missing/unreadable"):
        handoff.verify(tmp_path)

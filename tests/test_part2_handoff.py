from pathlib import Path
import hashlib
import json

import pytest

from scripts import part2_handoff as handoff


def test_prepare_refuses_shared_or_release_descendants(tmp_path):
    assets = tmp_path / "shared"
    release = tmp_path / "release"
    for destination in [assets, assets / "new", release / "new"]:
        with pytest.raises(ValueError):
            handoff.validate_destination(destination, assets, release)


def test_prepare_refuses_existing_workspace(tmp_path):
    destination = tmp_path / "existing"
    destination.mkdir()
    (destination / "user_file").write_text("keep")
    with pytest.raises(FileExistsError):
        handoff.validate_destination(destination, tmp_path / "assets", tmp_path / "release")
    assert (destination / "user_file").read_text() == "keep"


def test_prepare_accepts_fresh_sibling(tmp_path):
    destination = tmp_path / "fresh"
    assert handoff.validate_destination(destination, tmp_path / "assets", tmp_path / "release") == destination


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


def test_direct_commands_remove_proxy_environment(monkeypatch):
    monkeypatch.setenv("https_proxy", "http://example.invalid:1234")
    monkeypatch.setenv("ALL_PROXY", "socks5://example.invalid:1234")
    monkeypatch.setenv("TASK_SAFE_SETTING", "preserved")
    env = handoff.direct_env()
    assert not any(k.lower().endswith("_proxy") for k in env)
    assert env["TASK_SAFE_SETTING"] == "preserved"

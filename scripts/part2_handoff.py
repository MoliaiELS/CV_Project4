"""Verify the externally stored Part 2 assets used in the report."""
import argparse
import hashlib
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
FROZEN_REV = "bd2e3ecae209df5de315fcddb258accf15a57592"
FROZEN_SHA = "c8b6722a8b18deaecd6766279d8e69617e762ec9cb9add52300ba171424bf51c"
MANIFEST = REPO / "results/part2_formal_7000/artifact_hashes.json"


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load_manifest():
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


def verify(assets, full_hash=False):
    manifest = load_manifest()
    failures = []
    for record in manifest["files"]:
        path = assets / record["relative_path"]
        if not path.is_file():
            failures.append(record["relative_path"] + ": missing/unreadable")
        elif path.stat().st_size != record["bytes"]:
            failures.append(record["relative_path"] + ": size differs")
        elif full_hash and digest(path) != record["sha256"]:
            failures.append(record["relative_path"] + ": SHA256 differs")
    if failures:
        raise RuntimeError("\n".join(failures))
    print(json.dumps({"checked_files": len(manifest["files"]), "sha256_checked": full_hash,
                      "checkpoint_count": manifest["checkpoint_count"], "assets_root": str(assets)}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["verify"])
    parser.add_argument("--assets-root", type=Path, required=True)
    parser.add_argument("--sha256", action="store_true", help="Hash all archived assets")
    args = parser.parse_args()
    verify(args.assets_root.resolve(), args.sha256)


if __name__ == "__main__":
    main()

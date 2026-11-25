"""Prepare the three verified scenes with the checked-in fixed split lists."""
import argparse
from pathlib import Path
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from p4.common import sha256_file
from p4.data import load_manifest, prepare_dataset


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", default="data/raw/project4_3d")
    parser.add_argument("--output", default="data/prepared")
    parser.add_argument("--split-dir", default="configs/splits")
    parser.add_argument("--max-size", type=int, default=960)
    args = parser.parse_args()
    mapping = {"Waymo-405841": "405841/FRONT", "Dl3dv-2": "DL3DV-2", "Re10k-1": "Re10k-1"}
    for scene, relative in mapping.items():
        output = Path(args.output) / scene
        split_file = Path(args.split_dir) / (scene + ".json")
        frozen = json.loads(split_file.read_text(encoding="utf-8"))
        if (output / "manifest.json").exists():
            manifest = load_manifest(output)
            actual = {kind: [f["id"] for f in manifest["frames"] if f["split"] == kind] for kind in ("train", "test")}
            if actual != frozen or manifest["max_size"] != args.max_size:
                raise ValueError(f"Existing {scene} differs from frozen split/resolution; choose new output")
            if any(sha256_file(output / f["image"]) != f["sha256"] for f in manifest["frames"]):
                raise ValueError(f"Existing {scene} images changed")
        else:
            manifest = prepare_dataset(Path(args.raw) / relative, output, scene,
                                       split_file=split_file, max_size=args.max_size, reference_kind="reference")
        print(json.dumps({"scene": scene, "train": len(frozen["train"]), "test": len(frozen["test"]),
                          "split_hash": manifest["split_hash"]}), flush=True)


if __name__ == "__main__":
    main()

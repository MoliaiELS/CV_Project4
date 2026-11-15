"""Part 2 sparse manifests: sample original frame positions, retain held-out test."""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import numpy as np
from p4.common import sha256_file, write_json
from p4.data import load_manifest, natural_key

STRIDES = {"Waymo-405841": 10, "Dl3dv-2": 30, "Re10k-1": 30}

def prepare_sparse(dataset, output, stride, offset=1):
    dataset, output = Path(dataset).resolve(), Path(output).resolve()
    if stride < 2 or not 0 <= offset < stride:
        raise ValueError("Require stride >= 2 and 0 <= offset < stride")
    if output == dataset or output.is_relative_to(dataset) or dataset.is_relative_to(output):
        raise ValueError("Sparse output must be separate from dense prepared data")
    if output.exists():
        raise FileExistsError(f"Preserve existing dataset: {output}")
    parent = load_manifest(dataset)
    frames = parent["frames"]
    if [f["id"] for f in frames] != sorted((f["id"] for f in frames), key=natural_key):
        raise ValueError("Parent frames must be in natural original sequence order")
    candidates = list(range(offset, len(frames), stride))
    collision = [frames[i]["id"] for i in candidates if frames[i]["split"] != "train"]
    if collision:
        raise ValueError(f"Sparse sampling overlaps held-out test; choose a documented offset: {collision}")
    selected = {frames[i]["id"] for i in candidates}
    if len(selected) < 3:
        raise ValueError("At least three sparse cameras are needed for evaluation alignment")
    kept = [copy.deepcopy(f) for f in frames if f["split"] == "test" or f["id"] in selected]
    if not any(f["split"] == "test" for f in kept):
        raise ValueError("Need nonempty unchanged test set")
    for frame in kept:
        if sha256_file(dataset / frame["image"]) != frame["sha256"]:
            raise ValueError(f"Prepared RGB changed: {frame['image']}")
    reference = None
    if (dataset / "reference.npz").exists():
        with np.load(dataset / "reference.npz", allow_pickle=False) as z:
            lookup = {str(n): i for i, n in enumerate(z["names"])}
            indices = [lookup[f["image"]] for f in kept]
            reference = {k: z[k][indices].copy() for k in ("names", "c2w", "K")}
    manifest = copy.deepcopy(parent)
    manifest["frames"] = kept
    manifest["reference_poses"] = len(kept) if reference is not None else 0
    manifest["split_source"] = {
        "kind": "part2_original_sequence_stride",
        "parent_split_hash": parent["split_hash"],
        "parent_manifest_sha256": sha256_file(dataset / "manifest.json"),
        "stride": stride, "offset": offset, "selected_original_positions": candidates,
        "selected_original_frame_ids": [frames[i]["id"] for i in candidates],
        "source_frames": len(frames),
        "source_train_frames": sum(f["split"] == "train" for f in frames),
        "excluded_train_ids": [f["id"] for f in frames if f["split"] == "train" and f["id"] not in selected],
        "test_policy": "Exact original held-out frames; unsampled train frames excluded, not relabelled test",
        "phase_provenance": "Group-defined offset=1; first frame is held out. Course stride, not claimed exact BRPO split.",
        "uses_reference_to_select_frames": False,
    }
    fingerprint = {k: manifest[k] for k in ("schema_version", "scene", "camera_convention",
                                           "max_size", "preprocessing", "frames", "split_source")}
    manifest["split_hash"] = hashlib.sha256(json.dumps(fingerprint, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    output.mkdir(parents=True)
    for frame in kept:
        target = output / frame["image"]
        target.parent.mkdir(parents=True, exist_ok=True)
        # Hard links share the frozen canonical bytes, not dense geometry/cameras.
        os.link(dataset / frame["image"], target)
    if reference is not None:
        np.savez_compressed(output / "reference.npz", **reference)
    write_json(output / "manifest.json", manifest)
    return manifest

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--stride", required=True, type=int)
    p.add_argument("--offset", default=1, type=int)
    a = p.parse_args()
    m = prepare_sparse(a.dataset, a.output, a.stride, a.offset)
    print(json.dumps({"scene": m["scene"], "train": sum(f["split"]=="train" for f in m["frames"]),
                      "test": sum(f["split"]=="test" for f in m["frames"]), "split_hash": m["split_hash"],
                      "train_ids":m["split_source"]["selected_original_frame_ids"]}, indent=2))

if __name__ == "__main__":
    main()

"""Prepare the Mip-NeRF 360 Bonsai sparse benchmark from released inputs."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
from shutil import copyfile
from tempfile import TemporaryDirectory
from urllib.error import URLError
from urllib.request import urlopen

from PIL import Image

from p4.common import sha256_file, write_json
from p4.data import IMAGE_EXTENSIONS, natural_key, prepare_dataset, read_colmap_model


SCENE = "Bonsai"
OFFICIAL_PAGE = "https://jonbarron.info/mipnerf360/"
OFFICIAL_ARCHIVE = "https://storage.googleapis.com/gresearch/refraw360/bonsai.zip"
MIRROR_REPO = "alexmkwizu/gaussian_training_datasets"
MIRROR_REVISION = "f63877dbf97e58cbf9f9ad15d0aae7548e502098"
MIRROR_SCENE = "mipnerf360/bonsai"
CAMERA_FILES = ("cameras.bin", "images.bin")
SOURCE_VIEWS = 292
TRAIN_VIEWS = 12
TEST_STRIDE = 8
# SHA-256 of the canonical selected-file manifest, after matching each payload
# against the official ZIP member's uncompressed byte length and CRC-32.
OFFICIAL_SELECTION_HASHES = {
    4: "d06fccfd9048edd5a5a29d03e3e4b46cd1b4adb98da4fe43ee28a5ec37559022",
    8: "d97fbbbfb87f1ce3da20b8e5fd5c4b2475b6a6202a0cf3061967e36832b1387a",
}


def fetch(url):
    for attempt in range(3):
        try:
            with urlopen(url, timeout=75) as response:
                return response.read()
        except (URLError, TimeoutError):
            if attempt == 2:
                raise


def matching_blob(data, entry):
    if len(data) != entry["size"]:
        return False
    if "lfs" in entry:
        return hashlib.sha256(data).hexdigest() == entry["lfs"]["oid"]
    digest = hashlib.sha1(f"blob {len(data)}\0".encode("ascii"))
    digest.update(data)
    return digest.hexdigest() == entry["oid"]


def listing(endpoint, folder):
    base = f"{endpoint}/api/datasets/{MIRROR_REPO}/tree/{MIRROR_REVISION}/{MIRROR_SCENE}"
    return json.loads(fetch(f"{base}/{folder}?limit=1000"))


def cache_file(endpoint, cache, entry):
    prefix = f"{MIRROR_SCENE}/"
    relative = Path(entry["path"].removeprefix(prefix))
    if not entry["path"].startswith(prefix) or relative.is_absolute() or relative.drive or ".." in relative.parts:
        raise ValueError(f"Invalid mirror member path: {entry['path']}")
    path = cache / relative
    data = path.read_bytes() if path.is_file() else b""
    if not matching_blob(data, entry):
        url = f"{endpoint}/datasets/{MIRROR_REPO}/resolve/{MIRROR_REVISION}/{entry['path']}"
        data = fetch(url)
        if not matching_blob(data, entry):
            raise ValueError(f"Mirror bytes differ from pinned revision for {relative}")
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_bytes(data)
        temporary.replace(path)
    return relative.as_posix(), {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def frozen_split(names):
    ordered = sorted(names, key=natural_key)
    if len(ordered) != SOURCE_VIEWS or len(set(ordered)) != SOURCE_VIEWS:
        raise ValueError(f"Expected {SOURCE_VIEWS} unique source photographs")
    test = ordered[::TEST_STRIDE]
    candidates = [name for index, name in enumerate(ordered) if index % TEST_STRIDE]
    train = [candidates[round(i * (len(candidates) - 1) / (TRAIN_VIEWS - 1))]
             for i in range(TRAIN_VIEWS)]
    return ordered, {"train": train, "test": test}


def scaled_camera_transforms(cache, ordered, selection):
    cameras, registered = read_colmap_model(cache / "sparse/0")
    if {Path(name).name for name in ordered} != registered.keys():
        raise ValueError("Released COLMAP registration and the complete RGB listing differ")
    frames = []
    for name in selection:
        record = registered[Path(name).name]
        camera = cameras[record["camera_id"]]
        if camera["model"] != "PINHOLE" or camera["distortion"].size:
            raise ValueError(f"Unexpected released camera model for {name}")
        with Image.open(cache / name) as image:
            width, height = image.size
        sx = width / camera["width"]
        sy = height / camera["height"]
        K = camera["K"]
        frames.append({"file_path": name, "w": width, "h": height,
                       "fl_x": float(sx * K[0, 0]), "fl_y": float(sy * K[1, 1]),
                       "cx": float(sx * K[0, 2]), "cy": float(sy * K[1, 2]),
                       "transform_matrix": record["c2w"].tolist()})
    return {"camera_convention": "opencv", "camera_model": "PINHOLE", "frames": frames}


def prepare(cache, output, endpoint, level=4, max_size=512):
    cache, output = Path(cache).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError(f"Choose a new prepared output directory: {output}")
    image_folder = f"images_{level}"
    prefix = f"{MIRROR_SCENE}/"
    image_entries = {entry["path"].removeprefix(prefix): entry
                     for entry in listing(endpoint, image_folder)
                     if entry["type"] == "file" and Path(entry["path"]).suffix.lower() in IMAGE_EXTENSIONS}
    ordered, split = frozen_split(image_entries)
    selection = sorted(split["train"] + split["test"], key=natural_key)
    cameras = {Path(entry["path"]).name: entry for entry in listing(endpoint, "sparse/0")}
    required = [image_entries[name] for name in selection] + [cameras[name] for name in CAMERA_FILES]
    print(f"Staging {len(split['train'])} training views, {len(split['test'])} test views, and released cameras", flush=True)
    with ThreadPoolExecutor(max_workers=8) as executor:
        inputs = dict(executor.map(lambda entry: cache_file(endpoint, cache, entry), required))
    canonical_inputs = json.dumps(inputs, sort_keys=True, separators=(",", ":"))
    source_hash = hashlib.sha256(canonical_inputs.encode("utf-8")).hexdigest()
    if source_hash != OFFICIAL_SELECTION_HASHES[level]:
        raise ValueError("The selected source files differ from the verified official release")

    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output.parent, prefix=".bonsai-") as temporary:
        source = Path(temporary) / "source"
        for name in selection:
            target = source / name
            target.parent.mkdir(parents=True, exist_ok=True)
            copyfile(cache / name, target)
        write_json(source / "transforms.json", scaled_camera_transforms(cache, ordered, selection))
        split_file = source / "split.json"
        split_file.write_text(json.dumps(split, indent=2), encoding="utf-8", newline="\n")
        prepared = Path(temporary) / "prepared"
        manifest = prepare_dataset(source, prepared, SCENE, split_file=split_file,
                                   max_size=max_size, reference_kind="reference")
        if manifest["reference_poses"] != len(selection):
            raise RuntimeError("Exact released poses are required for all selected views")
        manifest["source"] = f"https://huggingface.co/datasets/{MIRROR_REPO}/tree/{MIRROR_REVISION}/{MIRROR_SCENE}"
        manifest["split_source"]["file"] = "split.json"
        write_json(prepared / "manifest.json", manifest)
        copyfile(split_file, prepared / "split.json")
        write_json(prepared / "source_receipt.json", {
            "dataset": "Mip-NeRF 360", "scene": SCENE,
            "official_project": OFFICIAL_PAGE, "official_archive": OFFICIAL_ARCHIVE,
            "official_archive_generation": "1704838090071815",
            "official_selection_sha256": source_hash,
            "input_mirror_repository": MIRROR_REPO, "input_mirror_revision": MIRROR_REVISION,
            "input_mirror_endpoint": endpoint, "input_mirror_scene": MIRROR_SCENE,
            "source_level": image_folder, "source_files": inputs,
            "source_natural_order": ordered,
            "split_rule": "Every eighth natural filename starting at index zero is held out. Select twelve evenly spaced ranks from the remaining names, rounded to the nearest integer.",
            "split_sha256": sha256_file(prepared / "split.json"), "prepared_split_hash": manifest["split_hash"],
            "reference_sha256": sha256_file(prepared / "reference.npz"),
            "reference_policy": "Released COLMAP calibration is evaluation-only. Rescale K rows by independent actual image-width and image-height ratios to the downloaded level, then apply canonical resize ratios. Preserve the released camera-to-world transforms.",
            "unused_non_test_views": SOURCE_VIEWS - len(selection),
        })
        prepared.rename(output)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-cache", type=Path, default=Path("data/raw/mipnerf360-bonsai"))
    parser.add_argument("--output", type=Path, default=Path("data/prepared/Bonsai"))
    parser.add_argument("--endpoint", default=os.environ.get("HF_ENDPOINT", "https://huggingface.co"))
    parser.add_argument("--level", type=int, choices=(4, 8), default=4)
    parser.add_argument("--max-size", type=int, default=512)
    args = parser.parse_args()
    manifest = prepare(args.source_cache, args.output, args.endpoint.rstrip("/"), args.level, args.max_size)
    print(json.dumps({"scene": SCENE, "output": str(args.output.resolve()),
                      "frames": len(manifest["frames"]), "split_hash": manifest["split_hash"]}), flush=True)


if __name__ == "__main__":
    main()

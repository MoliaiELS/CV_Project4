"""Initialize sparse 3DGS from verified training RGB using SuperPoint and LightGlue."""

from __future__ import annotations

import argparse
from itertools import combinations
from pathlib import Path
import time

import numpy as np
from PIL import Image

from p4.common import sha256_file, source_version, write_json
from p4.data import load_manifest, load_scene
from scripts.init_pycolmap_sparse import (PYCOLMAP_VERSION, SEED, THREADS,
                                            map_training_views, stage_train_images, verified_pairs)


MODEL_ID = "ETH-CVG/lightglue_superpoint"
MODEL_REVISION = "5f5f626efee99f37dbc9eafea879d24d297aeab8"
TRANSFORMERS_VERSION = "4.57.2"
TRANSFORMERS_REVISION = "12ceb06b27d8f343cf775ed680964e695c2b20fc"
MODEL_ASSETS = {
    "config.json": (898, "adea6db482a567b4f01c342b5771f18a0cdcdef047a44974f22e3bf490714471"),
    "preprocessor_config.json": (234, "481e9a7a4b07cf0230870bf7f5cec1eda3303248e13e2a56bee94825448ff8e0"),
    "model.safetensors": (55016112, "320864fabf972cd00ce314e889cf89039e44c238e08d74d2b42d9cf2e80fc867"),
}
MATCH_EDGE = 512
MAX_KEYPOINTS = 1024
MIN_INLIERS = 15
VERIFY_ERROR_PX = 2.0


def verify_model_assets(model_dir):
    for name, (size, digest) in MODEL_ASSETS.items():
        path = model_dir / name
        if not path.is_file() or path.stat().st_size != size or sha256_file(path) != digest:
            raise ValueError(f"Invalid or missing pinned LightGlue release file: {path}")


def colmap_keypoints(normalized, model_size, source_size):
    """Transform SuperPoint zero-center indices to COLMAP half-pixel coordinates."""
    normalized = np.asarray(normalized, dtype=np.float64)
    if normalized.ndim != 2 or normalized.shape[1] != 2 or not np.isfinite(normalized).all():
        raise ValueError("Expected finite normalized Nx2 SuperPoint keypoints")
    model_size, source_size = np.asarray(model_size), np.asarray(source_size)
    pixels = (normalized * model_size + .5) * (source_size / model_size)
    if np.any(pixels < 0) or np.any(pixels >= source_size):
        raise ValueError("SuperPoint keypoint lies outside the canonical image")
    return np.ascontiguousarray(pixels, dtype=np.float32)


def mutual_pairs(left_to_right, right_to_left, n_left, n_right):
    """Keep valid one-to-one detector indices from the two LightGlue match maps."""
    left_to_right, right_to_left = np.asarray(left_to_right), np.asarray(right_to_left)
    if (left_to_right.ndim != 1 or right_to_left.ndim != 1
            or len(left_to_right) < n_left or len(right_to_left) < n_right):
        raise ValueError("Expected one match-index vector per image")
    left = np.flatnonzero((left_to_right[:n_left] >= 0) & (left_to_right[:n_left] < n_right))
    right = left_to_right[left].astype(np.intp)
    mutual = right_to_left[right] == left
    return np.ascontiguousarray(np.column_stack((left[mutual], right[mutual])), dtype=np.uint32)


def _extract_features(model, processor, image_root, names, model_size, device):
    features = []
    for name in names:
        with Image.open(image_root / name) as disk_image:
            image = disk_image.convert("RGB")
        processed = processor([image, image], size={"width": model_size[0], "height": model_size[1]},
                              return_tensors="pt")["pixel_values"][:, 0].to(device)
        points, _, descriptors, valid = model.keypoint_detector(processed)[:4]
        keep = valid[0].bool()
        features.append((points[0, keep], descriptors[0, keep]))
    return features


def _match_pair(model, left, right, model_size):
    import torch
    from torch.nn.utils.rnn import pad_sequence

    if min(len(left[0]), len(right[0])) < MIN_INLIERS:
        return np.empty((0, 2), dtype=np.uint32)
    points = pad_sequence((left[0], right[0]), batch_first=True)
    descriptors = pad_sequence((left[1], right[1]), batch_first=True)
    counts = torch.tensor([len(left[0]), len(right[0])], device=points.device)
    valid = torch.arange(points.shape[1], device=points.device)[None, :] < counts[:, None]
    absolute = points * points.new_tensor(model_size)
    width, height = model_size
    matches = model._match_image_pair(absolute[None], descriptors[None], height, width, mask=valid[None],
                                      output_attentions=False, output_hidden_states=False)[0]
    forward, reverse = matches.reshape(2, -1).cpu().numpy()
    return mutual_pairs(forward, reverse, len(left[0]), len(right[0]))


def _populate_database(pycolmap, database_path, image_root, names, features, keypoints, match):
    options = pycolmap.ImageReaderOptions(camera_model="PINHOLE")
    pycolmap.import_images(str(database_path), str(image_root), pycolmap.CameraMode.SINGLE, names, options)
    database = pycolmap.Database()
    database.open(str(database_path))
    try:
        images = {image.name: image for image in database.read_all_images()}
        if set(images) != set(names):
            raise ValueError("LightGlue database must contain exactly the staged training images")
        ordered = [images[name] for name in names]
        cameras = [database.read_camera(image.camera_id) for image in ordered]
        for image, points in zip(ordered, keypoints, strict=True):
            database.write_keypoints(image.image_id, points)
        verification = pycolmap.TwoViewGeometryOptions(min_num_inliers=MIN_INLIERS)
        verification.ransac.max_error = VERIFY_ERROR_PX
        pairs = []
        for i, j in combinations(range(len(names)), 2):
            matches = match(features[i], features[j])
            left_id, right_id = ordered[i].image_id, ordered[j].image_id
            found, verified, config = len(matches), 0, None
            if found:
                database.write_matches(left_id, right_id, matches)
            if found >= MIN_INLIERS:
                geometry = pycolmap.estimate_two_view_geometry(cameras[i], keypoints[i].astype(np.float64), cameras[j],
                                                             keypoints[j].astype(np.float64), matches, verification)
                verified, config = len(geometry.inlier_matches), int(geometry.config)
                if verified >= MIN_INLIERS:
                    database.write_two_view_geometry(left_id, right_id, geometry)
            pairs.append({"names": [names[i], names[j]], "matches": found,
                          "inliers": verified, "geometry_config": config})
        return pairs
    finally:
        database.close()


def initialize(dataset, output, model_dir, device="cpu", match_edge=MATCH_EDGE):
    import pycolmap
    import torch
    import transformers
    from transformers import AutoImageProcessor, AutoModelForKeypointMatching

    if pycolmap.__version__ != PYCOLMAP_VERSION or transformers.__version__ != TRANSFORMERS_VERSION:
        raise ValueError(f"Expected PyCOLMAP {PYCOLMAP_VERSION} and Transformers {TRANSFORMERS_VERSION}")
    if match_edge < 64 or match_edge % 8:
        raise ValueError("The matching edge must be a multiple of eight and at least 64 pixels")
    dataset, output, model_dir = (Path(path).resolve() for path in (dataset, output, model_dir))
    verify_model_assets(model_dir)
    manifest_path = dataset / "manifest.json"
    manifest_sha256 = sha256_file(manifest_path)
    manifest = load_manifest(dataset)
    if manifest.get("camera_convention") != "opencv_c2w":
        raise ValueError("Expected the project's OpenCV camera convention")
    frames = [frame for frame in manifest["frames"] if frame["split"] == "train"]
    names = [frame["image"] for frame in frames]
    output.mkdir(parents=True)
    image_root, database_path = output / "train_rgb", output / "database.db"
    size = stage_train_images(dataset, image_root, frames)
    model_size = tuple(max(8, 8 * round(side * match_edge / (8 * max(size)))) for side in size)
    receipt = {**source_version(), "initializer_script_sha256": sha256_file(__file__), "schema_version": 1,
               "scene": manifest["scene"], "method": "lightglue_pycolmap", "pycolmap_version": PYCOLMAP_VERSION,
               "transformers_version": TRANSFORMERS_VERSION, "transformers_revision": TRANSFORMERS_REVISION,
               "model_id": MODEL_ID, "model_revision": MODEL_REVISION, "model_files": MODEL_ASSETS,
               "manifest_sha256": manifest_sha256, "split_hash": manifest["split_hash"], "train_images": names,
               "train_only_inputs": True, "reference_data_used": False, "camera_convention": "opencv_c2w",
               "camera_model": "shared PINHOLE", "canonical_image_wh": list(size), "matching_image_wh": list(model_size),
               "matcher": "exhaustive LightGlue on fixed per-image SuperPoint IDs",
               "adaptive_matching": False,
               "matching_device": device, "torch_version": torch.__version__, "num_threads": THREADS,
               "seed": SEED, "max_keypoints": MAX_KEYPOINTS, "geometric_verification_max_error_px": VERIFY_ERROR_PX,
               "geometric_verification_min_inliers": MIN_INLIERS, "bundle_adjustment": True,
               "mapping_attempts": [], "status": "running"}
    write_json(output / "scene.json", receipt)
    try:
        pycolmap.set_random_seed(SEED)
        torch.set_num_threads(THREADS)
        torch.manual_seed(SEED)
        started = time.perf_counter()
        processor = AutoImageProcessor.from_pretrained(model_dir, local_files_only=True, use_fast=False)
        model = AutoModelForKeypointMatching.from_pretrained(model_dir, local_files_only=True, use_safetensors=True)
        model = model.eval().to(device)
        model.keypoint_detector.config.max_keypoints = MAX_KEYPOINTS
        model.depth_confidence = model.width_confidence = -1
        receipt["matcher_load_seconds"] = time.perf_counter() - started
        started = time.perf_counter()
        with torch.inference_mode():
            features = _extract_features(model, processor, image_root, names, model_size, device)
            keypoints = [colmap_keypoints(points.cpu().numpy(), model_size, size) for points, _ in features]
            receipt.update(feature_extraction_seconds=time.perf_counter() - started,
                           keypoints_per_train_image=dict(zip(names, map(len, keypoints), strict=True)))
            started = time.perf_counter()
            pairs = _populate_database(pycolmap, database_path, image_root, names, features, keypoints,
                                       lambda left, right: _match_pair(model, left, right, model_size))
        receipt.update(matching_and_verification_seconds=time.perf_counter() - started,
                       pairs=pairs, database_sha256=sha256_file(database_path))
        seeds = verified_pairs(pycolmap, database_path)
        receipt["verified_seed_pairs"] = seeds
        write_json(output / "scene.json", receipt)
        scene, diagnostics = map_training_views(pycolmap, database_path, image_root, output, names, size, seeds, receipt)
        if sha256_file(manifest_path) != manifest_sha256:
            raise ValueError("Prepared manifest changed during LightGlue reconstruction")
        scene_path = output / "scene.npz"
        temporary = output / "_scene.npz"
        np.savez_compressed(temporary, **scene)
        load_scene(temporary)
        temporary.replace(scene_path)
        receipt.update(diagnostics, status="completed", registered_frames=len(names),
                       initial_points=len(scene["points"]), scene_sha256=sha256_file(scene_path))
    except Exception as error:
        receipt.update(status="failed", error=str(error))
        raise
    finally:
        write_json(output / "scene.json", receipt)
    return scene_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("dataset", "output", "model-dir"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--match-edge", type=int, default=MATCH_EDGE)
    args = parser.parse_args()
    print(initialize(args.dataset, args.output, args.model_dir, args.device, args.match_edge))


if __name__ == "__main__":
    main()

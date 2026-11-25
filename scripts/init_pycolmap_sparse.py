"""Initialize sparse 3DGS from hash-verified training RGB with CPU PyCOLMAP."""

from __future__ import annotations

import argparse
import hashlib
from io import BytesIO
from pathlib import Path
import time

import numpy as np
from PIL import Image

from p4.common import sha256_file, source_version, write_json
from p4.data import load_manifest, load_scene


PYCOLMAP_VERSION = "3.11.1"
SEED = 42
THREADS = 2
MIN_POINTS = 16


def stage_train_images(dataset, image_root, frames):
    dimensions = {(frame["width"], frame["height"]) for frame in frames}
    if len(frames) < 3 or len(dimensions) != 1:
        raise ValueError("A shared PINHOLE requires at least three training views at one prepared image size")
    size, = dimensions
    for frame in frames:
        name = frame["image"]
        source = (dataset / name).resolve()
        if not source.is_relative_to(dataset):
            raise ValueError(f"Training image escapes the prepared dataset: {name}")
        data = source.read_bytes()
        if hashlib.sha256(data).hexdigest() != frame["sha256"]:
            raise ValueError(f"Training RGB hash differs from the prepared manifest: {name}")
        with Image.open(BytesIO(data)) as image:
            if image.size != size:
                raise ValueError(f"Training RGB size differs from the prepared manifest: {name}")
        target = image_root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
    return size


def verified_pairs(pycolmap, database_path):
    database = pycolmap.Database()
    database.open(str(database_path))
    try:
        names = {image.image_id: image.name for image in database.read_all_images()}
        pair_ids, geometries = database.read_two_view_geometries()
        viable = {int(pycolmap.TwoViewGeometryConfiguration.CALIBRATED),
                  int(pycolmap.TwoViewGeometryConfiguration.UNCALIBRATED)}
        pairs = []
        for pair_id, geometry in zip(pair_ids, geometries, strict=True):
            ids = pycolmap.Database.pair_id_to_image_pair(pair_id)
            count = len(geometry.inlier_matches)
            if int(geometry.config) in viable and count >= 45:
                pairs.append({"image_ids": list(ids), "names": [names[i] for i in ids], "inliers": count})
        return sorted(pairs, key=lambda pair: (-pair["inliers"], pair["names"]))
    finally:
        database.close()


def reconstruction_to_scene(model, names, size):
    images = {model.image(image_id).name: model.image(image_id) for image_id in model.reg_image_ids()}
    if images.keys() != set(names):
        raise ValueError("The reconstruction must register exactly the training image names")
    camera_ids = {image.camera_id for image in images.values()}
    if len(camera_ids) != 1:
        raise ValueError("The reconstruction must have one shared training camera")
    camera = model.camera(camera_ids.pop())
    if camera.model.name != "PINHOLE" or (camera.width, camera.height) != size:
        raise ValueError("The shared PINHOLE must use the prepared image dimensions")
    w2c = np.repeat(np.eye(4)[None], len(names), axis=0)
    w2c[:, :3] = [images[name].cam_from_world.matrix() for name in names]
    c2w = np.linalg.inv(w2c)
    centers = c2w[:, :3, 3]
    singular = np.linalg.svd(centers - centers.mean(axis=0), compute_uv=False)
    if singular[0] <= 1e-10 or singular[1] / singular[0] < 1e-5:
        raise ValueError("Training camera centers cannot identify a global Sim(3)")
    points = [model.points3D[i] for i in sorted(model.points3D)]
    if len(points) < MIN_POINTS:
        raise ValueError(f"Expected at least {MIN_POINTS} triangulated seed points")
    scene = {"names": np.asarray(names), "c2w": c2w,
             "K": np.repeat(camera.calibration_matrix()[None], len(names), axis=0),
             "points": np.asarray([point.xyz for point in points], dtype=np.float32),
             "colors": np.asarray([point.color for point in points], dtype=np.uint8)}
    return scene, {"camera_center_singular_values": singular.tolist(),
                   "mean_reprojection_error_px": model.compute_mean_reprojection_error(),
                   "mean_track_length": model.compute_mean_track_length(),
                   "observed_points_per_train_image": {name: images[name].num_points3D for name in names}}


def map_training_views(pycolmap, database_path, image_root, output, names, size, pairs, receipt):
    # When the conservative automatic seed fails, use lower parallax and
    # robustly verified, non-planar train-pair anchors in inlier-count order.
    runs = [("automatic", None), ("automatic_low_parallax", None)]
    runs.extend(("anchored_low_parallax", pair) for pair in pairs[:3])
    for index, (policy, pair) in enumerate(runs):
        mapping = pycolmap.IncrementalPipelineOptions(num_threads=THREADS, min_model_size=3,
                                                      multiple_models=False, ba_refine_extra_params=False)
        mapping.triangulation.ignore_two_view_tracks = False
        if index:
            mapping.mapper.init_min_tri_angle = 6.0
            mapping.mapper.init_min_num_inliers = 45
        if pair is not None:
            mapping.init_image_id1, mapping.init_image_id2 = pair["image_ids"]
        model_path = output / f"mapping_{index:02d}"
        model_path.mkdir()
        pycolmap.set_random_seed(SEED)
        started = time.perf_counter()
        models = pycolmap.incremental_mapping(str(database_path), str(image_root), str(model_path), options=mapping)
        attempt = {"policy": policy, "anchor": pair, "initial_min_angle_degrees": mapping.mapper.init_min_tri_angle,
                   "initial_min_inliers": mapping.mapper.init_min_num_inliers, "seconds": time.perf_counter() - started,
                   "registered_frames": max((model.num_reg_images() for model in models.values()), default=0)}
        receipt["mapping_attempts"].append(attempt)
        for model in models.values():
            registered = {model.image(i).name for i in model.reg_image_ids()}
            missing = sorted(set(names) - registered)
            if missing:
                attempt["missing_train_images"] = missing
                continue
            try:
                scene, diagnostics = reconstruction_to_scene(model, names, size)
            except ValueError as exc:
                attempt["validation_error"] = str(exc)
                continue
            return scene, diagnostics
        write_json(output / "scene.json", receipt)
    raise RuntimeError("No valid fully registered PyCOLMAP model; see scene.json and mapping outputs")


def initialize(dataset, output):
    import pycolmap

    if pycolmap.__version__ != PYCOLMAP_VERSION:
        raise ValueError(f"Expected PyCOLMAP {PYCOLMAP_VERSION}, got {pycolmap.__version__}")
    dataset, output = Path(dataset).resolve(), Path(output).resolve()
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
    receipt = {**source_version(), "initializer_script_sha256": sha256_file(__file__), "schema_version": 1,
               "scene": manifest["scene"], "method": "pycolmap", "pycolmap_version": PYCOLMAP_VERSION,
               "manifest_sha256": manifest_sha256, "split_hash": manifest["split_hash"], "train_images": names,
               "train_only_inputs": True, "camera_convention": "opencv_c2w", "camera_model": "shared PINHOLE",
               "canonical_image_wh": list(size), "matcher": "exhaustive CPU SIFT", "num_threads": THREADS,
               "seed": SEED, "bundle_adjustment": True, "mapping_attempts": [], "status": "running"}
    write_json(output / "scene.json", receipt)
    try:
        started = time.perf_counter()
        pycolmap.set_random_seed(SEED)
        extraction = pycolmap.SiftExtractionOptions(num_threads=THREADS)
        pycolmap.extract_features(str(database_path), str(image_root), image_list=names,
                                  camera_mode=pycolmap.CameraMode.SINGLE, camera_model="PINHOLE",
                                  sift_options=extraction, device=pycolmap.Device.cpu)
        receipt["feature_extraction_seconds"] = time.perf_counter() - started
        started = time.perf_counter()
        matching = pycolmap.SiftMatchingOptions(num_threads=THREADS)
        pycolmap.match_exhaustive(str(database_path), sift_options=matching,
                                  matching_options=pycolmap.ExhaustiveMatchingOptions(block_size=16),
                                  device=pycolmap.Device.cpu)
        receipt["exhaustive_matching_seconds"] = time.perf_counter() - started
        pairs = verified_pairs(pycolmap, database_path)
        receipt["verified_seed_pairs"] = pairs
        scene, diagnostics = map_training_views(pycolmap, database_path, image_root, output, names, size, pairs, receipt)
        if sha256_file(manifest_path) != manifest_sha256:
            raise ValueError("Prepared manifest changed during PyCOLMAP reconstruction")
        scene_path = output / "scene.npz"
        temporary = output / "_scene.npz"
        np.savez_compressed(temporary, **scene)
        load_scene(temporary)
        temporary.replace(scene_path)
        receipt.update(diagnostics, status="completed", registered_frames=len(names),
                       initial_points=len(scene["points"]), scene_sha256=sha256_file(scene_path))
        write_json(output / "scene.json", receipt)
    except Exception as exc:
        receipt.update(status="failed", error=str(exc))
        write_json(output / "scene.json", receipt)
        raise
    return scene_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    print(initialize(args.dataset, args.output))


if __name__ == "__main__":
    main()

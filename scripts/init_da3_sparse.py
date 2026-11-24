"""Initialize a sparse 3DGS scene from train RGB using the pinned DA3-SMALL model."""

from __future__ import annotations

import argparse
import hashlib
from io import BytesIO
import json
from pathlib import Path
import random
import sys
import tempfile
import time

import numpy as np
from PIL import Image

from p4.common import sha256_file, source_version, write_json
from p4.data import load_manifest, load_scene
from p4.initialize import depth_to_points


SOURCE_REVISION = "3d835ec1a5802d64a8b8b15f817a1ab54809bfe4"
SOURCE_SHA256 = "5d33078424ffb1e51c34429ef7aac64c1298c632347aaaa3a97b3f8125a00be5"
MODEL_ID = "depth-anything/DA3-SMALL"
MODEL_REVISION = "e08cab65ca0ec38e7826075418411ab90cab4da3"
MODEL_SHA256 = "364492e38a3a06d221ac75da7f6621ada3f2361cd24fde11ba79091e9f40efcf"
CONFIG_SHA256 = "a486e29e82b7ab4a7d4cefc1ea4526cfe2ae438a572c8ca98917cfbcde7447d2"
PROCESS_RES = 504
RESIZE = "upper_bound_resize"
REF_VIEW = "middle"
SEED = 42
MAX_POINTS = 100_000
CONFIDENCE_QUANTILE = 0.5


def _package_sha256(package):
    digest = hashlib.sha256()
    files = sorted(path.relative_to(package).as_posix() for path in package.rglob("*")
                   if path.is_file() and path.suffix in {".py", ".yaml", ".yml", ".json"})
    for name in files:
        data = (package / name).read_bytes().replace(b"\r\n", b"\n")
        digest.update(name.encode() + b"\0" + hashlib.sha256(data).digest())
    return digest.hexdigest()


def _load_train_rgb(dataset):
    dataset = Path(dataset).resolve()
    manifest = load_manifest(dataset)
    if manifest.get("camera_convention") != "opencv_c2w":
        raise ValueError("Expected the project's OpenCV camera convention")
    ids = [frame["id"] for frame in manifest["frames"]]
    if len(ids) != len(set(ids)):
        raise ValueError("Manifest frame IDs must be unique")
    train = [frame for frame in manifest["frames"] if frame["split"] == "train"]
    dimensions = {(frame["width"], frame["height"]) for frame in train}
    if len(train) < 3 or len(dimensions) != 1:
        raise ValueError("DA3 requires at least three training views at one common canonical image size")
    size, = dimensions
    names, images = [], []
    for frame in train:
        name = frame["image"]
        path = (dataset / name).resolve()
        if not path.is_relative_to(dataset) or not path.is_file():
            raise ValueError(f"Missing or escaping training image: {name}")
        data = path.read_bytes()
        if hashlib.sha256(data).hexdigest() != frame.get("sha256"):
            raise ValueError(f"Training RGB hash differs from the prepared manifest: {name}")
        with Image.open(BytesIO(data)) as image:
            if image.size != size:
                raise ValueError(f"Training RGB size differs from the prepared manifest: {name}")
            images.append(np.array(image.convert("RGB")))
        names.append(name)
    return manifest, names, images, size


def prediction_to_scene(prediction, names, canonical_size):
    """Convert official DA3 prediction arrays to canonical OpenCV c2w and seed points."""
    count = len(names)
    if count < 3 or len(set(names)) != count:
        raise ValueError("Expected at least three uniquely named input views")
    depth = np.asarray(prediction.depth, dtype=np.float32)
    confidence = np.asarray(prediction.conf, dtype=np.float32)
    rgb = np.asarray(prediction.processed_images)
    if (depth.ndim != 3 or depth.shape[0] != count or confidence.shape != depth.shape
            or rgb.shape != (*depth.shape, 3) or rgb.dtype != np.uint8):
        raise ValueError("DA3 depth, confidence, and processed RGB must match the ordered input grid")
    valid = np.isfinite(depth) & (depth > 0) & np.isfinite(confidence)
    if not valid.reshape(count, -1).any(axis=1).all():
        raise ValueError("A DA3 training view has no valid depth and confidence")
    _, proc_h, proc_w = depth.shape
    width, height = canonical_size
    if min(width, height, proc_w, proc_h) <= 0:
        raise ValueError("DA3 canonical and processed image dimensions must be positive")

    extrinsics = np.asarray(prediction.extrinsics, dtype=np.float64)
    if extrinsics.shape == (count, 3, 4):
        w2c = np.broadcast_to(np.eye(4), (count, 4, 4)).copy()
        w2c[:, :3] = extrinsics
    elif extrinsics.shape == (count, 4, 4):
        w2c = extrinsics
    else:
        raise ValueError("DA3 did not predict one OpenCV w2c matrix per training image")
    rotation = w2c[:, :3, :3]
    if (not np.isfinite(w2c).all() or not np.allclose(w2c[:, 3], [0, 0, 0, 1], atol=1e-5)
            or not np.allclose(rotation @ rotation.transpose(0, 2, 1), np.eye(3), atol=1e-3)
            or not np.allclose(np.linalg.det(rotation), 1, atol=1e-3)):
        raise ValueError("DA3 predicted a nonrigid or invalid OpenCV w2c matrix")
    c2w = np.linalg.inv(w2c)
    centers = c2w[:, :3, 3]
    singular = np.linalg.svd(centers - centers.mean(axis=0), compute_uv=False)
    # The project's train-only Sim(3) convention needs noncollinear cameras.
    if singular[0] < 1e-10 or singular[1] / singular[0] < 1e-5:
        raise ValueError("DA3 training camera path cannot identify a global Sim(3)")

    model_K = np.asarray(prediction.intrinsics, dtype=np.float64)
    if (model_K.shape != (count, 3, 3) or not np.isfinite(model_K).all()
            or not np.allclose(model_K[:, 2], [0, 0, 1], atol=1e-5)
            or not np.allclose(model_K[:, (0, 1), (1, 0)], 0, atol=1e-5)
            or np.any(model_K[:, (0, 1), (0, 1)] <= 0)):
        raise ValueError("DA3 did not predict one valid OpenCV pinhole K per training image")
    # Equal input sizes rule out DA3's mixed-size center crop. The official
    # input processor scales the two rows of K using the actual resize ratios.
    resize = np.diag([proc_w / width, proc_h / height, 1.])
    canonical_K = np.linalg.inv(resize) @ model_K
    bounds = np.broadcast_to([0, 0, proc_w, proc_h], (count, 4))
    points, colors, selection = depth_to_points(depth, confidence, w2c, model_K, rgb, bounds,
                                                CONFIDENCE_QUANTILE, MAX_POINTS, SEED)
    if len(points) < 4:
        raise ValueError("DA3 depth did not yield at least four usable seed Gaussians")
    scene = {"names": np.asarray(names), "c2w": c2w, "K": canonical_K, "points": points, "colors": colors}
    metadata = {**selection, "registered_frames": count, "initial_points": len(points),
                "canonical_image_wh": [width, height], "processed_image_wh": [proc_w, proc_h],
                "canonical_to_model": resize.tolist(), "camera_center_singular_values": singular.tolist()}
    return scene, metadata


def initialize(dataset, output, repo, model_dir, device, source_revision=SOURCE_REVISION):
    dataset, output, repo, model_dir = (Path(path).resolve() for path in (dataset, output, repo, model_dir))
    if output.exists():
        raise FileExistsError(f"Use a new initializer directory: {output}")
    package = repo / "src/depth_anything_3"
    if source_revision != SOURCE_REVISION or _package_sha256(package) != SOURCE_SHA256:
        raise ValueError("DA3 source does not match the pinned revision")
    for name, digest in (("config.json", CONFIG_SHA256), ("model.safetensors", MODEL_SHA256)):
        if sha256_file(model_dir / name) != digest:
            raise ValueError(f"DA3-SMALL {name} does not match the official pinned release")
    manifest_sha256 = sha256_file(dataset / "manifest.json")
    manifest, names, images, size = _load_train_rgb(dataset)

    import torch

    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was selected but is unavailable")
    sys.path.insert(0, str(package.parent))
    from depth_anything_3 import api
    if Path(api.__file__).resolve() != (package / "api.py").resolve():
        raise RuntimeError("DA3 import did not resolve to the verified source checkout")
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    tick = time.perf_counter()
    model = api.DepthAnything3.from_pretrained(str(model_dir), local_files_only=True, strict=True)
    if model.model_name != "da3-small":
        raise ValueError("Expected the DA3-SMALL any-view preset")
    model = model.eval().to(device)
    model_load_seconds = time.perf_counter() - tick
    tick = time.perf_counter()
    prediction = model.inference(image=images, extrinsics=None, intrinsics=None, infer_gs=False,
                                 use_ray_pose=True, ref_view_strategy=REF_VIEW, process_res=PROCESS_RES,
                                 process_res_method=RESIZE, export_dir=None)
    inference_seconds = time.perf_counter() - tick
    scene, geometry = prediction_to_scene(prediction, names, size)
    if sha256_file(dataset / "manifest.json") != manifest_sha256:
        raise ValueError("Prepared manifest changed during DA3 inference")
    receipt = {**source_version(), **geometry, "schema_version": 1, "scene": manifest["scene"],
               "method": "da3-small", "model_id": MODEL_ID, "model_revision": MODEL_REVISION,
               "model_config_sha256": CONFIG_SHA256, "weights_sha256": MODEL_SHA256,
               "upstream_commit": source_revision, "upstream_package_sha256": SOURCE_SHA256,
               "manifest_sha256": manifest_sha256, "split_hash": manifest["split_hash"],
               "train_images": names, "train_only_inputs": True, "all_training_views_joint": True,
               "camera_convention": "opencv_c2w", "pose_estimation": "official ray pose",
               "reference_view_policy": REF_VIEW, "process_res": PROCESS_RES,
               "process_res_method": RESIZE, "seed": SEED, "point_cap": MAX_POINTS,
               "confidence_quantile": CONFIDENCE_QUANTILE, "device": device,
               "torch_version": torch.__version__, "model_load_seconds": model_load_seconds,
               "inference_seconds": inference_seconds}
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".da3-", dir=output.parent) as staging:
        staged = Path(staging) / "initializer"
        staged.mkdir()
        scene_path = staged / "scene.npz"
        np.savez_compressed(scene_path, **scene)
        load_scene(scene_path)
        write_json(staged / "scene.json", {**receipt, "scene_sha256": sha256_file(scene_path)})
        staged.rename(output)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("dataset", "output", "repo", "model-dir"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--source-revision", choices=(SOURCE_REVISION,), default=SOURCE_REVISION)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    args = parser.parse_args()
    output = initialize(args.dataset, args.output, args.repo, args.model_dir, args.device, args.source_revision)
    print(json.dumps({"initializer": str(output / "scene.npz"), "receipt": str(output / "scene.json")}))


if __name__ == "__main__":
    main()

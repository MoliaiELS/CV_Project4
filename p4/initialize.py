"""RGB-only COLMAP / VGGT / VGGT-Omega initializers.

All cameras exported here are OpenCV camera-to-world matrices. Reference
cameras are deliberately never opened by this module. Neural models run the
entire training sequence jointly; an OOM never silently changes the protocol.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np
from PIL import Image

MODEL_COMMITS = {
    "vggt": "a288dd0f14786c93483e45524328726ab7b1b4ce",
    "vggt-omega": "48b23c8ce72c9a8fcf987f0d7f43d8e4870759f6",
}
OMEGA_CHECKPOINT = "vggt_omega_1b_416_reproduce.pt"


def _sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def train_frames(dataset):
    """Return allowed RGB inputs, rejecting duplicates and escaping paths."""
    from p4.data import load_manifest
    root = Path(dataset).resolve()
    manifest = load_manifest(root)
    frames = [f for f in manifest["frames"] if f["split"] == "train"]
    if len(frames) < 2:
        raise ValueError("At least two training RGB images are required")
    names = [f["image"] for f in frames]
    if len(set(names)) != len(names):
        raise ValueError("Duplicate training image names")
    for frame, name in zip(frames, names):
        path = (root / name).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise ValueError(f"Missing or escaping training image: {name}")
        if not frame.get("sha256") or _sha256(path) != frame["sha256"]:
            raise ValueError(f"Training RGB hash differs from prepared manifest: {name}; re-prepare into a new dataset directory")
    return manifest, frames


def preprocessing_geometry(sizes, method):
    """Return canonical->model pixel transforms and valid image bounds.

    This mirrors the pinned official preprocessing (VGGT square padding;
    Omega balanced416, aspect-ratio crop, then common-shape center padding).
    A maps homogeneous canonical image pixels to model image pixels; therefore
    K_canonical = inv(A) @ K_model, including crop and asymmetric padding.
    """
    if method not in MODEL_COMMITS:
        raise ValueError(method)
    transforms, bounds, shapes = [], [], []
    for width, height in sizes:
        if min(width, height) <= 0:
            raise ValueError("Invalid image dimensions")
        if method == "vggt":
            side = max(width, height)
            left, top = (side - width) // 2, (side - height) // 2
            scale = 518 / side
            a = np.array([[scale, 0, left * scale], [0, scale, top * scale], [0, 0, 1.]])
            bound = [left * scale, top * scale, (left + width) * scale, (top + height) * scale]
            shape = (518, 518)
        else:
            cw, ch, left, top = width, height, 0, 0
            aspect = height / width
            if aspect < .5:
                cw = min(width, max(1, int(round(height / .5))))
                left = max((width - cw) // 2, 0)
            elif aspect > 2:
                ch = min(height, max(1, int(round(width * 2))))
                top = max((height - ch) // 2, 0)
            tokens = (416 // 16) ** 2
            wp = np.sqrt(tokens / (ch / cw))
            hp = tokens / wp
            w, h = max(1, int(np.round(wp))) * 16, max(1, int(np.round(hp))) * 16
            sx, sy = w / cw, h / ch
            a = np.array([[sx, 0, -left * sx], [0, sy, -top * sy], [0, 0, 1.]])
            bound, shape = [0, 0, w, h], (h, w)
        transforms.append(a)
        bounds.append(bound)
        shapes.append(shape)
    hmax, wmax = np.max(shapes, axis=0)
    for i, (h, w) in enumerate(shapes):
        top, left = (hmax - h) // 2, (wmax - w) // 2
        transforms[i][0, 2] += left
        transforms[i][1, 2] += top
        bounds[i] = np.array(bounds[i]) + [left, top, left, top]
    return np.stack(transforms), np.stack(bounds), (int(hmax), int(wmax))


def depth_to_points(depth, confidence, w2c, intrinsics, rgb, bounds,
                    confidence_quantile=.5, max_points=100000, seed=42):
    """Filter padded/nonfinite pixels, then deterministically sample and unproject."""
    depth, confidence = np.asarray(depth), np.asarray(confidence)
    if depth.ndim == 4 and depth.shape[-1] == 1:
        depth = depth[..., 0]
    if depth.ndim != 3 or confidence.shape != depth.shape:
        raise ValueError("Expected depth/confidence [N,H,W]")
    if not 0 <= confidence_quantile < 1 or max_points <= 0:
        raise ValueError("Require 0<=confidence_quantile<1 and max_points>0")
    n, h, w = depth.shape
    if rgb.shape != (n, h, w, 3):
        raise ValueError("RGB and predicted depth pixel grids differ")
    valid = np.isfinite(depth) & (depth > 0) & np.isfinite(confidence)
    ys, xs = np.mgrid[:h, :w]
    for i, (x1, y1, x2, y2) in enumerate(bounds):
        valid[i] &= (xs >= x1) & (xs < x2) & (ys >= y1) & (ys < y2)
    if not valid.any():
        raise ValueError("No valid depth predictions within RGB bounds")
    threshold = float(np.quantile(confidence[valid], confidence_quantile))
    valid &= confidence >= threshold
    selected = np.flatnonzero(valid)
    valid_count = len(selected)
    if len(selected) > max_points:
        selected = np.sort(np.random.default_rng(seed).choice(selected, max_points, replace=False))
    fi, yi, xi = np.unravel_index(selected, depth.shape)
    pixels = np.stack([xi, yi, np.ones(len(xi))], axis=-1)
    rays = np.einsum("nij,nj->ni", np.linalg.inv(intrinsics)[fi], pixels)
    camera_points = rays * depth[fi, yi, xi, None]
    camera_to_world = np.linalg.inv(w2c)
    points = np.einsum("nij,nj->ni", camera_to_world[fi, :3, :3], camera_points) + camera_to_world[fi, :3, 3]
    colors = np.asarray(rgb[fi, yi, xi], dtype=np.uint8)
    return points.astype(np.float32), colors, {"confidence_threshold": threshold, "valid_points_before_cap": valid_count}


def _repo_info(repo, method):
    if repo is None:
        raise ValueError(f"--repo must point to the pinned {method} checkout")
    repo = Path(repo).resolve()
    actual = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    if actual != MODEL_COMMITS[method]:
        raise ValueError(f"{method} source revision {actual}; expected {MODEL_COMMITS[method]}")
    dirty = subprocess.check_output(["git", "-C", str(repo), "status", "--porcelain", "--untracked-files=no"], text=True).strip()
    if dirty:
        raise ValueError(f"Pinned {method} tracked source has local modifications")
    return repo, actual


def _vggt_with_offloaded_features(model, images):
    """Keep joint VGGT attention and stage the saved head features on CPU."""
    import torch

    images = images.unsqueeze(0)
    batch_size, frame_count = images.shape[:2]
    aggregator = model.aggregator
    cached_layers = aggregator.cached_layer_indices
    depth_layers = model.depth_head.intermediate_layer_idx
    last_layer = len(aggregator.frame_blocks) - 1
    frame_features, global_features = {}, {}

    def capture(features, layer):
        def save_feature(_module, _inputs, tokens):
            features[layer] = tokens.detach().cpu()
        return save_feature

    hooks = []
    try:
        # Stock VGGT concatenates these same features on device. Save the
        # inputs to that concatenation as each layer runs, in their native dtype.
        aggregator.cached_layer_indices = set()
        for layer in sorted(cached_layers):
            hooks.append(aggregator.frame_blocks[layer].register_forward_hook(capture(frame_features, layer)))
            hooks.append(aggregator.global_blocks[layer].register_forward_hook(capture(global_features, layer)))
        layer_outputs, patch_start_idx = aggregator(images)
    finally:
        aggregator.cached_layer_indices = cached_layers
        for hook in hooks:
            hook.remove()

    def head_features(layer, start, end, token_stop=None):
        frame, global_ = frame_features[layer], global_features[layer]
        channels = frame.shape[-1]
        frame = frame.reshape(batch_size, frame_count, -1, channels)[:, start:end, :token_stop]
        global_ = global_.reshape(batch_size, frame_count, -1, channels)[:, start:end, :token_stop]
        return torch.cat((frame, global_), dim=-1).to(images.device)

    with torch.autocast(device_type=images.device.type, enabled=False):
        # CameraHead reads only token zero from the final layer, across all views.
        layer_outputs[-1] = head_features(last_layer, 0, frame_count, token_stop=1)
        pose_enc = model.camera_head(layer_outputs)[-1]
        layer_outputs[-1] = None

        depths, confidences = [], []
        # Match the pinned DPT head's default eight frames per decoder call.
        for start in range(0, frame_count, 8):
            end = min(start + 8, frame_count)
            for layer in depth_layers:
                layer_outputs[layer] = head_features(layer, start, end)
            depth, confidence = model.depth_head(
                layer_outputs, images=images[:, start:end].contiguous(),
                patch_start_idx=patch_start_idx,
            )
            depths.append(depth.cpu())
            confidences.append(confidence.cpu())
            for layer in depth_layers:
                layer_outputs[layer] = None
    return {"pose_enc": pose_enc, "depth": torch.cat(depths, dim=1),
            "depth_conf": torch.cat(confidences, dim=1)}


def _run_neural(dataset, output, frames, method, repo, weights, seed,
                max_points, confidence_quantile, vggt_offload_features=False, neural_backend="native"):
    repo, revision = _repo_info(repo, method)
    if not weights or not Path(weights).is_file():
        raise ValueError("An explicit local --weights file is required; see README for downloads/access")
    if method == "vggt-omega" and Path(weights).name != OMEGA_CHECKPOINT:
        raise ValueError(f"Benchmark protocol requires {OMEGA_CHECKPOINT} (not the original512 checkpoint)")
    try:
        import torch
    except ImportError as e:
        raise RuntimeError("Neural initialization requires the CUDA server environment") from e
    if not torch.cuda.is_available():
        raise RuntimeError("Neural initialization requires an NVIDIA CUDA GPU; use the server setup")
    sys.path.insert(0, str(repo))
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.cuda.reset_peak_memory_stats()
    paths = [str(Path(dataset) / f["image"]) for f in frames]
    sizes = []
    for path in paths:
        with Image.open(path) as im:
            sizes.append(im.size)
    a, bounds, shape = preprocessing_geometry(sizes, method)
    started = time.perf_counter()
    if method == "vggt":
        from vggt.models.vggt import VGGT
        from vggt.utils.load_fn import load_and_preprocess_images_square
        from vggt.utils.pose_enc import pose_encoding_to_extri_intri
        model = VGGT()
        decoder = pose_encoding_to_extri_intri
        images, _ = load_and_preprocess_images_square(paths, target_size=518)
    else:
        from vggt_omega.models import VGGTOmega
        from vggt_omega.utils.load_fn import load_and_preprocess_images
        from vggt_omega.utils.pose_enc import encoding_to_camera
        model, decoder = VGGTOmega(), encoding_to_camera
        images = load_and_preprocess_images(paths, image_resolution=416, mode="balanced")
    if tuple(images.shape[-2:]) != shape:
        raise ValueError(f"Official preprocessing shape {images.shape[-2:]} differs from geometry {shape}")
    checkpoint = torch.load(str(weights), map_location="cpu", weights_only=True, mmap=neural_backend == "offload")
    model.load_state_dict(checkpoint, strict=True)
    del checkpoint
    if method == "vggt":
        # These prediction heads do not contribute to camera/depth inference.
        model.point_head = None
        model.track_head = None
    model = model.eval()
    if neural_backend == "native":
        model = model.to("cuda")
        images = images.to("cuda")
    torch.cuda.synchronize()
    load_seconds = time.perf_counter() - started
    inference_start = time.perf_counter()
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    try:
        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=dtype):
            if vggt_offload_features:
                predictions = _vggt_with_offloaded_features(model, images)
            elif neural_backend == "offload":
                from p4.vggt_offload import forward
                predictions = forward(model, images, output / "offload_cache")
            else:
                predictions = model(images)
        with torch.inference_mode(), torch.autocast(device_type="cuda", enabled=False):
            ext, kin = decoder(predictions["pose_enc"].float(), images.shape[-2:])
        torch.cuda.synchronize()
    except torch.cuda.OutOfMemoryError as e:
        memory_hint = "Use a larger GPU."
        if method == "vggt" and not vggt_offload_features and neural_backend == "native":
            memory_hint = "Try --vggt-offload-features or --neural-backend offload, or use a larger GPU."
        raise RuntimeError(
            f"Joint {method} inference OOM for {len(frames)} train frames at {shape}. {memory_hint}"
        ) from e
    inference_seconds = time.perf_counter() - inference_start
    export_start = time.perf_counter()
    ext = ext[0].float().cpu().numpy()
    kin = kin[0].float().cpu().numpy()
    w2c = np.broadcast_to(np.eye(4), (len(frames), 4, 4)).copy()
    w2c[:, :3] = ext
    rgb = np.clip(images.detach().float().cpu().numpy().transpose(0, 2, 3, 1) * 255, 0, 255).astype(np.uint8)
    points, colors, filtering = depth_to_points(
        predictions["depth"][0].float().cpu().numpy(),
        predictions["depth_conf"][0].float().cpu().numpy(),
        w2c, kin, rgb, bounds, confidence_quantile, max_points, seed,
    )
    canonical_k = np.linalg.inv(a) @ kin
    np.savez_compressed(output / "scene.npz", names=np.array([f["image"] for f in frames]),
                        c2w=np.linalg.inv(w2c), K=canonical_k, points=points, colors=colors)
    return {
        "neural_backend": neural_backend, "all_training_views_joint": True,
        "source_commit": revision, "weights_file": Path(weights).name, "weights_sha256": _sha256(weights),
        "model_resolution": 518 if method == "vggt" else 416, "model_input_hw": list(shape),
        "preprocessing": "square_pad_bicubic518" if method == "vggt" else "official_balanced416_crop_and_pad",
        "canonical_to_model": a.tolist(), "valid_model_bounds": bounds.tolist(),
        "load_preprocess_seconds": load_seconds, "inference_seconds": inference_seconds,
        "export_filter_seconds": time.perf_counter() - export_start,
        "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated(),
        "gpu": torch.cuda.get_device_name(), "torch_version": torch.__version__,
        "registered_frames": len(frames), "initial_points": len(points),
        "bundle_adjustment": False, "confidence_quantile": confidence_quantile,
        "max_points": max_points, **filtering,
    }


def _read_colmap_text(model):
    """Read the PINHOLE model_converter output without importing pycolmap."""
    cameras = {}
    for line in (model / "cameras.txt").read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        v = line.split()
        if v[1] != "PINHOLE":
            raise ValueError("COLMAP branch requires PINHOLE to preserve common canonical RGB")
        fx, fy, cx, cy = map(float, v[4:8])
        cameras[int(v[0])] = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.]])
    # Every image record has a second observation line, which may be empty.
    lines = (model / "images.txt").read_text(encoding="utf-8").splitlines()
    images, i = {}, 0
    while i < len(lines):
        line = lines[i]
        i += 1
        if not line.strip() or line.startswith("#"):
            continue
        v = line.split(maxsplit=9)
        q = np.array(list(map(float, v[1:5])))
        q = q / np.linalg.norm(q)
        qw, qx, qy, qz = q
        r = np.array([
            [1-2*qy*qy-2*qz*qz, 2*qx*qy-2*qw*qz, 2*qx*qz+2*qw*qy],
            [2*qx*qy+2*qw*qz, 1-2*qx*qx-2*qz*qz, 2*qy*qz-2*qw*qx],
            [2*qx*qz-2*qw*qy, 2*qy*qz+2*qw*qx, 1-2*qx*qx-2*qy*qy],
        ])
        w2c = np.eye(4)
        w2c[:3, :3], w2c[:3, 3] = r, list(map(float, v[5:8]))
        images[v[9]] = (np.linalg.inv(w2c), cameras[int(v[8])])
        i += 1
    points, colors = [], []
    for line in (model / "points3D.txt").read_text(encoding="utf-8").splitlines():
        if line.strip() and not line.startswith("#"):
            v = line.split()
            points.append(list(map(float, v[1:4])))
            colors.append(list(map(int, v[4:7])))
    return images, np.array(points, dtype=np.float32).reshape(-1, 3), np.array(colors, dtype=np.uint8).reshape(-1, 3)


def _run_colmap(dataset, output, frames, seed, matcher, colmap):
    if matcher not in ("exhaustive", "sequential"):
        raise ValueError("COLMAP matcher must be exhaustive or sequential")
    if not shutil.which(colmap):
        raise RuntimeError("COLMAP executable not found; install COLMAP on the server")
    image_dir, sparse = output / "train_rgb", output / "colmap_sparse"
    image_dir.mkdir()
    sparse.mkdir()
    for f in frames:
        target = image_dir / f["image"]
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.link(Path(dataset) / f["image"], target)
        except OSError:
            shutil.copy2(Path(dataset) / f["image"], target)
    db = output / "database.db"
    commands = [
        [colmap, "feature_extractor", "--database_path", str(db), "--image_path", str(image_dir),
         "--ImageReader.camera_model", "PINHOLE", "--ImageReader.single_camera", "0", "--SiftExtraction.use_gpu", "0",
         "--random_seed", str(seed)],
        [colmap, matcher + "_matcher", "--database_path", str(db), "--SiftMatching.use_gpu", "0", "--random_seed", str(seed)],
        [colmap, "mapper", "--database_path", str(db), "--image_path", str(image_dir), "--output_path", str(sparse),
         "--random_seed", str(seed)],
    ]
    timings = {}
    with (output / "colmap.log").open("w", encoding="utf-8") as log:
        for command in commands:
            log.write(json.dumps(command) + "\n")
            log.flush()
            started = time.perf_counter()
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
            timings[command[1] + "_seconds"] = time.perf_counter() - started
        candidates = []
        for model in sorted(sparse.iterdir()):
            if not model.is_dir():
                continue
            txt = output / ("colmap_text_" + model.name)
            txt.mkdir()
            subprocess.run([colmap, "model_converter", "--input_path", str(model), "--output_path", str(txt),
                            "--output_type", "TXT"], stdout=log, stderr=subprocess.STDOUT, check=True)
            candidates.append(_read_colmap_text(txt))
    if not candidates:
        raise RuntimeError("COLMAP produced no registered model; see colmap.log")
    images, points, colors = max(candidates, key=lambda v: len(v[0]))
    expected = [f["image"] for f in frames]
    missing = [name for name in expected if name not in images]
    registration = {"registered_frames": len(images), "expected_frames": len(expected), "missing": missing,
                    "disconnected_models": len(candidates)}
    (output / "registration.json").write_text(json.dumps(registration, indent=2), encoding="utf-8")
    if missing:
        raise RuntimeError(f"COLMAP registered {len(expected)-len(missing)}/{len(expected)} allowed frames; "
                           "partial registration is a failed primary run, see registration.json")
    if not len(points):
        raise RuntimeError("COLMAP model has no sparse points")
    np.savez_compressed(output / "scene.npz", names=np.array(expected),
                        c2w=np.stack([images[n][0] for n in expected]), K=np.stack([images[n][1] for n in expected]),
                        points=points, colors=colors)
    # Mapper includes COLMAP's required local/global BA. It is never reported as no-BA.
    return {**timings, **registration, "initial_points": len(points), "bundle_adjustment": True,
            "ba_seconds": None, "ba_timing_note": "Included in mapper_seconds; individual BA events remain in colmap.log",
            "camera_model": "PINHOLE", "matcher": matcher, "sift_gpu": False,
            "commands": commands, "colmap_help": subprocess.run([colmap, "-h"], capture_output=True, text=True).stdout[:3000]}


def run_initializer(dataset, output, method, repo=None, weights=None, seed=42,
                    max_points=100000, confidence_quantile=.5, matcher="exhaustive", colmap="colmap",
                    vggt_offload_features=False, neural_backend="native"):
    """Create scene.npz and provenance scene.json, preserving failed-run evidence."""
    if method not in ("colmap", *MODEL_COMMITS):
        raise ValueError(f"Unknown initializer: {method}")
    if neural_backend not in {"native", "offload"}:
        raise ValueError(f"Unknown neural backend: {neural_backend}")
    if method != "vggt" and (vggt_offload_features or neural_backend == "offload"):
        raise ValueError("VGGT offload modes apply only to VGGT")
    if vggt_offload_features and neural_backend == "offload":
        raise ValueError("Choose one VGGT offload mode")
    output = Path(output).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Initializer output must be new/empty to prevent stale artifact reuse: {output}")
    output.mkdir(parents=True, exist_ok=True)
    manifest, frames = train_frames(dataset)
    metadata = {"schema_version": 1, "method": method, "seed": seed,
                "scene": manifest.get("scene"), "input_names": [f["image"] for f in frames],
                "input_frames": len(frames), "manifest_sha256": _sha256(Path(dataset) / "manifest.json"),
                "camera_convention": "OpenCV c2w; x-right y-down z-forward", "status": "running"}
    if method == "vggt":
        metadata["neural_backend"] = neural_backend
        strategy = "cpu_disk_layers" if neural_backend == "offload" else "stock"
        metadata["memory_strategy"] = "cpu_head_features" if vggt_offload_features else strategy
    started = time.perf_counter()
    try:
        if method == "colmap":
            result = _run_colmap(dataset, output, frames, seed, matcher, colmap)
        else:
            result = _run_neural(dataset, output, frames, method, repo, weights, seed,
                                 max_points, confidence_quantile, vggt_offload_features, neural_backend)
        metadata.update(result)
        metadata["status"] = "completed"
    except Exception as e:
        metadata.update(status="failed", error=f"{type(e).__name__}: {e}")
        raise
    finally:
        metadata["initialization_seconds"] = time.perf_counter() - started
        (output / "scene.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--method", choices=("colmap", *MODEL_COMMITS), required=True)
    p.add_argument("--neural-backend", choices=("native", "offload"), default="native")
    p.add_argument("--repo")
    p.add_argument("--weights")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-points", type=int, default=100000)
    p.add_argument("--confidence-quantile", type=float, default=.5)
    p.add_argument("--matcher", choices=("exhaustive", "sequential"), default="exhaustive")
    p.add_argument("--colmap", default="colmap")
    p.add_argument("--vggt-offload-features", action="store_true",
                   help="Store VGGT head features on CPU during joint all-frame inference")
    a = p.parse_args()
    print(json.dumps(run_initializer(**vars(a)), indent=2))


if __name__ == "__main__":
    main()

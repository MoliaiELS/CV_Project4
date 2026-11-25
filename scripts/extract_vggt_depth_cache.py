"""Decode an archived VGGT offload cache on CPU using sparse training images."""

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
from scipy.ndimage import map_coordinates
from scipy.spatial import cKDTree

from p4.common import sha256_file, source_version, write_json
from p4.data import load_manifest, load_scene
from p4.initialize import MODEL_COMMITS, depth_to_points, preprocessing_geometry


LAYERS = (4, 11, 17, 23)
PATCH_SIZE = 14
PATCH_START = 5
FEATURE_WIDTH = 2048


def _pinned_checkout(repo):
    # The historical course checkout can be group-readable but owned by the
    # other team member. Limit Git's trust exception to this invocation.
    command = ["git", "--no-optional-locks", "-c", f"safe.directory={repo.as_posix()}", "-C", str(repo)]
    revision = subprocess.check_output([*command, "rev-parse", "HEAD"], text=True).strip()
    dirty = subprocess.check_output([*command, "status", "--porcelain", "--untracked-files=no"], text=True)
    if revision != MODEL_COMMITS["vggt"] or dirty.strip():
        raise ValueError("Use the clean official VGGT checkout pinned by the original initializer")
    return revision


def _load_inputs(dataset, initializer):
    manifest = load_manifest(dataset)
    original = load_scene(initializer)
    metadata = json.loads(initializer.with_suffix(".json").read_text(encoding="utf-8"))
    frames = {frame["image"]: frame for frame in manifest["frames"] if frame["split"] == "train"}
    names = original["names"].tolist()
    if (metadata["method"] != "vggt" or metadata["neural_backend"] != "offload"
            or metadata["status"] != "completed" or names != metadata["input_names"]
            or set(names) != frames.keys() or metadata["input_frames"] != len(names)
            or metadata["manifest_sha256"] != sha256_file(dataset / "manifest.json")):
        raise ValueError("The archive does not describe this exact sparse training split and VGGT cache")
    paths = []
    for name in names:
        path = (dataset / name).resolve()
        if not path.is_relative_to(dataset) or sha256_file(path) != frames[name]["sha256"]:
            raise ValueError(f"Training RGB differs from the prepared manifest: {name}")
        paths.append(str(path))
    sizes = [(frames[name]["width"], frames[name]["height"]) for name in names]
    transform, bounds, model_size = preprocessing_geometry(sizes, "vggt")
    if (len(set(sizes)) != 1 or tuple(metadata["model_input_hw"]) != model_size
            or not np.array_equal(transform, metadata["canonical_to_model"])
            or not np.array_equal(bounds, metadata["valid_model_bounds"])):
        raise ValueError("Cache transforms or prepared sizes do not support a shared depth-map shape")
    return manifest, original, metadata, paths, sizes[0], transform, bounds, model_size


def _open_features(cache, count, model_size):
    height, width = model_size
    tokens = PATCH_START + (height // PATCH_SIZE) * (width // PATCH_SIZE)
    expected = (1, count, tokens, FEATURE_WIDTH)
    arrays, fingerprints = {}, {}
    for layer in LAYERS:
        path = cache / f"layer_{layer:02d}.npy"
        feature = np.load(path, mmap_mode="r", allow_pickle=False)
        if feature.shape != expected or feature.dtype != np.float32:
            raise ValueError(f"Unexpected feature-cache schema: {path}")
        arrays[layer] = feature
        fingerprints[path.name] = sha256_file(path)
    return arrays, fingerprints


def _decode(weights, repo, features, paths, model_size, threads):
    import torch

    sys.path.insert(0, str(repo))
    from vggt.heads.dpt_head import DPTHead
    from vggt.utils.load_fn import load_and_preprocess_images_square

    torch.set_num_threads(threads)
    state = torch.load(str(weights), map_location="cpu", mmap=True, weights_only=True)
    prefix = "depth_head."
    depth_state = {name.removeprefix(prefix): tensor for name, tensor in state.items()
                   if name.startswith(prefix)}
    del state
    with torch.device("meta"):
        head = DPTHead(dim_in=FEATURE_WIDTH, output_dim=2, activation="exp", conf_activation="expp1")
    head.load_state_dict(depth_state, strict=True, assign=True)
    head.eval()
    images, _ = load_and_preprocess_images_square(paths, target_size=model_size[0])
    if tuple(images.shape[-2:]) != model_size or tuple(head.intermediate_layer_idx) != LAYERS:
        raise ValueError("Pinned head or model-space image shape differs from the cache")

    depth_maps, confidence_maps = [], []
    with torch.inference_mode():
        for index in range(len(paths)):
            tokens = [None] * (max(LAYERS) + 1)
            for layer, array in features.items():
                tokens[layer] = torch.from_numpy(np.array(array[:, index:index + 1], copy=True))
            depth, confidence = head(tokens, images[None, index:index + 1], PATCH_START)
            depth_maps.append(depth[0, 0, :, :, 0].numpy())
            confidence_maps.append(confidence[0, 0].numpy())
    depth, confidence = np.stack(depth_maps), np.stack(confidence_maps)
    if (depth.shape != (len(paths), *model_size) or depth.shape != confidence.shape
            or not np.isfinite(depth).all() or not np.isfinite(confidence).all()
            or (depth <= 0).any() or (confidence < 0).any()):
        raise ValueError("Official DPT decoder produced invalid depth or confidence")
    rgb = np.clip(images.numpy().transpose(0, 2, 3, 1) * 255, 0, 255).astype(np.uint8)
    return depth, confidence, rgb, torch.__version__


def _prepared_geometry(depth, confidence, transform, size):
    width, height = size
    y, x = np.mgrid[:height, :width]
    pixels = np.stack((x, y, np.ones_like(x)), axis=0).reshape(3, -1)
    z_maps, c_maps = [], []
    for z_map, c_map, affine in zip(depth, confidence, transform):
        model_x, model_y, _ = affine @ pixels
        coords = [model_y.reshape(height, width), model_x.reshape(height, width)]
        z_maps.append(map_coordinates(z_map, coords, order=1, mode="nearest"))
        c_maps.append(map_coordinates(c_map, coords, order=1, mode="nearest"))
    return np.stack(z_maps).astype(np.float32), np.stack(c_maps).astype(np.float32)


def _cloud(scene, depth, confidence, rgb, transform, bounds, metadata):
    return depth_to_points(depth, confidence, np.linalg.inv(scene["c2w"]), transform @ scene["K"],
                           rgb, bounds, metadata["confidence_quantile"], metadata["max_points"], metadata["seed"])


def _validate_gauge(original, target):
    poses = original["c2w"]
    centers = poses[:, :3, 3]
    baseline = np.linalg.norm(centers - centers[0], axis=1)
    anchor = int(np.argmax(baseline))
    if (not np.array_equal(target["names"], original["names"])
            or not np.allclose(target["c2w"][0], poses[0], rtol=0, atol=1e-8)
            or not np.allclose(target["c2w"][anchor, :3, 3], centers[anchor], rtol=0, atol=1e-8)
            or baseline[anchor] <= 1e-8):
        raise ValueError("Target cameras must preserve the original training IDs, first pose, and anchor-center gauge")
    return float(baseline[anchor])


def _replay_summary(points, colors, original, radius):
    old = original["points"]
    forward = cKDTree(old).query(points)[0]
    backward = cKDTree(points).query(old)[0]
    distances = np.r_[forward, backward] / radius
    return {"points": len(points), "source_points": len(old),
            "median_symmetric_nearest_distance_over_anchor": float(np.median(distances)),
            "p95_symmetric_nearest_distance_over_anchor": float(np.quantile(distances, 0.95)),
            "identical_rgb_fraction_at_same_index": (float(np.all(colors == original["colors"], axis=1).mean())
                                                     if colors.shape == original["colors"].shape else None)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--initializer", type=Path, required=True, help="Original sparse VGGT scene.npz")
    parser.add_argument("--repo", type=Path, required=True, help="Pinned official VGGT checkout")
    parser.add_argument("--weights", type=Path, required=True, help="Verified official VGGT model.pt")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--export-cloud", action="store_true", help="Export a new scene.npz and bind the maps to it")
    parser.add_argument("--target-initializer", type=Path, help="Optional gauge-preserving calibration for cloud export")
    args = parser.parse_args()
    if args.threads < 1 or (args.target_initializer and not args.export_cloud):
        parser.error("--threads must be positive and --target-initializer requires --export-cloud")
    names = ("depth_maps.npz", "depth_receipt.json", "scene.npz", "scene.json")
    if any((args.output / name).exists() for name in names):
        raise FileExistsError(f"Use a fresh depth export directory: {args.output}")

    tick = time.perf_counter()
    dataset, repo = args.dataset.resolve(), args.repo.resolve()
    initializer = args.initializer.resolve()
    initializer = initializer / "scene.npz" if initializer.is_dir() else initializer
    manifest, original, metadata, paths, size, transforms, bounds, model_size = _load_inputs(dataset, initializer)
    source_sha = sha256_file(initializer)
    revision = _pinned_checkout(repo)
    weight_sha = sha256_file(args.weights)
    if revision != metadata["source_commit"] or weight_sha != metadata["weights_sha256"]:
        raise ValueError("Official VGGT files differ from the archived initializer provenance")
    target_path = args.target_initializer.resolve() if args.target_initializer else initializer
    target_path = target_path / "scene.npz" if target_path.is_dir() else target_path
    target = original if target_path == initializer else load_scene(target_path)
    radius = _validate_gauge(original, target)
    features, cache_hashes = _open_features(initializer.parent / "offload_cache", len(paths), model_size)
    print(f"Decoding cached DPT features for {len(paths)} training frames on CPU", flush=True)
    decode_started = time.perf_counter()
    depth, confidence, rgb, version = _decode(args.weights, repo, features, paths, model_size, args.threads)
    decode_seconds = time.perf_counter() - decode_started
    original_points, original_colors, original_filter = _cloud(original, depth, confidence, rgb, transforms, bounds, metadata)
    seed_replay = _replay_summary(original_points, original_colors, original, radius)
    prepared_depth, prepared_confidence = _prepared_geometry(depth, confidence, transforms, size)

    args.output.mkdir(parents=True, exist_ok=True)
    camera_sha = source_sha
    if args.export_cloud:
        points, colors, filtering = ((original_points, original_colors, original_filter) if target is original else
                                    _cloud(target, depth, confidence, rgb, transforms, bounds, metadata))
        scene_path = args.output / "scene.npz"
        np.savez_compressed(scene_path, **{**target, "points": points, "colors": colors})
        load_scene(scene_path)
        camera_sha = sha256_file(scene_path)
        write_json(args.output / "scene.json", {"method": "vggt_cached_dpt_reunprojection", "scene": manifest["scene"],
                                                "parent_initializer_sha256": source_sha, "scene_sha256": camera_sha,
                                                "depth_adjustment": "none; native VGGT camera-Z", **filtering})

    artifact = args.output / "depth_maps.npz"
    np.savez_compressed(artifact, names=original["names"], depth=prepared_depth,
                        confidence=prepared_confidence, split_hash=np.asarray(manifest["split_hash"]),
                        camera_initializer_sha256=np.asarray(camera_sha))
    receipt = {"scene": manifest["scene"], "split_hash": manifest["split_hash"],
               "manifest_sha256": metadata["manifest_sha256"], "training_ids": original["names"].tolist(),
               "source_metadata_sha256": sha256_file(initializer.with_suffix(".json")),
               "source_initializer_sha256": source_sha, "camera_initializer_sha256": camera_sha,
               "target_initializer_sha256": (sha256_file(target_path) if target is not original else source_sha),
               "depth_maps_sha256": sha256_file(artifact), "cache_sha256": cache_hashes,
               "vggt_commit": revision, "weights_sha256": weight_sha, "source": source_version(),
               "script_sha256": sha256_file(__file__), "torch": version, "cpu_threads": args.threads,
               "depth_units": "native VGGT camera-axis Z; not normalized or fit", "prepared_hw": [size[1], size[0]],
               "depth_resampling": "bilinear with canonical_to_model map at integer pixel centers",
               "confidence": "raw official DPT confidence", "reference_data_used": False,
               "original_seed_replay": {**original_filter, **seed_replay},
               "decode_seconds": decode_seconds, "elapsed_seconds": time.perf_counter() - tick}
    write_json(args.output / "depth_receipt.json", receipt)
    print(f"Saved {prepared_depth.shape} prepared depth maps to {artifact}", flush=True)


if __name__ == "__main__":
    main()

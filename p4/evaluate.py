"""Held-out rendering metrics and train-only reference camera alignment.

Reference poses never enter reconstruction. Render cameras are transformed by
one global similarity fitted using TRAIN correspondences; test RGBs and test
poses do not influence the fit. PSNR/SSIM use decoded RGB values in [0,1]; LPIPS
uses the AlexNet network with inputs mapped to [-1,1].
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from .data import IMAGE_EXTENSIONS, load_manifest, load_scene
from .geometry import align_training_cameras, transform_poses


def _json_alignment(alignment, reference_kind):
    def simple(value):
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, dict):
            return {k: simple(v) for k, v in value.items()}
        return value
    result = simple(alignment)
    result.update({"reference_kind": reference_kind, "alignment": "Sim3, train correspondences only",
                   "ate_units": "reference trajectory units (metric only if reference is metric)",
                   "test_pose_refinement": False})
    return result


def evaluation_cameras(dataset, scene_path):
    """Return ``{names,c2w,K,alignment}`` for exact held-out manifest frames.

    ``alignment`` is JSON-compatible and includes ATE RMSE in reference units.
    ``c2w`` cameras are in the estimated reconstruction coordinate frame.
    Missing test reference poses and unobservable train alignment fail explicitly.
    """
    dataset = Path(dataset)
    manifest = load_manifest(dataset)
    scene = load_scene(scene_path)
    train_names = {frame["image"] for frame in manifest["frames"] if frame["split"] == "train"}
    extra = set(scene["names"].tolist()) - train_names
    if extra:
        raise ValueError(f"Initializer contains non-train camera(s): {sorted(extra)[:5]}")
    reference_file = dataset / "reference.npz"
    if not reference_file.exists():
        raise FileNotFoundError("No reference.npz: test camera rendering and ATE unavailable")
    with np.load(reference_file, allow_pickle=False) as reference:
        names = reference["names"].astype(str)
        poses, intrinsics = reference["c2w"], reference["K"]
    if len(names) != len(set(names.tolist())):
        raise ValueError("Duplicate reference camera names")
    if poses.shape != (len(names), 4, 4) or intrinsics.shape != (len(names), 3, 3):
        raise ValueError("Invalid reference camera shapes")
    if not np.isfinite(poses).all() or not np.isfinite(intrinsics).all():
        raise ValueError("Nonfinite reference camera values")
    alignment = align_training_cameras(scene["names"], scene["c2w"], names, poses, train_names)
    lookup = {name: i for i, name in enumerate(names.tolist())}
    test = [frame["image"] for frame in manifest["frames"] if frame["split"] == "test"]
    missing = set(test) - lookup.keys()
    if missing:
        raise ValueError(f"Missing reference poses for test frames: {sorted(missing)[:5]}")
    if not test:
        raise ValueError("Empty held-out test set")
    indices = [lookup[name] for name in test]
    return {"names": np.asarray(test),
            "c2w": transform_poses(poses[indices], alignment["reference_to_estimate"]),
            "K": intrinsics[indices],
            "alignment": _json_alignment(alignment, manifest.get("reference_kind", "reference"))}


def rendering_metrics(prediction, target):
    """CPU PSNR and Gaussian-window SSIM, float64 RGB arrays in [0,1].

    SSIM follows an 11x11 Gaussian window (sigma=1.5), population covariance,
    valid-window averaging and data_range=1. PSNR has MSE floor 1e-12 (120 dB).
    """
    from skimage.metrics import structural_similarity
    prediction, target = np.asarray(prediction, dtype=np.float64), np.asarray(target, dtype=np.float64)
    if prediction.shape != target.shape or target.ndim != 3 or target.shape[2] != 3:
        raise ValueError("Metrics require corresponding H x W x 3 RGB images")
    if min(target.shape[:2]) < 11:
        raise ValueError("Images must be at least 11x11 for the frozen SSIM protocol")
    if not np.isfinite(prediction).all() or not np.isfinite(target).all():
        raise ValueError("Nonfinite metric input")
    if prediction.min() < 0 or prediction.max() > 1 or target.min() < 0 or target.max() > 1:
        raise ValueError("Metric inputs must be in [0,1]")
    mse = float(np.mean((prediction - target) ** 2))
    psnr = float(-10 * np.log10(max(mse, 1e-12)))
    ssim = float(structural_similarity(target, prediction, data_range=1.0,
                                      channel_axis=-1, gaussian_weights=True,
                                      sigma=1.5, use_sample_covariance=False))
    return {"psnr": psnr, "ssim": ssim}


def evaluate_renders(dataset, render_dir, lpips=True):
    """Measure exact held-out raw renders; report absent LPIPS explicitly.

    A rendered frame lives at ``render_dir / Path(frame['image']).with_suffix('.png')``.
    Missing and extra rendered image files fail rather than changing the test set.
    No rescaling, pose adjustment, postprocessing, masks, or exposure fitting is
    applied during scoring. Set ``lpips=False`` only for an explicitly partial run.
    """
    dataset, render_dir = Path(dataset), Path(render_dir)
    manifest = load_manifest(dataset)
    frames = [frame for frame in manifest["frames"] if frame["split"] == "test"]
    if not frames:
        raise ValueError("Empty held-out test set")
    expected = {Path(frame["image"]).with_suffix(".png").as_posix() for frame in frames}
    actual = {p.relative_to(render_dir).as_posix() for p in render_dir.rglob("*")
              if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS}
    if actual != expected:
        raise ValueError(f"Render set mismatch: missing={sorted(expected-actual)[:5]}, extra={sorted(actual-expected)[:5]}")
    network, lpips_status = None, "disabled; partial evaluation"
    if lpips:
        try:
            import torch
            import lpips as lpips_package
            network = lpips_package.LPIPS(net="alex", version="0.1").eval()
            lpips_status = "available; AlexNet v0.1, CPU, [-1,1]"
        except (ImportError, OSError, RuntimeError) as exc:
            # Missing models/download failures do not become invented LPIPS values.
            lpips_status = f"unavailable: {type(exc).__name__}: {exc}"
    per_frame = []
    for frame in frames:
        name = frame["image"]
        with Image.open(dataset / name) as im:
            target = np.asarray(im.convert("RGB"), dtype=np.float64) / 255.
        with Image.open(render_dir / Path(name).with_suffix(".png")) as im:
            prediction = np.asarray(im.convert("RGB"), dtype=np.float64) / 255.
        if target.shape != prediction.shape:
            raise ValueError(f"Render resolution mismatch: {name} {prediction.shape} != {target.shape}")
        metrics = rendering_metrics(prediction, target)
        metrics["lpips"] = None
        if network is not None:
            # float32, NCHW; network owns CPU tensors. Never optimize test input.
            pred_tensor = torch.from_numpy(prediction.astype(np.float32)).permute(2, 0, 1)[None] * 2 - 1
            target_tensor = torch.from_numpy(target.astype(np.float32)).permute(2, 0, 1)[None] * 2 - 1
            with torch.inference_mode():
                metrics["lpips"] = float(network(pred_tensor, target_tensor).item())
        per_frame.append({"id": frame["id"], "image": name, **metrics})
    averages = {key: float(np.mean([row[key] for row in per_frame])) if all(row[key] is not None for row in per_frame) else None
                for key in ("psnr", "ssim", "lpips")}
    return {"schema_version": 1, "scene": manifest["scene"], "split_hash": manifest["split_hash"],
            "test_frames": len(frames), **averages,
            "lpips_status": lpips_status, "complete": network is not None,
            "protocol": {"domain": "decoded RGB [0,1]", "psnr_mse_floor": 1e-12,
                         "ssim": "Gaussian 11x11 sigma=1.5; population covariance; valid-window mean",
                         "aggregation": "arithmetic mean of per-frame scores",
                         "test_rgb_in_reconstruction": False, "postprocessing": "none"},
            "per_frame": per_frame}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    poses = sub.add_parser("poses")
    poses.add_argument("--dataset", required=True)
    poses.add_argument("--scene", required=True)
    poses.add_argument("--output", required=True)
    images = sub.add_parser("renders")
    images.add_argument("--dataset", required=True)
    images.add_argument("--render-dir", required=True)
    images.add_argument("--output", required=True)
    images.add_argument("--no-lpips", action="store_true")
    args = parser.parse_args()
    if args.command == "poses":
        result = evaluation_cameras(args.dataset, args.scene)["alignment"]
    else:
        result = evaluate_renders(args.dataset, args.render_dir, lpips=not args.no_lpips)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()

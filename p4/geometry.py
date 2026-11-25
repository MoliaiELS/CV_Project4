"""OpenCV camera geometry and train-only similarity alignment (NumPy)."""
from __future__ import annotations

import numpy as np


def invert_poses(c2w):
    poses = np.asarray(c2w, dtype=np.float64)
    result = np.zeros_like(poses)
    result[..., :3, :3] = np.swapaxes(poses[..., :3, :3], -1, -2)
    result[..., :3, 3] = -np.einsum("...ij,...j->...i", result[..., :3, :3], poses[..., :3, 3])
    result[..., 3, 3] = 1
    return result


def fit_similarity(source, target):
    """Fit target = scale * rotation @ source + translation using Umeyama.

    At least three noncollinear correspondences are required. Rank-2 planar
    centers are sufficient; rank-1 trajectories do not identify global rotation.
    Reflections and zero/negative scale are rejected.
    """
    source, target = np.asarray(source, dtype=np.float64), np.asarray(target, dtype=np.float64)
    if source.shape != target.shape or source.ndim != 2 or source.shape[1] != 3 or len(source) < 3:
        raise ValueError("Sim3 alignment needs at least 3 corresponding 3D centers")
    if not np.isfinite(source).all() or not np.isfinite(target).all():
        raise ValueError("Nonfinite alignment coordinates")
    src_mean, dst_mean = source.mean(0), target.mean(0)
    src, dst = source - src_mean, target - dst_mean
    for label, values in (("source", src), ("target", dst)):
        singular = np.linalg.svd(values, compute_uv=False)
        if singular[0] < 1e-10 or singular[1] / singular[0] < 1e-5:
            raise ValueError(f"Degenerate {label} train camera centers: global Sim3 rotation unobservable")
    U, singular, Vt = np.linalg.svd(dst.T @ src / len(source))
    sign = np.ones(3)
    sign[-1] = np.linalg.det(U @ Vt)
    rotation = U @ np.diag(sign) @ Vt
    scale = float(np.sum(singular * sign) / np.mean(np.sum(src * src, axis=1)))
    if scale <= 1e-12:
        raise ValueError("Invalid Sim3 scale")
    translation = dst_mean - scale * rotation @ src_mean
    return {"scale": scale, "rotation": rotation, "translation": translation}


def transform_points(points, similarity):
    return (float(similarity["scale"]) * np.asarray(points) @ np.asarray(similarity["rotation"]).T
            + np.asarray(similarity["translation"]))


def transform_poses(c2w, similarity):
    poses = np.asarray(c2w, dtype=np.float64).copy()
    poses[..., :3, 3] = transform_points(poses[..., :3, 3], similarity)
    poses[..., :3, :3] = np.asarray(similarity["rotation"]) @ poses[..., :3, :3]
    return poses


def align_training_cameras(estimated_names, estimated_c2w, reference_names, reference_c2w, train_names):
    """Fit one reference-to-estimate Sim3 using common TRAIN frames only.

    This transform maps held-out reference cameras into the reconstruction frame.
    ATE is separately measured in reference units by fitting estimate-to-reference.
    Test camera centers are never used to estimate either transform.
    """
    estimated = dict(zip(map(str, estimated_names), np.asarray(estimated_c2w)))
    reference = dict(zip(map(str, reference_names), np.asarray(reference_c2w)))
    names = sorted(set(map(str, train_names)) & estimated.keys() & reference.keys())
    if len(names) < 3:
        raise ValueError("Need at least three matched train cameras for evaluation alignment")
    src = np.stack([reference[name][:3, 3] for name in names])
    dst = np.stack([estimated[name][:3, 3] for name in names])
    ref_to_est = fit_similarity(src, dst)
    est_to_ref = fit_similarity(dst, src)
    residual = transform_points(dst, est_to_ref) - src
    ate = float(np.sqrt(np.mean(np.sum(residual * residual, axis=1))))
    return {"reference_to_estimate": ref_to_est, "estimate_to_reference": est_to_ref,
            "train_names": names, "ate_rmse": ate, "per_frame_error": np.linalg.norm(residual, axis=1)}


def project_points(points, c2w, K):
    camera = np.asarray(points) @ c2w[:3, :3] - c2w[:3, :3].T @ c2w[:3, 3]
    pixels = camera @ np.asarray(K).T
    return pixels[:, :2] / pixels[:, 2:3], camera[:, 2]

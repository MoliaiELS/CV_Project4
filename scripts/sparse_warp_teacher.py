"""Build fixed in-between supervision from authenticated train RGB and DPT depth."""

from dataclasses import dataclass
import hashlib
import json

import numpy as np

from scripts.part3_views import Camera, Candidate, interpolate_views, read_rgb
from scripts.part3_warp_views import warp_real_prior


@dataclass(frozen=True)
class TeacherView:
    camera: Camera
    rgb: np.ndarray
    confidence: np.ndarray


def build_warp_teacher(dataset, cameras, depth_names, depth, confidence, radius, *, edge=256):
    """Generate every 1/3, 1/2, and 2/3 view between the supplied train cameras."""
    if edge < 8 or not np.isfinite(radius) or radius <= 0:
        raise ValueError("Invalid teacher render size or geometry normalization radius")
    if (len(depth_names) != len(cameras) or len(set(depth_names)) != len(depth_names)
            or {camera.name for camera in cameras} != set(depth_names)
            or np.shape(depth) != np.shape(confidence) or len(depth) != len(depth_names)):
        raise ValueError("Warp teacher requires exactly one DPT map for each real training camera")
    indices = {name: index for index, name in enumerate(depth_names)}
    anchors = {}
    for camera in cameras:
        index = indices[camera.name]
        z, score = depth[index], confidence[index]
        if z.shape != (camera.height, camera.width):
            raise ValueError(f"DPT depth grid differs from training image {camera.name}")
        valid = np.isfinite(z) & (z > 0.) & np.isfinite(score) & (score > 0.)
        if not valid.any():
            raise ValueError(f"No valid DPT depth for training image {camera.name}")
        weak, strong = np.quantile(score[valid], (0.10, 0.50))
        if strong > weak:
            alpha = np.where(valid, np.clip((score - weak) / (strong - weak), 0., 1.), 0.)
        else:
            alpha = (valid & (score >= strong)).astype(np.float32)
        geometry = {"depth": np.where(valid, z / radius, 0.), "alpha": alpha}
        anchors[camera.name] = (camera, geometry, read_rgb(dataset / camera.name))

    observations, details = [], []
    digest = hashlib.sha256()
    for original in interpolate_views(cameras):
        view = original.camera.render_at(edge)
        candidate = Candidate(view, original.anchors, original.interval, original.fraction)
        anchor_names = [camera.name for camera in candidate.anchors]
        prior = warp_real_prior(candidate, [anchors[name] for name in anchor_names], radius)
        observation = TeacherView(view, prior["rgb"], prior["confidence"])
        record = {"name": view.name, "anchors": anchor_names, "fraction": candidate.fraction,
                  "width": view.width, "height": view.height, "c2w": view.c2w.tolist(), "K": view.K.tolist(),
                  **prior["stats"]}
        digest.update(json.dumps(record, sort_keys=True, allow_nan=False).encode())
        for array in (observation.rgb, observation.confidence):
            digest.update(np.ascontiguousarray(array, dtype="<f4").view(np.uint8))
        observations.append(observation)
        details.append(record)
    if not observations or not any(record["prior_coverage"] > 0 for record in details):
        raise ValueError("No supported real-color projections in the in-between views")
    return observations, {
        "teacher_sha256": digest.hexdigest(), "max_edge": edge, "views": len(observations),
        "mean_coverage": float(np.mean([record["prior_coverage"] for record in details])),
        "mean_confidence": float(np.mean([record["prior_mean_confidence"] for record in details])),
        "source": "real sparse RGB forward warped with authenticated train DPT camera-Z",
        "policy": "all sparse train intervals, fractions 1/3, 1/2, and 2/3", "per_view": details,
    }


def warp_teacher_loss(rgb, target, confidence):
    """Robust photometric loss, with a support floor to bound sparse-mask amplification."""
    from torch.nn import functional as F

    residual = F.smooth_l1_loss(rgb, target, reduction="none", beta=.10).mean(dim=-1)
    return (confidence * residual).sum() / confidence.sum().clamp_min(.25 * confidence.numel())

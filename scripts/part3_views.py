"""Plan training-only views and check diffusion edits against real observations."""
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import map_coordinates
from scipy.spatial.transform import Rotation, Slerp

from p4.data import load_manifest, load_scene


@dataclass(frozen=True)
class Camera:
    name: str
    c2w: np.ndarray
    K: np.ndarray
    width: int
    height: int

    def render_at(self, limit):
        scale = min(1.0, limit / max(self.width, self.height))
        width, height = round(self.width * scale), round(self.height * scale)
        K = np.diag([width / self.width, height / self.height, 1.0]) @ self.K
        return Camera(self.name, self.c2w, K, width, height)


@dataclass(frozen=True)
class Candidate:
    camera: Camera
    anchors: tuple[Camera, Camera]
    interval: int
    fraction: float

    @property
    def reference(self):
        return self.anchors[int(self.fraction > 0.5)]


def training_cameras(dataset, scene_path):
    scene = load_scene(scene_path)
    manifest = load_manifest(dataset)
    train = [frame for frame in manifest["frames"] if frame["split"] == "train"]
    lookup = {name: i for i, name in enumerate(scene["names"].tolist())}
    if set(lookup) != {frame["image"] for frame in train}:
        raise ValueError("Part 3 view planning requires the exact sparse training cameras")
    return [
        Camera(frame["image"], scene["c2w"][lookup[frame["image"]]], scene["K"][lookup[frame["image"]]],
               frame["width"], frame["height"])
        for frame in train
    ]


def interpolate_views(cameras):
    candidates = []
    for interval, (left, right) in enumerate(zip(cameras, cameras[1:])):
        if (left.width, left.height) != (right.width, right.height):
            raise ValueError("Both anchors must use the same image resolution")
        rotations = Rotation.from_matrix(np.stack([left.c2w[:3, :3], right.c2w[:3, :3]]))
        interpolate_rotation = Slerp([0., 1.], rotations)
        for fraction in (1 / 3, 1 / 2, 2 / 3):
            pose = np.eye(4)
            pose[:3, :3] = interpolate_rotation(fraction).as_matrix()
            pose[:3, 3] = (1 - fraction) * left.c2w[:3, 3] + fraction * right.c2w[:3, 3]
            K = (1 - fraction) * left.K + fraction * right.K
            name = f"interval_{interval:03d}_{round(fraction * 100):02d}"
            candidates.append(Candidate(Camera(name, pose, K, left.width, left.height),
                                        (left, right), interval, fraction))
    return candidates


def select_views(candidates, render, count, policy="active"):
    """Prefer poorly supported views, covering distinct anchor intervals."""
    if count < 1 or count > len({view.interval for view in candidates}):
        raise ValueError("Requested views must fit in distinct sparse intervals")
    if policy == "fixed":
        midpoints = [candidate for candidate in candidates if candidate.fraction == 0.5]
        indices = np.linspace(0, len(midpoints) - 1, count, dtype=int)
        return [midpoints[index] for index in indices], []
    if policy != "active":
        raise ValueError(f"Unknown view policy: {policy}")
    scores = []
    for candidate in candidates:
        alpha = render(candidate.camera.render_at(256))["alpha"]
        if not np.isfinite(alpha).all():
            raise ValueError(f"Nonfinite opacity for {candidate.camera.name}")
        hole_fraction = float(np.mean(alpha < 0.5))
        transmittance = float(np.mean(1 - np.clip(alpha, 0, 1)))
        # Very wide or difficult pairs are useful only when part of the view
        # still has enough support for meaningful reference conditioning.
        score = (0.7 * hole_fraction + 0.3 * transmittance) * min(1., (1. - hole_fraction) / 0.2)
        scores.append({"target": candidate.camera.name, "hole_fraction": hole_fraction,
                       "mean_transmittance": transmittance, "score": score})
    ranked = sorted(zip(candidates, scores), key=lambda item: (-item[1]["score"], item[0].camera.name))
    chosen, covered = [], set()
    for candidate, _ in ranked:
        if candidate.interval in covered:
            continue
        chosen.append(candidate)
        covered.add(candidate.interval)
        if len(chosen) == count:
            break
    return sorted(chosen, key=lambda candidate: candidate.interval), scores


def _reproject(target, target_depth, anchor, anchor_render, normalization_radius):
    """Project target gsplat z-depth into an anchor using consistent normalization."""
    height, width = target_depth.shape
    gy, gx = np.indices((height, width), dtype=np.float32)
    pixels = np.stack([gx + 0.5, gy + 0.5, np.ones_like(gx)], axis=-1)
    z = np.where(np.isfinite(target_depth), target_depth, 0).astype(np.float32)
    target_points = (pixels @ np.linalg.inv(target.K).T) * z[..., None]
    offset = (target.c2w[:3, 3] - anchor.c2w[:3, 3]) / normalization_radius
    rotation = target.c2w[:3, :3].T @ anchor.c2w[:3, :3]
    anchor_points = target_points @ rotation + (offset @ anchor.c2w[:3, :3])
    z_anchor = anchor_points[..., 2]
    projected = anchor_points @ anchor.K.T
    with np.errstate(divide="ignore", invalid="ignore"):
        coordinates = projected[..., :2] / z_anchor[..., None] - 0.5
    x, y = np.moveaxis(np.nan_to_num(coordinates, nan=-1., posinf=-1., neginf=-1.), -1, 0)
    valid = ((z > 0) & (z_anchor > 0) & (x >= 0) & (x <= anchor.width - 1) &
             (y >= 0) & (y <= anchor.height - 1))
    xy = [y, x]
    anchor_depth = map_coordinates(anchor_render["depth"], xy, order=1, mode="constant", cval=0)
    anchor_alpha = map_coordinates(anchor_render["alpha"], xy, order=1, mode="constant", cval=0)
    depth_disagreement = np.abs(anchor_depth - z_anchor) / np.maximum(z_anchor, 1e-5)
    visible = valid & (anchor_alpha >= 0.45) & (anchor_depth > 0) & (depth_disagreement < 0.15)
    return xy, visible


def confidence_map(candidate, render, enhanced, anchor_inputs, normalization_radius):
    """Softly weight visible consistent edits; retain weak weight where uncertain."""
    if normalization_radius <= 0:
        raise ValueError("Invalid normalization for cross-view depth")
    target_supported = render["alpha"] >= 0.45
    best_error = np.full(target_supported.shape, np.inf, dtype=np.float32)
    evidence = np.zeros_like(best_error, dtype=np.uint8)
    for anchor, anchor_render, anchor_rgb in anchor_inputs:
        coordinates, visible = _reproject(candidate.camera, render["depth"], anchor, anchor_render,
                                           normalization_radius)
        warped = np.stack([map_coordinates(anchor_rgb[..., channel], coordinates, order=1, mode="constant",
                                           cval=0) for channel in range(3)], axis=-1)
        residual = np.mean(np.abs(warped - enhanced), axis=-1)
        visible &= target_supported
        evidence += visible
        best_error = np.minimum(best_error, np.where(visible, residual, np.inf))
    verified = evidence > 0
    edit = np.mean(np.abs(enhanced - render["rgb"]), axis=-1)
    unknown_weight = 0.20 * np.exp(-edit / 0.25)
    confidence = np.where(verified, np.exp(-(best_error / 0.18) ** 2), unknown_weight).astype(np.float32)
    stats = {
        "verified_fraction": float(verified.mean()),
        "two_anchor_fraction": float((evidence == 2).mean()),
        "mean_confidence": float(confidence.mean()),
        "low_confidence_fraction": float((confidence < 0.2).mean()),
        "mean_edit": float(edit.mean()),
    }
    return confidence, stats


def read_rgb(path):
    with Image.open(Path(path)) as image:
        return np.asarray(image.convert("RGB"), dtype=np.float32) / 255.


def save_rgb(path, values):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.rint(np.clip(values, 0, 1) * 255).astype(np.uint8)).save(path)

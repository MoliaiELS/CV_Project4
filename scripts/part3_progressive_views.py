"""Plan progressive virtual views and fuse two independently restored images."""

import numpy as np
from scipy.ndimage import map_coordinates
from scipy.spatial.transform import Rotation, Slerp

from scripts.part3_views import Camera, Candidate, _reproject


FRACTIONS = {1: (0.20, 0.80), 2: (1 / 3, 0.50, 2 / 3)}
PLANNING_EDGE = 256
TARGET_ALPHA_MIN = 0.20


def progressive_candidates(cameras, round_index):
    """Use near-anchor interpolants first and move toward interval interiors in round two."""
    if round_index not in FRACTIONS:
        raise ValueError("Progressive view planning supports rounds one and two")
    candidates = []
    for interval, (left, right) in enumerate(zip(cameras, cameras[1:])):
        if (left.width, left.height) != (right.width, right.height):
            raise ValueError("Both anchors must use the same image resolution")
        rotations = Rotation.from_matrix(np.stack([left.c2w[:3, :3], right.c2w[:3, :3]]))
        interpolate_rotation = Slerp([0., 1.], rotations)
        for fraction in FRACTIONS[round_index]:
            pose = np.eye(4)
            pose[:3, :3] = interpolate_rotation(fraction).as_matrix()
            pose[:3, 3] = (1 - fraction) * left.c2w[:3, 3] + fraction * right.c2w[:3, 3]
            K = (1 - fraction) * np.asarray(left.K, dtype=np.float64) + fraction * right.K
            name = f"interval_{interval:03d}_{round(fraction * 100):02d}"
            camera = Camera(name, pose, K, left.width, left.height)
            candidates.append(Candidate(camera, (left, right), interval, fraction))
    return candidates


def _geometry(camera, render):
    shape = (camera.height, camera.width)
    alpha, depth = (np.asarray(render[key], dtype=np.float32) for key in ("alpha", "depth"))
    if alpha.shape != shape or depth.shape != shape:
        raise ValueError(f"Render geometry shape does not match {camera.name}")
    alpha = np.clip(np.where(np.isfinite(alpha), alpha, 0.), 0., 1.)
    depth = np.where(np.isfinite(depth) & (depth > 0), depth, 0.)
    return {"alpha": alpha, "depth": depth}


def _project_to_anchor(target, target_geometry, anchor, anchor_geometry, radius):
    coordinates, visible = _reproject(target, target_geometry["depth"], anchor, anchor_geometry, radius)
    return coordinates, visible & (target_geometry["alpha"] >= TARGET_ALPHA_MIN)


def _validate_radius(radius):
    if not np.isfinite(radius) or radius <= 0:
        raise ValueError("Scene normalization radius must be finite and positive")


def select_progressive_views(candidates, render, count, normalization_radius):
    """Select distinct intervals with deficits that real anchors can verify."""
    _validate_radius(normalization_radius)
    if not 1 <= count <= len({view.interval for view in candidates}):
        raise ValueError("Requested views must fit in distinct sparse intervals")
    anchors = {}
    scores = []
    for view in candidates:
        target = view.camera.render_at(PLANNING_EDGE)
        geometry = _geometry(target, render(target))
        visible = []
        for anchor in view.anchors:
            if anchor.name not in anchors:
                low = anchor.render_at(PLANNING_EDGE)
                anchors[anchor.name] = low, _geometry(low, render(low))
            low, anchor_geometry = anchors[anchor.name]
            _, support = _project_to_anchor(target, geometry, low, anchor_geometry, normalization_radius)
            visible.append(support)
        supported = visible[0] | visible[1]
        bilateral = visible[0] & visible[1]
        evidence = 0.4 * supported + 0.6 * bilateral
        deficit = 1 - geometry["alpha"]
        # Unsupported holes may have unreliable target depths. Favor deficits
        # with observable real evidence and give bilateral projections priority.
        score = np.mean((deficit + 0.03) * evidence)
        scores.append({"target": view.camera.name, "interval": view.interval, "fraction": view.fraction,
                       "hole_fraction": float((geometry["alpha"] < 0.5).mean()),
                       "mean_transmittance": float(deficit.mean()), "supported_fraction": float(supported.mean()),
                       "bilateral_fraction": float(bilateral.mean()),
                       "supported_deficit": float((deficit * supported).mean()), "score": float(score)})
    ranked = sorted(zip(candidates, scores),
                    key=lambda item: (-item[1]["score"], -item[1]["bilateral_fraction"], item[0].camera.name))
    chosen, visited = [], set()
    for view, _ in ranked:
        if view.interval not in visited:
            chosen.append(view)
            visited.add(view.interval)
        if len(chosen) == count:
            break
    return sorted(chosen, key=lambda view: view.interval), scores


def _rgb(camera, image):
    image = np.asarray(image, dtype=np.float32)
    if (image.shape != (camera.height, camera.width, 3) or not np.isfinite(image).all()
            or image.min() < 0. or image.max() > 1.):
        raise ValueError(f"Expected finite H×W×3 RGB in [0,1] for {camera.name}")
    return image


def _color_distance(first, second):
    return np.mean(np.abs(first - second), axis=-1)


def fuse_restorations(candidate, raw, restored_pair, anchor_inputs, normalization_radius):
    """Fuse supported residuals from independent left- and right-reference restorations.

    Each edit must improve agreement with its own train anchor, and a visible
    second anchor can veto it. Unsupported pixels keep the raw RGB and carry
    zero pseudo-loss weight. Single-anchor evidence can contribute at most 0.4.
    """
    _validate_radius(normalization_radius)
    if len(restored_pair) != 2 or len(anchor_inputs) != 2:
        raise ValueError("Bilateral fusion requires two restorations and their train anchors")
    target = candidate.camera
    raw_rgb = _rgb(target, raw["rgb"])
    target_geometry = _geometry(target, raw)
    restorations = [_rgb(target, image) for image in restored_pair]
    warped, visible = [], []
    for expected, (anchor, render, real_rgb) in zip(candidate.anchors, anchor_inputs):
        if anchor.name != expected.name:
            raise ValueError("Anchor inputs must be ordered like the candidate's two real cameras")
        xy, valid = _project_to_anchor(target, target_geometry, anchor, _geometry(anchor, render), normalization_radius)
        real_rgb = _rgb(anchor, real_rgb)
        warped.append(np.stack([map_coordinates(channel, xy, order=1, mode="constant", cval=0.)
                                for channel in np.moveaxis(real_rgb, -1, 0)], axis=-1))
        visible.append(valid)

    weights = []
    raw_error = [_color_distance(raw_rgb, donor) for donor in warped]
    for index, restoration in enumerate(restorations):
        other = 1 - index
        error = _color_distance(restoration, warped[index])
        gain = raw_error[index] - error
        need = np.clip((raw_error[index] - 0.02) / 0.05, 0., 1.)
        supported_gain = np.clip((gain - 0.002) / 0.025, 0., 1.)
        cross_gain = raw_error[other] - _color_distance(restoration, warped[other])
        cross_check = np.where(visible[other], np.clip((cross_gain + 0.008) / 0.025, 0., 1.), 1.)
        weights.append(visible[index] * need * supported_gain * np.exp(-(error / 0.12) ** 2) * cross_check)

    highest = np.maximum(*weights)
    total = weights[0] + weights[1]
    agreement = np.exp(-(_color_distance(*restorations) / 0.06) ** 2)
    donor_agreement = np.exp(-(_color_distance(*warped) / 0.13) ** 2)
    bilateral = visible[0] & visible[1]
    support = visible[0] | visible[1]
    confidence = np.where(bilateral, np.sqrt(highest * total * 0.5) * agreement * donor_agreement,
                          0.40 * highest).astype(np.float32)
    accepted = confidence >= 0.03
    confidence[~accepted] = 0
    residual = sum(weight[..., None] * (image - raw_rgb) for weight, image in zip(weights, restorations))
    residual /= np.maximum(total[..., None], 1e-8)
    enhanced = np.clip(raw_rgb + residual * accepted[..., None], 0., 1.)
    stats = {"verified_fraction": float(support.mean()), "two_anchor_fraction": float(bilateral.mean()),
             "mean_confidence": float(confidence.mean()), "low_confidence_fraction": float((confidence < 0.2).mean()),
             "mean_edit": float(_color_distance(enhanced, raw_rgb).mean()), "edit_fraction": float(accepted.mean()),
             "bilateral_edit_fraction": float((accepted & bilateral).mean()),
             "single_anchor_edit_fraction": float((accepted & support & ~bilateral).mean())}
    return enhanced.astype(np.float32, copy=False), confidence, stats

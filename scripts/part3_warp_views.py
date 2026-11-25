"""Forward-project real anchor colors into virtual views using current 3DGS depth."""

import numpy as np
from scipy.ndimage import maximum_filter, minimum_filter

from scripts.part3_progressive_views import _geometry, _rgb, _validate_radius


SOURCE_ALPHA_MIN = 0.55
DEPTH_EDGE_TOLERANCE = 0.10
SPLAT_DEPTH_TOLERANCE = 0.05
PAIR_DEPTH_TOLERANCE = 0.08
PAIR_COLOR_TOLERANCE = 0.10
SINGLE_DONOR_WEIGHT = 0.40


def _neighbors(xy, width, height):
    base = np.floor(xy).astype(np.int32)
    fraction = xy - base.astype(np.float32)
    for row in (0, 1):
        for col in (0, 1):
            indices = base + [col, row]
            x, y = indices.T
            weight_x = fraction[:, 0] if col else 1 - fraction[:, 0]
            weight_y = fraction[:, 1] if row else 1 - fraction[:, 1]
            weight = weight_x * weight_y
            # Prevent a tiny corner contribution from winning the z-buffer.
            hit = (x >= 0) & (x < width) & (y >= 0) & (y < height) & (weight >= 0.02)
            yield np.flatnonzero(hit), y[hit] * width + x[hit], weight[hit]


def _project(source, target, y, x, depth, radius):
    pixels = np.stack((x + 0.5, y + 0.5, np.ones_like(x)), axis=1).astype(np.float32)
    source_points = (pixels @ np.linalg.inv(source.K).T.astype(np.float32)) * depth[:, None]
    rotation = (source.c2w[:3, :3].T @ target.c2w[:3, :3]).astype(np.float32)
    offset = (source.c2w[:3, 3] - target.c2w[:3, 3]) / radius
    translation = (offset @ target.c2w[:3, :3]).astype(np.float32)
    target_points = source_points @ rotation + translation
    target_depth = target_points[:, 2]
    projected = target_points @ np.asarray(target.K.T, dtype=np.float32)
    with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
        xy = projected[:, :2] / target_depth[:, None] - 0.5
    valid = (np.isfinite(xy).all(axis=1) & np.isfinite(target_depth) & (target_depth > 0.01)
             & (xy[:, 0] > -1) & (xy[:, 0] < target.width) & (xy[:, 1] > -1) & (xy[:, 1] < target.height))
    return xy[valid], target_depth[valid], valid


def forward_splat_rgb(source, target, source_render, real_rgb, normalization_radius):
    """Splat real source RGB with a two-pass z-buffer and depth-edge rejection.

    Source depth is gsplat expected camera Z in the normalized scene space.
    Returned RGB and depth are undefined where the returned support is false.
    """
    _validate_radius(normalization_radius)
    geometry = _geometry(source, source_render)
    real_rgb = _rgb(source, real_rgb)
    alpha, depth = geometry["alpha"], geometry["depth"]
    valid = (alpha >= SOURCE_ALPHA_MIN) & (depth > 0.01)
    near = minimum_filter(np.where(valid, depth, 0.), size=3, mode="nearest")
    far = maximum_filter(np.where(valid, depth, 0.), size=3, mode="nearest")
    usable = valid & (near > 0) & ((far - near) <= DEPTH_EDGE_TOLERANCE * depth)
    y, x = np.nonzero(usable)
    xy, z, projected = _project(source, target, y, x, depth[usable], normalization_radius)
    colors = real_rgb[y[projected], x[projected]]
    source_weight = alpha[y[projected], x[projected]]

    width, height = target.width, target.height
    pixels = width * height
    z_buffer = np.full(pixels, np.inf, dtype=np.float32)
    for points, bins, _ in _neighbors(xy, width, height):
        np.minimum.at(z_buffer, bins, z[points])

    mass = np.zeros(pixels, dtype=np.float32)
    depth_sum = np.zeros(pixels, dtype=np.float32)
    rgb_sum = np.zeros((pixels, 3), dtype=np.float32)
    for points, bins, bilinear in _neighbors(xy, width, height):
        z_nearest = z_buffer[bins]
        separation = (z[points] - z_nearest) / z_nearest
        front = separation <= SPLAT_DEPTH_TOLERANCE
        points, bins = points[front], bins[front]
        weight = (bilinear[front] * source_weight[points]
                  * np.exp(-0.5 * (separation[front] / SPLAT_DEPTH_TOLERANCE) ** 2))
        np.add.at(mass, bins, weight)
        np.add.at(depth_sum, bins, weight * z[points])
        np.add.at(rgb_sum, bins, weight[:, None] * colors[points])

    support = mass >= 0.14
    rgb = np.divide(rgb_sum, mass[:, None], out=np.zeros_like(rgb_sum), where=support[:, None])
    z = np.divide(depth_sum, mass, out=np.zeros_like(depth_sum), where=support)
    confidence = np.where(support, np.minimum(mass, 1.), 0.)
    return {"rgb": np.clip(rgb.reshape(height, width, 3), 0., 1.), "depth": z.reshape(height, width),
            "confidence": confidence.reshape(height, width), "support": support.reshape(height, width),
            "stats": {"valid_source_fraction": float(valid.mean()),
                      "edge_rejected_source_fraction": float((valid & ~usable).mean()),
                      "projected_source_fraction": float(projected.sum() / valid.size),
                      "target_support_fraction": float(support.mean())}}


def warp_real_prior(candidate, anchor_inputs, normalization_radius):
    """Combine two z-resolved real-color projections and expose their support.

    A single donor contributes at most 0.4 confidence. Overlap with disagreeing
    color or depth remains unresolved for the later completion stage.
    """
    if len(anchor_inputs) != 2:
        raise ValueError("Real-color prior requires the two neighboring train anchors")
    views = []
    for expected, (anchor, source_render, real_rgb) in zip(candidate.anchors, anchor_inputs):
        if anchor.name != expected.name:
            raise ValueError("Anchor inputs must be ordered like the candidate's two real cameras")
        views.append(forward_splat_rgb(anchor, candidate.camera, source_render, real_rgb, normalization_radius))
    left, right = views
    source_support = np.stack([view["support"] for view in views])
    support_count = source_support.sum(axis=0, dtype=np.uint8)
    overlap = support_count == 2
    left_depth, right_depth = left["depth"], right["depth"]
    depth_delta = np.abs(left_depth - right_depth) / np.maximum(np.minimum(left_depth, right_depth), 1e-5)
    color_delta = np.mean(np.abs(left["rgb"] - right["rgb"]), axis=-1)
    agreement = overlap & (depth_delta <= PAIR_DEPTH_TOLERANCE) & (color_delta <= PAIR_COLOR_TOLERANCE)
    single = support_count == 1
    accepted = agreement | single

    fraction = candidate.fraction
    left_weight = left["confidence"] * (1.15 - fraction)
    right_weight = right["confidence"] * (0.15 + fraction)
    total_weight = left_weight + right_weight
    mixed_rgb = left["rgb"] * left_weight[..., None] + right["rgb"] * right_weight[..., None]
    rgb = np.divide(mixed_rgb, total_weight[..., None], out=np.zeros_like(mixed_rgb), where=accepted[..., None])
    depth = np.divide(left_depth * left_weight + right_depth * right_weight, total_weight,
                      out=np.zeros_like(total_weight), where=accepted)

    pair_weight = np.zeros_like(total_weight)
    pair_weight[agreement] = (
        np.sqrt(left["confidence"][agreement] * right["confidence"][agreement])
        * np.exp(-0.5 * (depth_delta[agreement] / 0.06) ** 2 - 0.5 * (color_delta[agreement] / 0.08) ** 2)
    )
    one_donor = SINGLE_DONOR_WEIGHT * np.maximum(left["confidence"], right["confidence"])
    confidence = np.where(single, one_donor, pair_weight)
    return {"rgb": np.clip(rgb, 0., 1.), "confidence": confidence, "depth": depth,
            "support_count": support_count, "source_support": source_support, "bilateral_agreement": agreement,
            "stats": {"source_support": [view["stats"] for view in views],
                      "prior_coverage": float(accepted.mean()), "single_donor_fraction": float(single.mean()),
                      "two_projected_fraction": float(overlap.mean()),
                      "bilateral_agreement_fraction": float(agreement.mean()),
                      "conflicting_overlap_fraction": float((overlap & ~agreement).mean()),
                      "prior_mean_confidence": float(confidence.mean()),
                      "prior_mean_active_confidence": float(confidence[accepted].mean()) if accepted.any() else 0.}}

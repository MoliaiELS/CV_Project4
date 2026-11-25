"""Refine sparse train cameras with RGB tracks and robust bundle adjustment."""

import argparse
import hashlib
from itertools import combinations
from pathlib import Path
import time

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix
from scipy.spatial.transform import Rotation

from p4.common import sha256_file, source_version, write_json
from p4.data import load_manifest, load_scene


SEED = 42
VALIDATION_ERROR_CAP = 30.0


def _training_features(dataset, frames):
    sift = cv2.SIFT_create(nfeatures=2000)
    features = []
    for frame in frames:
        path = (dataset / frame["image"]).resolve()
        if not path.is_relative_to(dataset):
            raise ValueError(f"Training image is outside the prepared dataset: {frame['image']}")
        content = path.read_bytes()
        if hashlib.sha256(content).hexdigest() != frame["sha256"]:
            raise ValueError(f"Training image does not match the manifest: {frame['image']}")
        image = cv2.imdecode(np.frombuffer(content, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
        if image is None or image.shape != (frame["height"], frame["width"]):
            raise ValueError(f"Invalid training image dimensions: {frame['image']}")
        keys, descriptors = sift.detectAndCompute(image, None)
        if descriptors is None or len(descriptors) < 12:
            raise ValueError(f"Too few SIFT features in training image: {frame['image']}")
        features.append((np.asarray([key.pt for key in keys]), descriptors))
    return features


def _ratio_matches(matcher, left, right):
    return {first.queryIdx: (first.trainIdx, first.distance / second.distance)
            for first, second in matcher.knnMatch(left, right, k=2)
            if first.distance < 0.78 * second.distance}


def _build_tracks(features):
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    edges, pairs = [], []
    for i, j in combinations(range(len(features)), 2):
        xy1, d1 = features[i]
        xy2, d2 = features[j]
        forward = _ratio_matches(matcher, d1, d2)
        reverse = _ratio_matches(matcher, d2, d1)
        matches = [(a, b, max(ratio, reverse[b][1])) for a, (b, ratio) in forward.items()
                   if b in reverse and reverse[b][0] == a]
        if len(matches) < 12:
            continue
        p = np.asarray([xy1[a] for a, _, _ in matches])
        q = np.asarray([xy2[b] for _, b, _ in matches])
        fundamental, inliers = cv2.findFundamentalMat(p, q, cv2.USAC_MAGSAC, 1.25, 0.999, 5000)
        if fundamental is None or inliers is None or np.count_nonzero(inliers) < 12:
            continue
        pairs.append({"frames": [i, j], "matches": len(matches), "inliers": int(np.count_nonzero(inliers))})
        edges.extend((quality, (i, a), (j, b)) for keep, (a, b, quality)
                     in zip(inliers.ravel(), matches) if keep)

    # A track contains at most one observation per image. Strong matches get
    # priority if two proposed unions disagree about the same image.
    parents, members = {}, {}

    def root(node):
        if node not in parents:
            parents[node] = node
            members[node] = {node[0]: node[1]}
        while parents[node] != node:
            parents[node] = parents[parents[node]]
            node = parents[node]
        return node

    for _, left, right in sorted(edges):
        a, b = root(left), root(right)
        if a == b or members[a].keys() & members[b].keys():
            continue
        if len(members[a]) < len(members[b]):
            a, b = b, a
        parents[b] = a
        members[a].update(members.pop(b))

    tracks = []
    for observations in members.values():
        if len(observations) >= 2:
            frames = np.asarray(sorted(observations), dtype=np.int64)
            pixels = np.asarray([features[i][0][observations[i]] for i in frames])
            tracks.append((frames, pixels))
    return tracks, pairs


def _projection_matrices(poses, K):
    return K @ np.linalg.inv(poses)[:, :3, :]


def _project(points, indices, poses, K):
    rotations = poses[indices, :3, :3]
    camera = np.einsum("ni,nij->nj", points - poses[indices, :3, 3], rotations)
    pixels = np.einsum("nij,nj->ni", K[indices], camera)
    return pixels[:, :2] / np.maximum(camera[:, 2:3], 1e-5), camera[:, 2]


def _triangulate(track, anchors, projections):
    frames, xy = track
    i, j = anchors
    point = cv2.triangulatePoints(projections[frames[i]], projections[frames[j]],
                                  xy[i].reshape(2, 1), xy[j].reshape(2, 1)).ravel()
    if abs(point[3]) <= 1e-10:
        return None
    result = point[:3] / point[3]
    return result if np.isfinite(result).all() and np.linalg.norm(result) <= 150.0 else None


def _split_tracks(tracks, poses, K):
    train, initial_points, validation = [], [], []
    projections = _projection_matrices(poses, K)
    centers = poses[:, :3, 3]
    validation_budget = min(160, sum(len(track[0]) >= 3 for track in tracks) // 5)
    for index in np.random.default_rng(SEED).permutation(len(tracks)):
        track = tracks[index]
        frames, xy = track
        anchors = max(combinations(range(len(frames)), 2),
                      key=lambda ij: np.linalg.norm(centers[frames[ij[0]]] - centers[frames[ij[1]]]))
        point = _triangulate(track, anchors, projections)
        if point is None:
            continue
        projected, depths = _project(point[None], frames, poses, K)
        if np.any(depths < 0.005):
            continue
        if len(frames) >= 3 and len(validation) < validation_budget:
            validation.append((track, anchors))
            continue
        error = np.linalg.norm(projected - xy, axis=1)
        if len(train) < 1000 and np.median(error) <= 7.0 and np.max(error) <= 18.0:
            initial_points.append(point)
            train.append(track)
    return train, np.asarray(initial_points), validation


def _bundle_adjust(poses, K, train, initial_points, gauge_anchor, fixed_focal):
    count = len(poses)
    frames = np.concatenate([track[0] for track in train])
    xy = np.concatenate([track[1] for track in train])
    point_ids = np.repeat(np.arange(len(train)), [len(track[0]) for track in train])
    coverage = np.bincount(frames, minlength=count)
    connected = {0}
    for _ in range(count):
        for observed, _ in train:
            if connected.intersection(observed):
                connected.update(observed)
    if np.min(coverage) < 8 or len(connected) != count:
        raise ValueError(f"Insufficient connected track observations per training camera: {coverage.tolist()}")

    # The first pose fixes the reference frame. Fixing the distant anchor's
    # center preserves scale and limits changes to the source scene gauge.
    rot_views = np.arange(1, count)
    center_views = np.asarray([i for i in rot_views if i != gauge_anchor])
    rot_size, center_size = 3 * len(rot_views), 3 * len(center_views)
    pose_size = rot_size + center_size
    camera_size = pose_size + int(not fixed_focal)
    priors = np.r_[np.full(rot_size, np.deg2rad(4)), np.full(center_size, 0.05)]
    limits = np.r_[np.full(rot_size, np.deg2rad(12)), np.full(center_size, 0.12)]
    lower, upper = -limits, limits.copy()
    if not fixed_focal:
        priors = np.r_[priors, 0.12]
        lower, upper = np.r_[lower, np.log(0.70)], np.r_[upper, np.log(1.35)]

    def unpack(x):
        fitted = poses.copy()
        fitted[rot_views, :3, :3] = poses[rot_views, :3, :3] @ Rotation.from_rotvec(
            x[:rot_size].reshape(-1, 3)).as_matrix()
        fitted[center_views, :3, 3] += x[rot_size:pose_size].reshape(-1, 3)
        scale = 1.0 if fixed_focal else float(np.exp(x[pose_size]))
        intrinsics = K.copy()
        intrinsics[:, [0, 1], [0, 1]] *= scale
        return fitted, intrinsics, x[camera_size:].reshape(-1, 3), scale

    def residual(x):
        fits, intrinsics, points, _ = unpack(x)
        projection, depth = _project(points[point_ids], frames, fits, intrinsics)
        obs = np.column_stack((projection - xy, np.minimum(depth - 0.005, 0.0) * 100.0))
        return np.r_[obs.ravel(), 3.0 * x[:camera_size] / priors]

    x0 = np.r_[np.zeros(camera_size), initial_points.ravel()]
    row_count = 3 * len(frames)
    pattern = lil_matrix((row_count + camera_size, len(x0)), dtype=bool)
    for view, start in zip(rot_views, range(0, rot_size, 3)):
        for row in np.flatnonzero(frames == view):
            pattern[3 * row:3 * row + 3, start:start + 3] = True
    for view, start in zip(center_views, range(rot_size, pose_size, 3)):
        for row in np.flatnonzero(frames == view):
            pattern[3 * row:3 * row + 3, start:start + 3] = True
    if not fixed_focal:
        pattern[:row_count, pose_size] = True
    for row, point_id in enumerate(point_ids):
        start = camera_size + 3 * point_id
        pattern[3 * row:3 * row + 3, start:start + 3] = True
    for i in range(camera_size):
        pattern[row_count + i, i] = True

    result = least_squares(residual, x0, jac_sparsity=pattern.tocsr(), x_scale="jac",
                           loss="soft_l1", f_scale=1.5, max_nfev=80,
                           ftol=1e-5, xtol=1e-5, gtol=1e-5,
                           bounds=(np.r_[lower, np.full(initial_points.size, -np.inf)],
                                   np.r_[upper, np.full(initial_points.size, np.inf)]))
    if not np.isfinite(result.x).all():
        raise ValueError("Bundle adjustment returned nonfinite parameters")
    fitted, intrinsics, points, scale = unpack(result.x)
    before, depth_before = _project(initial_points[point_ids], frames, poses, K)
    after, depth_after = _project(points[point_ids], frames, fitted, intrinsics)
    report = {"observations_per_camera": coverage.tolist(), "focal_multiplier": scale,
              "active_camera_parameter_bounds": int(np.count_nonzero(result.active_mask[:camera_size])),
              "optimizer_status": int(result.status), "optimizer_message": str(result.message),
              "function_evaluations": int(result.nfev), "final_robust_cost": float(result.cost),
              "original": _errors(before, depth_before, xy), "refined": _errors(after, depth_after, xy)}
    return fitted, intrinsics, report


def _errors(projection, depth, target):
    errors = np.linalg.norm(projection - target, axis=1)
    invalid = (depth < 0.005) | ~np.isfinite(errors)
    errors = np.where(invalid, VALIDATION_ERROR_CAP, np.minimum(errors, VALIDATION_ERROR_CAP))
    return {"observations": len(errors), "invalid": int(invalid.sum()),
            "median_px": float(np.median(errors)), "p90_px": float(np.quantile(errors, 0.9)),
            "rmse_px": float(np.sqrt(np.mean(errors ** 2)))}


def _reserved_error(poses, K, validation):
    projections = _projection_matrices(poses, K)
    predicted, depth, observed = [], [], []
    for track, anchors in validation:
        frames, xy = track
        query = np.ones(len(frames), dtype=bool)
        query[list(anchors)] = False
        point = _triangulate(track, anchors, projections)
        if point is None:
            prediction, z = np.full_like(xy, np.nan), np.zeros(len(xy))
        else:
            prediction, z = _project(point[None], frames, poses, K)
            if np.any(z[list(anchors)] < 0.005):
                z[:] = 0.0
        predicted.extend(prediction[query])
        depth.extend(z[query])
        observed.extend(xy[query])
    return _errors(np.asarray(predicted), np.asarray(depth), np.asarray(observed)) if observed else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--initializer", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fixed-focal", action="store_true", help="Refine poses with the original intrinsics")
    args = parser.parse_args()
    dataset = args.dataset.resolve()
    initializer = args.initializer / "scene.npz" if args.initializer.is_dir() else args.initializer
    outputs = [args.output / name for name in ("scene.npz", "scene.json", "refinement.json")]
    if any(path.exists() for path in outputs):
        raise FileExistsError(f"Use a fresh calibration output directory: {args.output}")

    start = time.perf_counter()
    manifest = load_manifest(dataset)
    scene = load_scene(initializer)
    train_frames = {frame["image"]: frame for frame in manifest["frames"] if frame["split"] == "train"}
    names = scene["names"].tolist()
    if set(names) != set(train_frames) or len(names) < 3:
        raise ValueError("Expected a scene with exactly the manifest's sparse training cameras (at least three)")
    cv2.setNumThreads(1)
    cv2.setRNGSeed(SEED)
    features = _training_features(dataset, [train_frames[name] for name in names])
    tracks, pairs = _build_tracks(features)
    poses, K = scene["c2w"].astype(np.float64), scene["K"].astype(np.float64)
    origin = poses[0, :3, 3].copy()
    radii = np.linalg.norm(poses[:, :3, 3] - origin, axis=1)
    anchor = int(np.argmax(radii))
    radius = radii[anchor]
    if radius < 1e-7:
        raise ValueError("Initial cameras have insufficient translation for triangulation")
    poses[:, :3, 3] = (poses[:, :3, 3] - origin) / radius
    train, points, validation = _split_tracks(tracks, poses, K)
    if len(train) < 30:
        raise ValueError(f"Too few triangulated training tracks: {len(train)}")
    fitted, intrinsics, optimization = _bundle_adjust(poses, K, train, points, anchor, args.fixed_focal)
    reserved = {"tracks": len(validation), "original": _reserved_error(poses, K, validation),
                "refined": _reserved_error(fitted, intrinsics, validation),
                "protocol": "Track-disjoint from BA; re-triangulate the same two anchor observations and score others"}
    changes = Rotation.from_matrix(poses[:, :3, :3].transpose(0, 2, 1) @ fitted[:, :3, :3]).magnitude()
    normalized_center_change = np.linalg.norm(poses[:, :3, 3] - fitted[:, :3, 3], axis=1)
    fitted[:, :3, 3] = fitted[:, :3, 3] * radius + origin

    args.output.mkdir(parents=True, exist_ok=True)
    temporary = outputs[0].with_suffix(".tmp.npz")
    np.savez_compressed(temporary, **{**scene, "c2w": fitted, "K": intrinsics})
    load_scene(temporary)
    temporary.replace(outputs[0])
    source = {"scene": manifest["scene"], "split_hash": manifest["split_hash"],
              "method": "train_rgb_track_ba", "reference_data_used": False,
              "parent_initializer_sha256": sha256_file(initializer),
              "scene_sha256": sha256_file(outputs[0]), "seed_points_preserved": len(scene["points"]),
              "source": source_version(), "script_sha256": sha256_file(__file__)}
    write_json(outputs[1], source)
    write_json(outputs[2], {**source, "elapsed_seconds": time.perf_counter() - start,
                            "training_ids": names, "accepted_pair_matches": pairs,
                            "fitted_tracks": len(train), "seed": SEED,
                            "gauge_anchor_ids": [names[0], names[anchor]],
                            "error_cap_px": VALIDATION_ERROR_CAP,
                            "camera_rotation_change_degrees": np.rad2deg(changes).tolist(),
                            "center_change_over_anchor_baseline": normalized_center_change.tolist(),
                            "bundle_adjustment": optimization, "reserved_tracks": reserved})
    print(f"Fitted {len(train)} training tracks and scored {len(validation)} reserved tracks in {outputs[2]}")


if __name__ == "__main__":
    main()

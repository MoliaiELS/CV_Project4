"""Seed sparse 3DGS with verified train tracks and consistent VGGT depth."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sqlite3

import numpy as np
from scipy.ndimage import map_coordinates

from p4.common import sha256_file, source_version, write_json
from p4.data import load_scene
from p4.initialize import depth_to_points
from scripts.init_da3_sparse import _load_train_rgb


NATIVE_UNITS = "native VGGT camera-axis Z; not normalized or fit"
SEED = 42
POINT_CAP = 100_000
COLMAP_PAIR_BASE = 2_147_483_647


def read_tracks(path, names):
    """Group verified matches, dropping components with competing keypoints in one image."""
    parent = {}

    def root(key):
        while parent[key] != key:
            parent[key] = parent[parent[key]]
            key = parent[key]
        return key

    uri = path.resolve().as_uri() + "?mode=ro"
    with sqlite3.connect(uri, uri=True) as db:
        images = dict(db.execute("SELECT image_id, name FROM images"))
        if sorted(images.values()) != sorted(names):
            raise ValueError("SIFT database must contain exactly the frozen training images")
        indices = {name: i for i, name in enumerate(names)}
        frame_by_id = {image_id: indices[name] for image_id, name in images.items()}
        keypoints = {}
        for image_id, count, columns, data in db.execute("SELECT image_id, rows, cols, data FROM keypoints"):
            keypoints[frame_by_id[image_id]] = np.frombuffer(data or b"", dtype="<f4").reshape(count, columns)[:, :2]
        pairs = db.execute("SELECT pair_id, rows, data FROM two_view_geometries "
                           "WHERE rows >= 15 AND config IN (2, 3, 4, 8) ORDER BY pair_id").fetchall()
        for pair_id, count, data in pairs:
            left, right = [frame_by_id[image_id] for image_id in divmod(pair_id, COLMAP_PAIR_BASE)]
            for a, b in np.frombuffer(data, dtype="<u4").reshape(count, 2):
                va, vb = (left, int(a)), (right, int(b))
                parent.setdefault(va, va)
                parent.setdefault(vb, vb)
                parent[root(vb)] = root(va)
    components = {}
    for node in parent:
        components.setdefault(root(node), []).append(node)
    tracks = []
    for component in components.values():
        views = np.array([view for view, _ in component])
        if len(set(views)) == len(views):
            # COLMAP keypoint coordinates start at (.5,.5); DPT arrays start at (0,0).
            tracks.append((views, np.array([keypoints[i][j] for i, j in component]) - 0.5))
    return tracks, {"verified_pairs": len(pairs), "unambiguous_tracks": len(tracks),
                    "ambiguous_tracks_discarded": len(components) - len(tracks)}


def triangulate_track(views, xy, w2c, K, inverse_K):
    """Fit a track with fixed cameras and check parallax, cheirality and reprojection."""
    pixels = np.column_stack([xy, np.ones(len(xy))])
    xy_normalized = np.einsum("nij,nj->ni", inverse_K[views], pixels)[:, :2]
    projection = w2c[views, :3]
    rows = (xy_normalized[:, :, None] * projection[:, 2:3] - projection[:, :2]).reshape(-1, 4)
    point = np.linalg.svd(rows, full_matrices=False)[2][-1]
    if not np.isfinite(point).all() or abs(point[3]) < 1e-10:
        return None
    point = point[:3] / point[3]
    camera = np.einsum("nij,j->ni", projection[:, :, :3], point) + projection[:, :, 3]
    z = camera[:, 2]
    if (z <= 0).any():
        return None
    rays = np.einsum("nji,nj->ni", projection[:, :, :3], camera)
    rays /= np.linalg.norm(rays, axis=1)[:, None]
    if (rays @ rays.T).min() >= np.cos(np.deg2rad(1.5)):
        return None
    image = np.einsum("nij,nj->ni", K[views], camera)
    residual = np.linalg.norm(image[:, :2] / z[:, None] - xy, axis=1)
    if residual.max() > 2.0:
        return None
    return point, z, residual


def sample(array, xy, order=1):
    return map_coordinates(array, xy.T[::-1], order=order, mode="constant", cval=np.nan)


def collect_anchors(tracks, scene, rgb):
    K, w2c = scene["K"], np.linalg.inv(scene["c2w"])
    inverse_K = np.linalg.inv(K)
    anchors = [[] for _ in K]
    points, colors, errors = [], [], []
    for views, xy in tracks:
        h, w = rgb.shape[1:3]
        if ((xy < 0) | (xy > [w - 1, h - 1])).any():
            continue
        fit = triangulate_track(views, xy, w2c, K, inverse_K)
        if fit is None:
            continue
        point, z, residual = fit
        points.append(point)
        errors.extend(residual.tolist())
        x, y = np.rint(xy).astype(int).T
        colors.append(np.median(rgb[views, y, x], axis=0).astype(np.uint8))
        for index, view in enumerate(views):
            anchors[view].append([*xy[index], z[index]])
    return (np.asarray(points, dtype=np.float32).reshape(-1, 3),
            np.asarray(colors, dtype=np.uint8).reshape(-1, 3), anchors,
            {"accepted_tracks": len(points), "track_observations": len(errors),
             "median_track_reprojection_px": float(np.median(errors)) if errors else None})


def fit_log_scale(observations, depth, confidence, view_index, native_gauge):
    """Fit scale on training tracks and gate the correction on disjoint training tracks."""
    result = {"scale": 1., "usable_anchors": 0, "occupied_cells": 0, "status": "insufficient_anchors"}
    if not len(observations):
        return result
    observations = np.asarray(observations)
    xy, reference_z = observations[:, :2], observations[:, 2]
    source_z, source_conf = sample(depth, xy), sample(confidence, xy)
    keep = (source_z > 0) & (source_conf > 0) & (reference_z > 0)
    xy, source_z, reference_z = xy[keep], source_z[keep], reference_z[keep]
    height, width = depth.shape
    cells = np.minimum((xy * [4 / width, 4 / height]).astype(int), 3)
    result.update(usable_anchors=len(xy), occupied_cells=len(np.unique(cells, axis=0)))
    if len(xy) < 16 or result["occupied_cells"] < 4:
        return result
    ratio = np.log(reference_z / source_z)
    ids = np.random.default_rng(SEED + view_index).permutation(len(ratio))
    reserved_count = max(4, len(ids) // 4)
    validation, fitting = ratio[ids[:reserved_count]], ratio[ids[reserved_count:]]
    offset = float(np.median(fitting))
    error_before, error_after = np.abs(validation), np.abs(validation - offset)
    before, after = float(np.median(error_before)), float(np.median(error_after))
    p90_after = float(np.quantile(error_after, .90))
    result.update(fit_anchors=len(fitting), reserved_anchors=len(validation),
                  candidate_scale=float(np.exp(offset)), reserved_median_log_error_native=before,
                  reserved_median_log_error_scaled=after, reserved_p90_log_error_scaled=p90_after)
    if after > .15 or p90_after > .32:
        result["status"] = "inconsistent_reserved_anchors"
    elif native_gauge and (abs(offset) > np.log(2) or before - after < max(.01, .1 * before)):
        result["status"] = "native_scale_retained"
    else:
        result.update(scale=result["candidate_scale"], status="calibrated")
    return result


def consistency_mask(depth, confidence, rgb, c2w, K, pixel_offset):
    """Accept a confident depth sample when one nearby training view corroborates its color and Z."""
    n, h, w = depth.shape
    intrinsics_inverse, w2c = np.linalg.inv(K), np.linalg.inv(c2w)
    centers = c2w[:, :3, 3]
    neighbors = np.argsort(np.linalg.norm(centers[:, None] - centers, axis=2), axis=1, kind="stable")
    masks = np.zeros(depth.shape, dtype=bool)
    rgb = rgb.astype(np.float32) / 255.
    for i in range(n):
        positive = confidence[i] > 0
        if not positive.any():
            continue
        strong = np.quantile(confidence[i][positive], .5)
        y, x = np.nonzero(positive & (depth[i] > 0) & (confidence[i] >= strong))
        pixels = np.column_stack([x + pixel_offset, y + pixel_offset, np.ones(len(y))])
        camera = (pixels @ intrinsics_inverse[i].T) * depth[i, y, x, None]
        world = camera @ c2w[i, :3, :3].T + centers[i]
        agrees = np.zeros(len(x), dtype=bool)
        for j in [j for j in neighbors[i] if j != i][:4]:
            camera_j = world @ w2c[j, :3, :3].T + w2c[j, :3, 3]
            projection = camera_j @ K[j].T
            z = camera_j[:, 2]
            ids = np.flatnonzero((z > 0) & ~agrees)
            xy = projection[ids, :2] / z[ids, None] - pixel_offset
            depth_j = sample(depth[j], xy)
            conf_j = sample(confidence[j], xy)
            valid = (depth_j > 0) & (conf_j > 0)
            depth_match = np.abs(np.log(depth_j[valid] / z[ids][valid])) <= .12
            ids, xy = ids[valid][depth_match], xy[valid][depth_match]
            rgb_j = np.column_stack([sample(rgb[j, :, :, k], xy) for k in range(3)])
            color_match = np.mean(np.abs(rgb_j - rgb[i, y[ids], x[ids]]), axis=1) <= .12
            agrees[ids[color_match]] = True
        masks[i, y[agrees], x[agrees]] = True
    return masks


def load_native_maps(path, manifest, manifest_sha, names, size):
    receipt_path = path.with_name("depth_receipt.json")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    digest = sha256_file(path)
    if (receipt.get("depth_maps_sha256") != digest or receipt.get("training_ids") != names
            or receipt.get("manifest_sha256") != manifest_sha or receipt.get("split_hash") != manifest["split_hash"]
            or receipt.get("reference_data_used") is not False or receipt.get("depth_units") != NATIVE_UNITS
            or receipt.get("source_initializer_sha256") != receipt.get("camera_initializer_sha256")):
        raise ValueError("Input depth receipt must authenticate native train-only DPT data in the prepared grid")
    with np.load(path, allow_pickle=False) as maps:
        if (maps["names"].tolist() != names or maps["split_hash"].item() != manifest["split_hash"]
                or maps["camera_initializer_sha256"].item() != receipt["camera_initializer_sha256"]):
            raise ValueError("DPT array metadata disagrees with the authenticated source receipt")
        depth, confidence = maps["depth"], maps["confidence"]
    if (depth.shape != (len(names), *size[::-1]) or confidence.shape != depth.shape
            or not np.isfinite(depth).all() or not np.isfinite(confidence).all()
            or np.any(depth <= 0) or np.any(confidence < 0)):
        raise ValueError("Expected aligned DPT depth and confidence arrays in the prepared training image grid")
    return depth, confidence, digest, receipt["camera_initializer_sha256"]


def initialize(dataset, camera_initializer, matches_database, depth_maps, output):
    dataset = dataset.resolve()
    manifest_sha = sha256_file(dataset / "manifest.json")
    manifest, names, images, size = _load_train_rgb(dataset)
    rgb = np.asarray(images)
    scene = load_scene(camera_initializer)
    source_sha = sha256_file(camera_initializer)
    if scene["names"].tolist() != names:
        raise ValueError("Initializer must have exactly the ordered prepared training views")
    depth, confidence, depth_sha, depth_camera_sha = load_native_maps(depth_maps, manifest, manifest_sha, names, size)
    native_gauge = source_sha == depth_camera_sha
    if not native_gauge:
        camera_receipt = json.loads(camera_initializer.with_name("scene.json").read_text(encoding="utf-8"))
        if (camera_receipt.get("method") != "pycolmap" or camera_receipt.get("status") != "completed"
                or camera_receipt.get("scene_sha256") != source_sha
                or camera_receipt.get("manifest_sha256") != manifest_sha
                or camera_receipt.get("registered_frames") != len(names)
                or camera_receipt.get("train_only_inputs") is not True):
            raise ValueError("Non-native camera gauge requires a verified fully registered training-only PyCOLMAP scene")
    pixel_offset = 0. if native_gauge else .5
    array_K = scene["K"].copy()
    array_K[:, :2, 2] -= pixel_offset
    db_sha = sha256_file(matches_database)
    tracks, track_info = read_tracks(matches_database, names)
    track_scene = {**scene, "K": array_K}
    points, colors, anchors, triangulation = collect_anchors(tracks, track_scene, rgb)
    scales = [fit_log_scale(anchors[i], depth[i], confidence[i], i, native_gauge) for i in range(len(names))]
    output.mkdir(parents=True)
    receipt = {**source_version(), "schema_version": 1, "scene": manifest["scene"], "method": "track_depth_seed",
               "script_sha256": sha256_file(__file__), "split_hash": manifest["split_hash"],
               "manifest_sha256": manifest_sha, "training_ids": names, "train_only_inputs": True,
               "reference_data_used": False, "source_camera_initializer_sha256": source_sha,
               "native_dpt_camera_initializer_sha256": depth_camera_sha, "native_dpt_sha256": depth_sha,
               "sift_database_sha256": db_sha, "prepared_hw": list(size[::-1]), "seed": SEED, "point_cap": POINT_CAP,
               "camera_policy": "fixed native VGGT" if native_gauge else "fixed fully registered PyCOLMAP",
               "track_selection": {**track_info, **triangulation}, "scale_by_training_image": dict(zip(names, scales)),
               "support_policy": {"source_confidence_quantile": .5, "neighbor_count": 4,
                                  "max_abs_log_depth_residual": .12, "max_mean_abs_rgb_residual": .12},
               "status": "running"}
    write_json(output / "scene.json", receipt)
    try:
        if not native_gauge and any(fit["status"] != "calibrated" for fit in scales):
            raise ValueError("Some train views lack independently validated depth scales in the PyCOLMAP gauge")
        depth *= np.asarray([fit["scale"] for fit in scales], dtype=np.float32)[:, None, None]
        supported = consistency_mask(depth, confidence, rgb, scene["c2w"], scene["K"], pixel_offset)
        per_view = supported.reshape(len(names), -1).sum(axis=1)
        receipt["supported_pixels_by_training_image"] = dict(zip(names, per_view.tolist()))
        if supported.sum() < 128:
            raise ValueError("Insufficient DPT pixels confirmed by training-view color and depth")
        if not native_gauge:
            points, colors = scene["points"], scene["colors"]
        dense, dense_colors, _ = depth_to_points(np.where(supported, depth, 0.), confidence,
                                                  np.linalg.inv(scene["c2w"]), array_K, rgb,
                                                  [(0, 0, *size)] * len(names), confidence_quantile=0.,
                                                  max_points=POINT_CAP - len(points), seed=SEED)
        if sha256_file(dataset / "manifest.json") != manifest_sha or sha256_file(matches_database) != db_sha:
            raise ValueError("Prepared dataset or SIFT matches changed during seed construction")
        scene_path = output / "scene.npz"
        temporary = output / "_scene.npz"
        np.savez_compressed(temporary, names=scene["names"], K=scene["K"], c2w=scene["c2w"],
                            points=np.concatenate([points, dense]), colors=np.concatenate([colors, dense_colors]))
        load_scene(temporary)
        temporary.replace(scene_path)
        scene_sha = sha256_file(scene_path)
        receipt.update(scene_sha256=scene_sha, registered_frames=len(names), initial_points=len(points) + len(dense),
                       seed_sparse_points=len(points), seed_dense_points=len(dense), status="completed")
        if per_view.min() > 0:
            artifact = output / "depth_maps.npz"
            np.savez_compressed(artifact, names=scene["names"], depth=np.where(supported, depth, 0.),
                                confidence=np.where(supported, confidence, 0.),
                                split_hash=np.asarray(manifest["split_hash"]),
                                camera_initializer_sha256=np.asarray(scene_sha))
            write_json(output / "depth_receipt.json",
                       {**receipt, "camera_initializer_sha256": scene_sha, "depth_maps_sha256": sha256_file(artifact),
                        "depth_units": "camera-axis Z in output initializer gauge",
                        "invalid_depth": "zero depth and confidence on unsupported pixels"})
            receipt["exports_warp_depth"] = True
        else:
            receipt["exports_warp_depth"] = False
        write_json(output / "scene.json", receipt)
    except Exception as exc:
        receipt.update(status="failed", error=str(exc))
        write_json(output / "scene.json", receipt)
        raise
    return scene_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("dataset", "camera-initializer", "matches-database", "depth-maps", "output"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    args = parser.parse_args()
    print(initialize(args.dataset, args.camera_initializer, args.matches_database, args.depth_maps, args.output))


if __name__ == "__main__":
    main()

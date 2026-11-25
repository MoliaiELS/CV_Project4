"""Audit fixed-camera Part 2 results and train-track geometry on paired views."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from p4.common import sha256_file, source_version, write_json
from p4.data import load_manifest, load_scene
from p4.geometry import align_training_cameras, invert_poses, transform_poses


SCENE_COUNTS = {"Waymo-405841": (174, 20, 25), "Dl3dv-2": (267, 11, 39), "Re10k-1": (244, 10, 35)}
CONDITIONS = ("full", "sparse")
METRICS = ("psnr", "ssim", "lpips")
ALPHA_METRICS = ("mean_alpha", "coverage_alpha_0_1", "coverage_alpha_0_5", "coverage_alpha_0_9")
STEP = 7000


def _read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _frame_map(manifest, split):
    return {item["image"]: item for item in manifest["frames"] if item["split"] == split}


def _rows(camera_table, names, field):
    index = {name: i for i, name in enumerate(camera_table["names"].tolist())}
    return camera_table[field][[index[name] for name in names]]


def _angles(reference, estimate):
    relative = reference.swapaxes(-1, -2) @ estimate
    return np.rad2deg(Rotation.from_matrix(relative).magnitude())


def _distribution(values):
    values = np.asarray(values, dtype=np.float64)
    if not len(values):
        return {"count": 0, "mean": None, "median": None, "rmse": None, "p90": None}
    return {"count": len(values), "mean": float(values.mean()), "median": float(np.median(values)),
            "rmse": float(np.sqrt(np.mean(values ** 2))), "p90": float(np.quantile(values, 0.90))}


def pose_diagnostics(estimated, reference, names):
    """Compare poses on a preselected, temporally ordered set of training names."""
    poses = _rows(estimated, names, "c2w")
    target = _rows(reference, names, "c2w")
    fit = align_training_cameras(estimated["names"], estimated["c2w"],
                                 reference["names"], reference["c2w"], names)
    aligned = transform_poses(poses, fit["estimate_to_reference"])
    position_error = np.linalg.norm(aligned[:, :3, 3] - target[:, :3, 3], axis=1)
    rotation = _angles(target[:, :3, :3], aligned[:, :3, :3])
    target_relative = invert_poses(target[:-1]) @ target[1:]
    aligned_relative = invert_poses(aligned[:-1]) @ aligned[1:]
    relative_translation = np.linalg.norm(aligned_relative[:, :3, 3] - target_relative[:, :3, 3], axis=1)
    relative_rotation = _angles(target_relative[:, :3, :3], aligned_relative[:, :3, :3])
    K, reference_K = (_rows(table, names, "K") for table in (estimated, reference))
    focal = K[:, [0, 1], [0, 1]]
    reference_focal = reference_K[:, [0, 1], [0, 1]]
    _require(np.all(np.isfinite(focal) & (focal > 0)) and np.all(np.isfinite(reference_focal) & (reference_focal > 0)),
             "Estimated and reference focal lengths must be finite and positive")
    ratio = focal / reference_focal
    estimate_to_reference = fit["estimate_to_reference"]
    return {
        "paired_ate_rmse": fit["ate_rmse"],
        "rotation_error_degrees": _distribution(rotation),
        "focal_median_ratio_fx_fy": np.median(ratio, axis=0).tolist(),
        "focal_mean_abs_error_percent_fx_fy": (100 * np.abs(ratio - 1).mean(axis=0)).tolist(),
        "focal_mean_abs_error_pixels_fx_fy": np.abs(focal - reference_focal).mean(axis=0).tolist(),
        "relative_translation": _distribution(relative_translation),
        "relative_rotation_degrees": _distribution(relative_rotation),
        "estimate_to_reference": {key: np.asarray(value).tolist() for key, value in estimate_to_reference.items()},
        "frames": [{"image": name, "translation_error": float(t), "rotation_error_degrees": float(r),
                    "focal_ratio_fx_fy": q.tolist()} for name, t, r, q in zip(names, position_error, rotation, ratio)],
        "relative_pairs": [{"from": a, "to": b, "translation_error": float(t), "rotation_error_degrees": float(r)}
                           for a, b, t, r in zip(names[:-1], names[1:], relative_translation, relative_rotation)],
    }


def reserved_reprojection_errors(tracks, estimated, names):
    """Use only estimated cameras, with temporally first/last observations as anchors."""
    from scripts.seed_sparse_colmap_depth import triangulate_track

    K = _rows(estimated, names, "K")
    w2c = invert_poses(_rows(estimated, names, "c2w"))
    inverse_K = np.linalg.inv(K)
    errors, anchors_failed, projections_invalid = {}, 0, 0
    for index, (frames, xy) in enumerate(tracks):
        if len(frames) < 3:
            continue
        anchor = np.array([0, len(frames) - 1])
        fit = triangulate_track(frames[anchor], xy[anchor], w2c, K, inverse_K)
        if fit is None:
            anchors_failed += 1
            continue
        point = fit[0]
        views = frames[1:-1]
        camera = w2c[views, :3, :3] @ point + w2c[views, :3, 3]
        z = camera[:, 2]
        valid = (z > 0) & np.all(np.isfinite(camera), axis=1)
        image = (K[views[valid]] @ camera[valid, :, None])[..., 0]
        residual = np.linalg.norm(image[:, :2] / z[valid, None] - xy[1:-1][valid], axis=1)
        projections_invalid += len(views) - int(valid.sum())
        errors.update({(index, int(view)): float(e) for view, e in zip(views[valid], residual) if np.isfinite(e)})
    return errors, {"tracks_failing_anchor_checks": anchors_failed, "nonpositive_or_nonfinite_predictions": projections_invalid}


def _reserved_train_tracks(dataset, train_frames, cameras, common):
    import cv2
    from scripts.refine_sparse_cameras import _build_tracks, _training_features

    cv2.setNumThreads(2)
    cv2.setRNGSeed(42)
    tracks, pairs = _build_tracks(_training_features(dataset, [train_frames[name] for name in common]))
    tracks = [(frames, xy) for frames, xy in tracks if len(frames) >= 3]
    measured = {key: reserved_reprojection_errors(tracks, cameras[key], common) for key in CONDITIONS}
    shared = sorted(measured["full"][0].keys() & measured["sparse"][0].keys())
    return {
        "matching": "OpenCV SIFT, mutual 0.78 ratio, MAGSAC fundamental matrix on common training RGB",
        "opencv": cv2.__version__,
        "policy": "Triangulate first/last match per track with frozen estimated cameras and reserve every interior observation",
        "anchor_requirements": {"minimum_parallax_degrees": 1.5, "maximum_reprojection_px": 2.0,
                                "positive_camera_z": True},
        "uses_reference_cameras": False,
        "verified_image_pairs": len(pairs),
        "tracks_at_least_three_views": len(tracks),
        "attempted_reserved_observations": sum(len(frames) - 2 for frames, _ in tracks),
        "paired_valid_reserved_observations": len(shared),
        "conditions": {key: {**status, "all_valid_error_px": _distribution(list(errors.values())),
                              "paired_valid_error_px": _distribution([errors[k] for k in shared])}
                       for key, (errors, status) in measured.items()},
    }


def _read_reference(path):
    with np.load(path, allow_pickle=False) as ref:
        result = {key: ref[key] for key in ("names", "c2w", "K")}
    _require(len(result["names"]) == len(set(result["names"].tolist())), f"Duplicate reference names in {path}")
    return result


def _load_case(scene, condition, dataset, manifest, reference, case, archived, cover, alpha):
    train = _frame_map(manifest, "train")
    test = _frame_map(manifest, "test")
    scene_path = case / "init/scene.npz"
    camera = load_scene(scene_path)
    case_names = camera["names"].tolist()
    init = _read_json(case / "init/scene.json")
    render = _read_json(case / "renders/render.json")
    provenance = _read_json(case / "train/provenance.json")
    metrics_path = case / "metrics.json"
    metrics = _read_json(metrics_path)
    checkpoint = case / "train/checkpoints/step_007000.pt"
    manifest_hash, init_hash = sha256_file(dataset / "manifest.json"), sha256_file(scene_path)
    step = _read_json(case / "train/config.json")["steps"]
    label = f"{scene}/{condition}"
    _require(init["status"] == "completed" and init["method"] == "vggt" and init["input_names"] == list(train)
             and init["registered_frames"] == len(case_names) and set(case_names) <= train.keys(),
             f"Initializer provenance mismatch for {label}")
    _require(init["manifest_sha256"] == provenance["manifest_sha256"] == manifest_hash
             and provenance["scene_sha256"] == init_hash and provenance["pose_optimization"] is False,
             f"Fixed-camera training provenance mismatch for {label}")
    digest = sha256_file(checkpoint)
    _require(digest == archived["checkpoint_sha256"] == render["checkpoint_sha256"] == cover["checkpoint_sha256"]
             and init["weights_sha256"] == archived["weights_sha256"] and step == render["step"] == STEP,
             f"Frozen checkpoint mismatch for {label}")
    _require(metrics["scene"] == scene and metrics["split_hash"] == manifest["split_hash"] == archived["split_hash"]
             and metrics["complete"] and metrics["test_frames"] == render["count"] == len(test)
             and metrics["protocol"]["test_rgb_in_reconstruction"] is False
             and metrics["protocol"]["postprocessing"] == "none" and render["raw_gaussian_renders"] is True
             and render["test_pose_refinement"] is False,
             f"Held-out rendering protocol mismatch for {label}")
    alignment = align_training_cameras(camera["names"], camera["c2w"], reference["names"], reference["c2w"], train)
    _require(render["train_names"] == alignment["train_names"] and np.isclose(render["ate_rmse"], alignment["ate_rmse"]),
             f"Archived held-out camera alignment mismatch for {label}")
    _require([(r["id"], r["image"]) for r in metrics["per_frame"]] == [(r["id"], r["image"]) for r in test.values()]
             and list(alpha) == list(test) and int(cover["test_frames"]) == len(test),
             f"Frame order or count mismatch in rendering and coverage for {label}")
    aggregate = {key: float(np.mean([r[key] for r in metrics["per_frame"]])) for key in METRICS}
    for key, value in aggregate.items():
        _require(np.isclose(value, metrics[key], rtol=0, atol=1e-10)
                 and np.isclose(value, archived[key], rtol=0, atol=1e-10), f"Archived {key} mismatch for {label}")
    coverage = {key: float(np.mean([r[key] for r in alpha.values()])) for key in ALPHA_METRICS}
    for key, value in coverage.items():
        _require(0 <= value <= 1 and np.isclose(value, float(cover[key]), rtol=0, atol=1e-10),
                 f"Archived alpha summary mismatch for {label}")
    _require(int(cover["replay_max_pixel_difference"]) == 0
             and all(r["replay_max_pixel_difference"] == 0 for r in alpha.values()),
             f"Alpha replay did not match archived raw render pixels for {label}")
    return camera, metrics, {
        "attempted_training_frames": len(train), "registered_training_frames": len(case_names),
        "registered_with_reference_frames": len(alignment["train_names"]), "test_frames": len(test),
        "render_alignment_all_train_ate_rmse": alignment["ate_rmse"],
        "rendering": aggregate, "alpha_support": coverage,
        "split_hash": manifest["split_hash"], "manifest_sha256": manifest_hash,
        "initializer_sha256": init_hash, "checkpoint_sha256": digest, "render_metrics_sha256": sha256_file(metrics_path),
    }


def _scene_diagnostics(assets, scene, cover, alpha):
    formal = assets / "runs/part2_formal" / scene / "full_sparse_7000_v1"
    datasets = {"full": assets / "data/prepared" / scene, "sparse": assets / "data/part2_sparse" / scene}
    manifests = {key: load_manifest(dataset) for key, dataset in datasets.items()}
    train = {key: _frame_map(manifest, "train") for key, manifest in manifests.items()}
    test = {key: _frame_map(manifest, "test") for key, manifest in manifests.items()}
    counts = (len(train["full"]), len(train["sparse"]), len(test["sparse"]))
    _require(counts == SCENE_COUNTS[scene] and test["full"] == test["sparse"]
             and all(frame == train["full"].get(name) for name, frame in train["sparse"].items()),
             f"Official full/sparse split mismatch for {scene}")
    _require(all(set(train[key]).isdisjoint(test[key]) for key in CONDITIONS), f"Overlapping split for {scene}")
    reference_paths = {key: dataset / "reference.npz" for key, dataset in datasets.items()}
    references = {key: _read_reference(path) for key, path in reference_paths.items()}
    ref_common = list(train["sparse"]) + list(test["sparse"])
    _require(all(np.array_equal(_rows(references["full"], ref_common, field),
                                _rows(references["sparse"], ref_common, field)) for field in ("c2w", "K")),
             f"Shared reference camera mismatch in {scene}")
    status = _read_json(formal / "status.json")
    archive = {row["condition"]: row for row in _read_json(formal / "comparison.json")}
    stages = {stage["name"]: stage["returncode"] for stage in status["stages"]}
    stage_names = {f"{key}_{stage}" for key in CONDITIONS for stage in ("initialize", "train", "render", "evaluate")}
    frozen_hash = sha256_file(formal / "frozen_config.json")
    status_config_hash = status.get("frozen_config_sha256")
    # The original Re10k receipt predates this field; the same config bytes
    # are authenticated by the other two scenes and checked across the matrix.
    _require(status_config_hash == frozen_hash or (scene == "Re10k-1" and status_config_hash is None),
             f"Frozen configuration hash mismatch for {scene}")
    _require(status["status"] == "completed" and status["scene"] == scene and status["steps"] == STEP
             and status["seed"] == 42 and status["test_frames"] == counts[2]
             and status["training_frames"] == dict(zip(CONDITIONS, counts[:2]))
             and stages == dict.fromkeys(stage_names, 0) and len(status["stages"]) == len(stage_names),
             f"Frozen stage receipt mismatch for {scene}")
    cameras, metrics, conditions = {}, {}, {}
    frozen = _read_json(formal / "frozen_config.json")
    for key in CONDITIONS:
        case = formal / key
        _require(_read_json(case / "train/config.json") == frozen.get("training", frozen),
                 f"Frozen train settings mismatch for {scene}/{key}")
        cameras[key], metrics[key], conditions[key] = _load_case(
            scene, key, datasets[key], manifests[key], references[key], case, archive[key], cover[key], alpha[key])
    registered = set(cameras["full"]["names"].tolist()) & set(cameras["sparse"]["names"].tolist())
    common = [name for name in train["sparse"] if name in registered]
    for key in CONDITIONS:
        pose = pose_diagnostics(cameras[key], references["full"], common)
        _require(len(common) == archive[key]["paired_ate_frame_count"]
                 and np.isclose(pose["paired_ate_rmse"], archive[key]["paired_ate_rmse"], rtol=0, atol=1e-10),
                 f"Paired trajectory mismatch for {scene}/{key}")
        conditions[key].update(pose, common_training_frames=len(common))
    pairs = list(zip(metrics["full"]["per_frame"], metrics["sparse"]["per_frame"]))
    return {
        "scene": scene, "steps": STEP, "matching_order": "sparse manifest order",
        "reference_sha256": {key: sha256_file(path) for key, path in reference_paths.items()},
        "frozen_config_sha256": frozen_hash, "status_config_sha256": status_config_hash,
        "common_training_images": common, "ordered_test_ids": [row["id"] for row in test["sparse"].values()],
        "conditions": conditions,
        "heldout_sparse_minus_full": {
            **{key: float(np.mean([r[key] - l[key] for l, r in pairs])) for key in METRICS},
            **{key: conditions["sparse"]["alpha_support"][key] - conditions["full"]["alpha_support"][key]
               for key in ALPHA_METRICS},
        },
        "reference_free_reserved_training_reprojection": _reserved_train_tracks(datasets["sparse"], train["sparse"],
                                                                                 cameras, common),
    }


def audit(assets):
    supplement = assets / "results/part2_supplement_v1"
    coverage_path, alpha_path = supplement / "coverage.csv", supplement / "coverage_per_frame.json"
    alpha_records = _read_json(alpha_path)
    with coverage_path.open(newline="", encoding="utf-8") as stream:
        coverage = {(row["scene"], row["condition"]): row for row in csv.DictReader(stream)}
    cases = []
    for scene in SCENE_COUNTS:
        alpha = {key: {r["image"]: r for r in alpha_records if (r["scene"], r["condition"]) == (scene, key)}
                 for key in CONDITIONS}
        cases.append(_scene_diagnostics(assets, scene, {key: coverage[scene, key] for key in CONDITIONS}, alpha))
    _require(len({case["frozen_config_sha256"] for case in cases}) == 1,
             "Part 2 baselines did not use the same frozen config")
    return {
        "schema_version": 1, "status": "completed", "source": source_version(),
        "diagnostic_script_sha256": sha256_file(__file__),
        "protocol": {
            "alignment": "Independent estimate-to-reference Sim(3) fit to the same sparse training IDs per condition",
            "ate": "Camera-center RMSE in scene-specific reference trajectory units",
            "rpe": "Consecutive common sequence frames after Sim(3). Compare inverse(T_i) @ T_j. Translation in reference units and rotation geodesic in degrees.",
            "focal": "Estimated-to-reference fx and fy ratios and absolute errors on the prepared image grid",
            "coverage": "Archived same-checkpoint accumulated alpha support on the frozen held-out pixels",
            "reference_usage": "Only evaluate geometric accuracy and verify reference consistency. Track reprojection uses estimated cameras and train RGB.",
            "coverage_csv_sha256": sha256_file(coverage_path), "coverage_frames_sha256": sha256_file(alpha_path),
        },
        "scenes": cases,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets-root", type=Path, required=True)
    parser.add_argument("--output", default="-", help="JSON path, or '-' for stdout")
    args = parser.parse_args()
    result = audit(args.assets_root.resolve())
    if args.output == "-":
        print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))
    else:
        write_json(Path(args.output), result)


if __name__ == "__main__":
    main()

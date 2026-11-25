import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from p4.geometry import transform_poses
from scripts.part2_pose_diagnostics import pose_diagnostics, reserved_reprojection_errors


def _table(names, centers, focal=100.):
    poses = np.repeat(np.eye(4)[None], len(names), axis=0)
    poses[:, :3, 3] = centers
    K = np.repeat(np.diag([focal, focal, 1.])[None], len(names), axis=0)
    return {"names": np.asarray(names), "c2w": poses, "K": K}


def test_pose_diagnostics_uses_temporal_common_order_and_scene_gauge():
    order = ["010.png", "030.png", "020.png"]
    reference = _table(order, [[0., 0., 0.], [1., 0., 0.], [1., 1., 0.]])
    gauge = {"scale": 2., "rotation": Rotation.from_euler("z", 20, degrees=True).as_matrix(),
             "translation": [3., -2., .5]}
    estimate = {key: values[::-1].copy() for key, values in reference.items()}
    estimate["c2w"] = transform_poses(estimate["c2w"], gauge)
    estimate["K"][:, :2, :2] *= 1.2
    assert pose_diagnostics(estimate, reference, order)["relative_rotation_degrees"]["rmse"] < 1e-7
    angular_change = Rotation.from_euler("z", 10, degrees=True).as_matrix()
    estimate["c2w"][1, :3, :3] = estimate["c2w"][1, :3, :3] @ angular_change
    result = pose_diagnostics(estimate, reference, order)
    assert result["paired_ate_rmse"] < 1e-10
    assert result["rotation_error_degrees"]["mean"] == pytest.approx(10. / 3.)
    assert result["relative_rotation_degrees"]["rmse"] == pytest.approx(10.)
    assert result["focal_median_ratio_fx_fy"] == pytest.approx([1.2, 1.2])
    assert result["focal_mean_abs_error_percent_fx_fy"] == pytest.approx([20., 20.])
    assert [(row["from"], row["to"]) for row in result["relative_pairs"]] == list(zip(order, order[1:]))
    assert result["relative_pairs"][0]["translation_error"] == pytest.approx(0., abs=1e-10)
    assert result["relative_pairs"][1]["translation_error"] == pytest.approx(2 * np.sin(np.deg2rad(5.)))


def test_reprojection_reserves_the_middle_observation():
    names = ["first", "reserve", "last"]
    camera = _table(names, [[0., 0., 0.], [1., 0., 0.], [2., 0., 0.]], focal=20.)
    frames = np.array([0, 1, 2])
    points = np.array([[10., 5.], [5., 5.], [0., 5.]])
    errors, diagnostics = reserved_reprojection_errors([(frames, points)], camera, names)
    assert errors == pytest.approx({(0, 1): 0.}, abs=1e-10)
    assert not diagnostics["tracks_failing_anchor_checks"]
    points[1, 1] += 3.
    errors, _ = reserved_reprojection_errors([(frames, points)], camera, names)
    assert errors == pytest.approx({(0, 1): 3.})
    camera["c2w"][2, 0, 3] = 0.
    errors, diagnostics = reserved_reprojection_errors([(frames, points)], camera, names)
    assert not errors and diagnostics["tracks_failing_anchor_checks"] == 1

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

pytest.importorskip("cv2")
from scripts.refine_sparse_cameras import _bundle_adjust, _project, _reserved_error, _split_tracks


def test_fixed_focal_adjustment_preserves_gauge_and_improves_disjoint_tracks():
    rng = np.random.default_rng(14)
    poses = np.repeat(np.eye(4)[None], 4, axis=0)
    poses[:, :3, 3] = [[0., 0., 0.], [0.3, 0.1, 0.], [0.65, -0.06, 0.08], [1., 0., 0.]]
    intrinsics = np.repeat(np.array([[[360., 0., 180.], [0., 350., 120.], [0., 0., 1.]]]), 4, axis=0)
    points = rng.uniform([-1., -0.7, 3.], [1., 0.7, 7.], (60, 3))
    views = np.arange(len(poses))
    tracks = []
    for point in points:
        pixels, _ = _project(point[None], views, poses, intrinsics)
        tracks.append((views, pixels))

    initial = poses.copy()
    initial[1:3, :3, 3] += [[0.012, -0.016, 0.004], [0.009, 0.010, 0.]]
    initial[1:, :3, :3] = Rotation.from_rotvec([[0.006, -0.012, 0.008], [0.004, 0.01, 0.], [0., -0.008, 0.003]]).as_matrix()
    fit_tracks, fit_points, reserved = _split_tracks(tracks, initial, intrinsics)
    assert len(fit_tracks) == 48 and len(reserved) == 12
    assert {id(track) for track in fit_tracks}.isdisjoint(id(track) for track, _ in reserved)

    fitted, fitted_K, diagnostics = _bundle_adjust(initial, intrinsics, fit_tracks, fit_points, 3, True)
    before = _reserved_error(initial, intrinsics, reserved)
    after = _reserved_error(fitted, fitted_K, reserved)

    np.testing.assert_array_equal(fitted[0], initial[0])
    np.testing.assert_array_equal(fitted[3, :3, 3], initial[3, :3, 3])
    np.testing.assert_array_equal(fitted_K, intrinsics)
    assert diagnostics["focal_multiplier"] == 1.
    assert before["observations"] == after["observations"] == 24
    assert after["invalid"] == 0
    assert after["median_px"] < 0.3 * before["median_px"]

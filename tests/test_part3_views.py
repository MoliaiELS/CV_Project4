import numpy as np
from scipy.spatial.transform import Rotation

from scripts.part3_views import Camera, Candidate, confidence_map, interpolate_views, select_views


def camera(name, translation=0., rotation=0.):
    pose = np.eye(4)
    pose[0, 3] = translation
    pose[:3, :3] = Rotation.from_euler("y", rotation, degrees=True).as_matrix()
    K = np.array([[9., 0., 4.5], [0., 9., 4.5], [0., 0., 1.]])
    return Camera(name, pose, K, 9, 9)


def test_interpolates_only_between_sparse_anchors():
    left, right = camera("left", 0., 0.), camera("right", 3., 90.)
    candidates = interpolate_views([left, right])
    assert len(candidates) == 3
    midpoint = candidates[1].camera
    np.testing.assert_allclose(midpoint.c2w[0, 3], 1.5)
    np.testing.assert_allclose(midpoint.c2w[:3, :3], Rotation.from_euler("y", 45, degrees=True).as_matrix())
    assert candidates[1].reference is left
    assert candidates[2].reference is right


def test_active_planner_does_not_repeat_interval():
    candidates = interpolate_views([camera(str(index), index) for index in range(3)])

    def render(target):
        opacity = np.ones((9, 9), dtype=np.float32)
        if target.name.startswith("interval_000"):
            opacity[:6] = 0.
        return {"alpha": opacity}

    chosen, scores = select_views(candidates, render, 2)
    assert {target.interval for target in chosen} == {0, 1}
    assert len(scores) == 6
    assert max(row["score"] for row in scores if "000" in row["target"]) > 0.


def test_gate_uses_geometry_before_colour():
    target, reference = camera("synthetic"), camera("real")
    candidate = Candidate(target, (reference, reference), 0, 0.5)
    alpha = np.ones((9, 9), dtype=np.float32)
    depth = np.full_like(alpha, 2.)
    rgb = np.full((9, 9, 3), 0.5, dtype=np.float32)
    render = {"alpha": alpha, "depth": depth, "rgb": rgb}
    verified, stats = confidence_map(candidate, render, rgb, [(reference, render, rgb)], 1.)
    np.testing.assert_allclose(verified, 1.)
    assert stats["verified_fraction"] == 1.
    inconsistent, _ = confidence_map(candidate, render, np.ones_like(rgb), [(reference, render, rgb)], 1.)
    assert inconsistent.max() < .01
    occluded = {**render, "depth": depth / 2.}
    unsupported, stats = confidence_map(candidate, render, rgb, [(reference, occluded, rgb)], 1.)
    np.testing.assert_allclose(unsupported, .20)
    assert stats["verified_fraction"] == 0.

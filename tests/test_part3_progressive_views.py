import numpy as np
import pytest

from p4.common import sha256_file
from scripts.part3_progressive import _complete_warp, _load_warp_depth
from scripts.part3_progressive_views import fuse_restorations, progressive_candidates, select_progressive_views
from scripts.part3_views import Camera, Candidate


def camera(name, offset=0.0):
    pose = np.eye(4)
    pose[0, 3] = offset
    intrinsics = np.array([[4., 0., 4.], [0., 4., 4.], [0., 0., 1.]])
    return Camera(name, pose, intrinsics, 8, 8)


def render(camera, value=0.5, alpha=1.0, depth=1.0):
    shape = (camera.height, camera.width)
    return {"rgb": np.full((*shape, 3), value, dtype=np.float32),
            "alpha": np.full(shape, alpha, dtype=np.float32),
            "depth": np.full(shape, depth, dtype=np.float32)}


def fuse(raw_value, real_values, restoration_values, *, right_depth=1., target_alpha=1.):
    target = camera("virtual")
    left, right = camera("left"), camera("right")
    candidate = Candidate(target, (left, right), 0, 0.2)
    raw = render(target, raw_value, alpha=target_alpha)
    restores = [render(target, value)["rgb"] for value in restoration_values]
    anchor_inputs = [(left, render(left), render(left, real_values[0])["rgb"]),
                     (right, render(right, depth=right_depth), render(right, real_values[1])["rgb"])]
    return fuse_restorations(candidate, raw, restores, anchor_inputs, 1.)


def test_progression_uses_near_anchors_then_interior_and_distinct_ids():
    cameras = [camera("left"), camera("middle", 1.), camera("right", 3.)]
    first = progressive_candidates(cameras, 1)
    second = progressive_candidates(cameras, 2)
    assert [view.fraction for view in first] == [0.2, 0.8, 0.2, 0.8]
    assert [view.fraction for view in second] == [1 / 3, 0.5, 2 / 3] * 2
    assert [view.camera.c2w[0, 3] for view in first] == pytest.approx([0.2, 0.8, 1.4, 2.6])
    assert len({view.camera.name for view in first + second}) == len(first + second)


def test_ranking_favors_supported_holes_and_caches_real_renders():
    anchors = [camera(f"real_{index}") for index in range(4)]
    candidates = progressive_candidates(anchors, 1)
    alphas = {"interval_000_20": 0.25, "interval_000_80": 0.30,
              "interval_001_20": 0.35, "interval_001_80": 0.90,
              "interval_002_20": 0.01, "interval_002_80": 0.95}
    calls = []

    def render_selected(camera):
        calls.append(camera.name)
        return render(camera, alpha=alphas.get(camera.name, 1.0))

    selected, scores = select_progressive_views(candidates, render_selected, 2, 1.)
    assert [view.camera.name for view in selected] == ["interval_000_20", "interval_001_20"]
    assert len(calls) == len(set(calls)) == len(anchors) + len(candidates)
    assert scores[4]["hole_fraction"] == 1.
    assert scores[4]["score"] == 0.


def test_bilateral_agreement_accepts_independent_residuals():
    enhanced, confidence, stats = fuse(0.50, (0.32, 0.33), (0.34, 0.35))
    np.testing.assert_allclose(enhanced[3, 3], 0.345, atol=0.002)
    assert confidence[3, 3] > 0.8
    assert stats["bilateral_edit_fraction"] > 0.8
    assert stats["single_anchor_edit_fraction"] == 0.


@pytest.mark.parametrize("reference,proposals", [((0.5, 0.5), (0.2, 0.2)), ((0.32, 0.68), (0.32, 0.68))])
def test_fusion_vetoes_unnecessary_or_conflicting_edits(reference, proposals):
    enhanced, confidence, _ = fuse(0.5, reference, proposals)
    np.testing.assert_allclose(enhanced, 0.5)
    np.testing.assert_array_equal(confidence, 0.)


def test_single_anchor_has_limited_confidence_and_unsupported_pixels_stay_raw():
    enhanced, confidence, stats = fuse(0.5, (0.3, 0.3), (0.3, 0.3), right_depth=np.nan)
    np.testing.assert_allclose(enhanced[3, 3], 0.3)
    assert confidence[3, 3] == pytest.approx(0.4)
    assert stats["bilateral_edit_fraction"] == 0.
    enhanced, confidence, stats = fuse(0.5, (0.3, 0.3), (0.3, 0.3), target_alpha=np.nan)
    np.testing.assert_allclose(enhanced, 0.5)
    np.testing.assert_array_equal(confidence, 0.)
    assert stats["verified_fraction"] == 0.


def test_warp_completion_preserves_real_colors_and_rejects_disputed_or_remote_pixels():
    raw = np.full((64, 64, 3), 0.5, dtype=np.float32)
    rgb = np.full_like(raw, 0.8)
    confidence = np.zeros(raw.shape[:2], dtype=np.float32)
    confidence[30, 30] = 0.7
    support = np.zeros_like(confidence, dtype=np.uint8)
    support[30, 30] = 2
    support[30, 31] = 2  # A disputed projection is not an empty hole.
    left = np.full_like(raw, 0.2)
    right = np.full_like(raw, 0.22)
    right[31, 30] = 0.9

    result, weights, stats = _complete_warp(raw, {"rgb": rgb, "confidence": confidence,
                                                  "support_count": support, "stats": {}}, (left, right))
    np.testing.assert_array_equal(result[30, 30], rgb[30, 30])
    assert weights[30, 30] == pytest.approx(0.7)
    np.testing.assert_allclose(result[29, 30], 0.21)
    assert weights[29, 30] == pytest.approx(0.15)
    for row, col in [(30, 31), (31, 30), (35, 30)]:
        np.testing.assert_array_equal(result[row, col], raw[row, col])
        assert weights[row, col] == 0.
    assert stats["real_warp_fraction"] == pytest.approx(1 / 64 ** 2)


def test_independent_depth_binds_to_split_initializer_and_camera_dimensions(tmp_path, monkeypatch):
    target = camera("real")
    initializer = tmp_path / "scene.npz"
    initializer.write_bytes(b"sparse training cameras")
    monkeypatch.setattr("scripts.part3_progressive.load_manifest", lambda _: {"split_hash": "sparse-train"})
    archive = tmp_path / "depth_maps.npz"
    depth = np.ones((1, 8, 8), dtype=np.float32)
    depth[0, 0, 0] = np.nan
    confidence = np.linspace(1., 2., 64, dtype=np.float32).reshape(depth.shape)
    np.savez(archive, names=np.array(["real"]), depth=depth, confidence=confidence,
             split_hash="sparse-train", camera_initializer_sha256=sha256_file(initializer))
    z, alpha = _load_warp_depth(archive, [target], tmp_path, initializer)["real"]
    assert z[0, 0] == alpha[0, 0] == 0.
    assert z[-1, -1] == alpha[-1, -1] == 1.
    with pytest.raises(ValueError, match="dimensions"):
        _load_warp_depth(archive, [Camera(target.name, target.c2w, target.K, 7, 8)], tmp_path, initializer)
    initializer.write_bytes(b"other cameras")
    with pytest.raises(ValueError, match="camera initializer"):
        _load_warp_depth(archive, [target], tmp_path, initializer)

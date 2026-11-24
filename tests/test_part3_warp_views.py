import numpy as np
import pytest

from scripts.part3_views import Camera, Candidate
from scripts.part3_warp_views import forward_splat_rgb, warp_real_prior


def camera(name, translation=0.0, width=16, height=8, focal=8.):
    pose = np.eye(4)
    pose[0, 3] = translation
    K = np.array([[focal, 0., width / 2], [0., focal, height / 2], [0., 0., 1.]])
    return Camera(name, pose, K, width, height)


def geometry(view, depth=1., alpha=1.):
    shape = (view.height, view.width)
    return {"depth": np.full(shape, depth, np.float32), "alpha": np.full(shape, alpha, np.float32)}


def solid(view, color):
    return np.broadcast_to(np.asarray(color, dtype=np.float32), (view.height, view.width, 3))


def test_identity_forward_splat_keeps_camera_z_and_color():
    source, target = camera("source"), camera("target")
    color = np.random.default_rng(7).random((source.height, source.width, 3), dtype=np.float32)
    result = forward_splat_rgb(source, target, geometry(source, depth=1.75), color, 1.)
    np.testing.assert_allclose(result["rgb"], color, atol=1e-6)
    np.testing.assert_allclose(result["depth"], 1.75)
    np.testing.assert_array_equal(result["support"], True)


def test_subpixel_splat_uses_calibrated_translation_and_normalization():
    source, target = camera("source", translation=.125), camera("target")
    stripes = np.broadcast_to(np.linspace(0, 1, source.width, dtype=np.float32)[None, :, None],
                              (source.height, source.width, 3))
    result = forward_splat_rgb(source, target, geometry(source), stripes, normalization_radius=2.)
    expected = (stripes[:, :-1] + stripes[:, 1:]) / 2
    np.testing.assert_allclose(result["rgb"][:, 1:], expected, atol=1e-6)
    np.testing.assert_allclose(result["confidence"][:, 0], .5, atol=1e-6)


def test_z_buffer_keeps_nearest_plane_and_removes_depth_edges():
    source, target = camera("source", translation=.75), camera("target")
    source_geometry = geometry(source)
    source_geometry["depth"][:, 8:] = 2.
    colors = solid(source, [1., 0., 0.]).copy()
    colors[:, 8:] = [0., 0., 1.]
    result = forward_splat_rgb(source, target, source_geometry, colors, 1.)
    assert result["stats"]["edge_rejected_source_fraction"] == pytest.approx(2 / 16)
    np.testing.assert_array_equal(result["rgb"][4, 12], [1., 0., 0.])
    np.testing.assert_array_equal(result["rgb"][4, 13], [0., 0., 1.])
    np.testing.assert_allclose(result["depth"][4, 12:14], [1., 2.])


def test_bilateral_prior_combines_close_donors_and_vetoes_conflicts():
    left, right, target = [camera(name) for name in ("left", "right", "target")]
    candidate = Candidate(target, (left, right), 0, .5)
    right_rgb = solid(right, [.14, .20, .31])
    left_input = left, geometry(left), solid(left, [.10, .20, .30])
    result = warp_real_prior(candidate, [left_input, (right, geometry(right), right_rgb)], 1.)
    np.testing.assert_allclose(result["rgb"][4, 12], [.12, .20, .305], atol=1e-6)
    np.testing.assert_array_equal(result["support_count"], 2)
    assert result["stats"]["bilateral_agreement_fraction"] == 1.
    assert result["confidence"].min() > .9
    color_conflict = warp_real_prior(candidate, [left_input, (right, geometry(right), solid(right, [1., 1., 1.]))], 1.)
    np.testing.assert_array_equal(color_conflict["confidence"], 0.)
    assert color_conflict["stats"]["conflicting_overlap_fraction"] == 1.
    depth_conflict = warp_real_prior(candidate, [left_input, (right, geometry(right, depth=2.), right_rgb)], 1.)
    np.testing.assert_array_equal(depth_conflict["confidence"], 0.)


def test_missing_geometry_is_explicit_and_one_donor_receives_lower_weight():
    left, right, target = [camera(name) for name in ("left", "right", "target")]
    candidate = Candidate(target, (left, right), 0, .5)
    left_input = left, geometry(left), solid(left, [.10, .20, .30])
    right_input = right, geometry(right, depth=np.nan), solid(right, [.7, .8, .9])
    result = warp_real_prior(candidate, [left_input, right_input], 1.)
    np.testing.assert_array_equal(result["support_count"], 1)
    np.testing.assert_allclose(result["rgb"], left_input[2])
    np.testing.assert_allclose(result["confidence"], .4)
    unknown_left = left, geometry(left, alpha=.1), solid(left, [.1, .2, .3])
    missing = warp_real_prior(candidate, [unknown_left, right_input], 1.)
    np.testing.assert_array_equal(missing["support_count"], 0)
    np.testing.assert_array_equal(missing["confidence"], 0.)
    assert missing["stats"]["prior_coverage"] == 0.

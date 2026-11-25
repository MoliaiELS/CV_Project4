import sqlite3

import numpy as np
import pytest

from scripts.seed_sparse_colmap_depth import (COLMAP_PAIR_BASE, consistency_mask, fit_log_scale,
                                              read_tracks, triangulate_track)


def test_verified_track_pixels_use_the_prepared_array_grid(tmp_path):
    path = tmp_path / "matches.db"
    with sqlite3.connect(path) as db:
        db.executescript("CREATE TABLE images(image_id, name); "
                         "CREATE TABLE keypoints(image_id, rows, cols, data); "
                         "CREATE TABLE two_view_geometries(pair_id, rows, data, config);")
        for image_id, name in ((1, "a"), (2, "b")):
            db.execute("INSERT INTO images VALUES (?,?)", (image_id, name))
            xy = np.column_stack([np.arange(15) + image_id + .5, np.full(15, 8.5)]).astype("<f4")
            db.execute("INSERT INTO keypoints VALUES (?,15,2,?)", (image_id, xy.tobytes()))
        pairs = np.repeat(np.arange(15, dtype="<u4")[:, None], 2, axis=1)
        db.execute("INSERT INTO two_view_geometries VALUES (?,15,?,2)", (COLMAP_PAIR_BASE + 2, pairs.tobytes()))
    tracks, summary = read_tracks(path, ["a", "b"])
    assert summary["verified_pairs"] == 1 and len(tracks) == 15
    np.testing.assert_array_equal(tracks[0][0], [0, 1])
    np.testing.assert_array_equal(tracks[0][1], [[1, 8], [2, 8]])
    with pytest.raises(ValueError, match="exactly"):
        read_tracks(path, ["a", "absent"])


def test_fixed_camera_triangulation_uses_cheirality_parallax_and_reprojection():
    w2c = np.repeat(np.eye(4)[None], 3, axis=0)
    w2c[1, 0, 3], w2c[2, 1, 3] = -1., -1.
    K = np.repeat(np.diag([20., 20., 1.])[None], 3, axis=0)
    views = np.arange(3)
    xy = np.array([[5., 5.], [0., 5.], [5., 0.]])
    point, z, error = triangulate_track(views, xy, w2c, K, np.linalg.inv(K))
    np.testing.assert_allclose(point, [1, 1, 4])
    np.testing.assert_allclose(z, 4)
    np.testing.assert_allclose(error, 0, atol=1e-8)
    xy[2] += 40
    assert triangulate_track(views, xy, w2c, K, np.linalg.inv(K)) is None


def test_depth_scale_and_train_color_gates():
    depth = np.full((3, 32, 32), 2., dtype=np.float32)
    confidence = np.ones_like(depth)
    observations = [[x, y, 2.8] for x in (2, 10, 18, 25) for y in (2, 10, 18, 25)]
    fit = fit_log_scale(observations, depth[0], confidence[0], 0, native_gauge=True)
    assert fit["status"] == "calibrated" and fit["scale"] == pytest.approx(1.4)
    assert fit["fit_anchors"] == 12 and fit["reserved_median_log_error_scaled"] == pytest.approx(0.)
    assert fit_log_scale(observations[:3], depth[0], confidence[0], 0, native_gauge=True)["scale"] == 1.
    c2w = np.repeat(np.eye(4)[None], 3, axis=0)
    c2w[1, 0, 3], c2w[2, 1, 3] = .2, .2
    K = np.repeat(np.diag([20., 20., 1.])[None], 3, axis=0)
    rgb = np.full((3, 32, 32, 3), 64, np.uint8)
    rgb[2] = 255
    corroborated = consistency_mask(depth, confidence, rgb, c2w, K, pixel_offset=0.)
    assert corroborated[0, 16, 16] and corroborated[1, 16, 16]
    assert not corroborated[2].any()
    depth[1] *= 2.
    assert not consistency_mask(depth, confidence, rgb, c2w, K, pixel_offset=0.).any()

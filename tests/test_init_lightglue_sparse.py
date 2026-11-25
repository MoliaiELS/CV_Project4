from types import SimpleNamespace

import numpy as np
import pytest

from scripts.init_lightglue_sparse import MIN_INLIERS, _populate_database, colmap_keypoints, mutual_pairs


def test_keypoints_keep_native_ids_and_map_pixel_centers_to_colmap():
    detected = np.array([[4, 5], [100, 202], [508, 501]]) / 512
    points = colmap_keypoints(detected, (512, 512), (256, 256))
    np.testing.assert_array_equal(points, [[2.25, 2.75], [50.25, 101.25], [254.25, 250.75]])
    assert points.dtype == np.float32 and points.flags.c_contiguous
    with pytest.raises(ValueError, match="outside"):
        colmap_keypoints([[1, 0.5]], (512, 512), (256, 256))


def test_mutual_matches_exclude_padding_and_keep_detector_indices():
    pairs = mutual_pairs([2, 1, -1, 0, 1], [3, 0, 0, -1, 3], 4, 3)
    np.testing.assert_array_equal(pairs, [[0, 2], [3, 0]])
    assert pairs.dtype == np.uint32
    assert mutual_pairs([-1, -1], [-1, -1], 2, 2).shape == (0, 2)


def test_pair_geometry_is_estimated_and_only_verified_edges_reach_mapper(tmp_path):
    names = ["train_a.png", "train_b.png", "train_c.png"]
    paths = tmp_path / "database.db", tmp_path / "train_rgb"
    points = [np.zeros((20, 2), dtype=np.float32) for name in names]
    matches = np.column_stack([np.arange(20), np.arange(20)]).astype(np.uint32)
    calls = []
    images = [SimpleNamespace(name=name, image_id=i, camera_id=1) for i, name in enumerate(names, 1)]
    database = SimpleNamespace(
        open=lambda path: calls.append(("open", path)), close=lambda: calls.append(("close",)),
        read_all_images=lambda: images[::-1], read_camera=lambda camera: camera,
        write_keypoints=lambda i, xy: calls.append(("points", i, len(xy))),
        write_matches=lambda i, j, rows: calls.append(("matches", i, j, len(rows))),
        write_two_view_geometry=lambda i, j, geom: calls.append(("geometry", i, j, len(geom.inlier_matches))),
    )
    counts = iter((MIN_INLIERS, MIN_INLIERS - 1))

    def verify(camera_a, xy_a, camera_b, xy_b, rows, options):
        assert camera_a == camera_b == 1 and len(xy_a) == len(xy_b) == len(rows) == 20
        assert options.ransac.max_error == 2 and options.min_num_inliers == MIN_INLIERS
        return SimpleNamespace(config=2, inlier_matches=rows[:next(counts)])

    pycolmap = SimpleNamespace(
        Database=lambda: database, CameraMode=SimpleNamespace(SINGLE=1),
        ImageReaderOptions=SimpleNamespace,
        import_images=lambda database_path, root, mode, train_names, options: calls.append(
            ("import", database_path, root, mode, train_names, options.camera_model)),
        TwoViewGeometryOptions=lambda min_num_inliers: SimpleNamespace(
            min_num_inliers=min_num_inliers, ransac=SimpleNamespace()),
        estimate_two_view_geometry=verify,
    )
    ledger = _populate_database(pycolmap, *paths, names, range(3), points,
                                lambda i, j: matches if i == 0 else matches[:0])
    assert calls[0] == ("import", str(paths[0]), str(paths[1]), 1, names, "PINHOLE")
    assert calls[2:5] == [("points", i, 20) for i in range(1, 4)]
    assert [call for call in calls if call[0] == "geometry"] == [("geometry", 1, 2, MIN_INLIERS)]
    assert ledger == [{"names": names[:2], "matches": 20, "inliers": MIN_INLIERS, "geometry_config": 2},
                      {"names": [names[0], names[2]], "matches": 20, "inliers": MIN_INLIERS - 1,
                       "geometry_config": 2},
                      {"names": names[1:], "matches": 0, "inliers": 0, "geometry_config": None}]
    assert calls[-1] == ("close",)

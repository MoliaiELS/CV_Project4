import json
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest

from p4.common import sha256_file
from p4.data import load_manifest
from scripts.init_pycolmap_sparse import map_training_views, reconstruction_to_scene, stage_train_images


def training_model():
    centers = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]])
    images = {}
    for index, center in enumerate(centers):
        matrix = np.column_stack([np.eye(3), -center])
        images[index] = SimpleNamespace(name=f"view_{index}.png", camera_id=1, num_points3D=16,
                                        cam_from_world=SimpleNamespace(matrix=lambda matrix=matrix: matrix))
    camera = SimpleNamespace(model=SimpleNamespace(name="PINHOLE"), width=8, height=9,
                             calibration_matrix=lambda: np.array([[4., 0., 4.], [0., 4., 4.5], [0., 0., 1.]]))
    points = {index: SimpleNamespace(xyz=[0, index, 1], color=[17, 22, 48]) for index in range(16)}
    return SimpleNamespace(reg_image_ids=lambda: list(images), image=images.__getitem__, camera=lambda _: camera,
                           points3D=points, num_reg_images=lambda: len(images),
                           compute_mean_reprojection_error=lambda: 0.5, compute_mean_track_length=lambda: 3.0)


def test_camera_export_preserves_manifest_order_and_inverts_opencv_w2c():
    model = training_model()
    names = ["view_2.png", "view_0.png", "view_1.png"]
    scene, receipt = reconstruction_to_scene(model, names, (8, 9))
    assert scene["names"].tolist() == names
    np.testing.assert_allclose(scene["c2w"][:, :3, 3], [[0, 1, 0], [0, 0, 0], [1, 0, 0]])
    np.testing.assert_allclose(scene["K"][0], [[4., 0., 4.], [0., 4., 4.5], [0., 0., 1.]])
    np.testing.assert_array_equal(scene["colors"][0], [17, 22, 48])
    assert scene["colors"].dtype == np.uint8
    assert receipt["observed_points_per_train_image"] == dict.fromkeys(names, 16)
    with pytest.raises(ValueError, match="exactly"):
        reconstruction_to_scene(model, [*names[:2], "absent.png"], (8, 9))


def test_staging_copies_exact_train_bytes_and_validates_hash(tmp_path):
    dataset, staged = tmp_path / "dataset", tmp_path / "train"
    dataset.mkdir()
    frames = []
    for index in range(3):
        path = dataset / f"view_{index}.png"
        Image.new("RGB", (8, 9), (index, 0, 0)).save(path)
        frames.append({"image": path.name, "split": "train", "width": 8, "height": 9, "sha256": sha256_file(path)})
    frames.append({"image": "absent-test.png", "split": "test"})
    (dataset / "manifest.json").write_text(json.dumps({"schema_version": 1, "frames": frames}), encoding="utf8")
    train = [frame for frame in load_manifest(dataset)["frames"] if frame["split"] == "train"]
    assert stage_train_images(dataset, staged, train) == (8, 9)
    assert sorted(path.name for path in staged.iterdir()) == sorted(frame["image"] for frame in train)
    assert all(sha256_file(staged / frame["image"]) == frame["sha256"] for frame in train)
    Image.new("RGB", (8, 9), (255, 0, 0)).save(dataset / train[0]["image"])
    with pytest.raises(ValueError, match="hash differs"):
        stage_train_images(dataset, tmp_path / "changed", train)


def test_registration_falls_back_to_verified_train_pair(tmp_path):
    model = training_model()
    partial = SimpleNamespace(**vars(model))
    partial.reg_image_ids = lambda: [0, 1]
    partial.num_reg_images = lambda: 2

    def options(**kwargs):
        return SimpleNamespace(**kwargs, init_image_id1=-1, init_image_id2=-1,
                               mapper=SimpleNamespace(init_min_tri_angle=16., init_min_num_inliers=100),
                               triangulation=SimpleNamespace(ignore_two_view_tracks=True))

    def mapping(*_, options):
        assert options.num_threads == 2 and not options.triangulation.ignore_two_view_tracks
        return {0: model if (options.init_image_id1, options.init_image_id2) == (2, 3) else partial}

    api = SimpleNamespace(IncrementalPipelineOptions=options, set_random_seed=lambda _: None,
                          incremental_mapping=mapping)
    names = ["view_0.png", "view_1.png", "view_2.png"]
    anchor = {"image_ids": [2, 3], "names": names[1:], "inliers": 75}
    receipt = {"mapping_attempts": []}
    _, diagnostics = map_training_views(api, tmp_path / "db", tmp_path / "train", tmp_path,
                                        names, (8, 9), [anchor], receipt)
    assert [attempt["registered_frames"] for attempt in receipt["mapping_attempts"]] == [2, 2, 3]
    assert receipt["mapping_attempts"][0]["missing_train_images"] == [names[2]]
    assert receipt["mapping_attempts"][-1]["anchor"] == anchor
    assert diagnostics["mean_reprojection_error_px"] == 0.5

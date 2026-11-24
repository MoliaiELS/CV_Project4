import json
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest

from p4.common import sha256_file
from p4.data import load_scene
from scripts.init_da3_sparse import _load_train_rgb, prediction_to_scene


def prediction():
    centers = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]])
    K = np.array([[4., 0., 2.], [0., 6., 1.], [0., 0., 1.]])
    w2c = np.broadcast_to(np.eye(4), (3, 4, 4)).copy()
    w2c[:, :3, 3] = -centers
    pixels = np.broadcast_to(np.array([22, 33, 44], np.uint8), (3, 3, 4, 3)).copy()
    return SimpleNamespace(depth=np.full((3, 3, 4), 2.), conf=np.ones((3, 3, 4)), processed_images=pixels,
                           extrinsics=w2c[:, :3], intrinsics=np.broadcast_to(K, (3, 3, 3)).copy())


def test_da3_export_inverts_w2c_and_maps_anisotropic_resize(tmp_path):
    source = prediction()
    names = ["frame_b.png", "frame_a.png", "frame_c.png"]
    scene, metadata = prediction_to_scene(source, names, (8, 9))
    assert scene["names"].tolist() == names
    np.testing.assert_allclose(scene["c2w"][:, :3, 3], [[0, 0, 0], [1, 0, 0], [0, 1, 0]])
    np.testing.assert_allclose(scene["K"][0], [[8, 0, 4], [0, 18, 3], [0, 0, 1]])
    np.testing.assert_allclose(scene["points"][0], [-1, -1 / 3, 2])
    np.testing.assert_array_equal(scene["colors"][0], [22, 33, 44])
    assert metadata["processed_image_wh"] == [4, 3]
    np.savez_compressed(tmp_path / "scene.npz", **scene)
    assert load_scene(tmp_path)["names"].tolist() == names


def test_da3_export_rejects_invalid_w2c_and_unobservable_camera_path():
    names = ["a", "b", "c"]
    source = prediction()
    source.extrinsics[0, 0, 0] = -1
    with pytest.raises(ValueError, match="nonrigid"):
        prediction_to_scene(source, names, (8, 9))
    source = prediction()
    source.extrinsics[2, :3, 3] = [-2, 0, 0]
    with pytest.raises(ValueError, match="global Sim"):
        prediction_to_scene(source, names, (8, 9))


def test_da3_reads_only_sha_verified_train_rgb(tmp_path):
    rows = []
    for frame_id, color in enumerate((71, 72, 73)):
        name = f"{frame_id}.png"
        path = tmp_path / name
        Image.new("RGB", (8, 9), (color, 0, 0)).save(path)
        rows.append({"id": frame_id, "image": name, "split": "train", "width": 8, "height": 9,
                     "sha256": sha256_file(path)})
    rows.insert(1, {"id": 7, "image": "absent-test.png", "split": "test"})
    (tmp_path / "manifest.json").write_text(json.dumps({"schema_version": 1, "frames": rows,
                                                      "camera_convention": "opencv_c2w"}), encoding="utf-8")
    _, names, images, size = _load_train_rgb(tmp_path)
    assert names == ["0.png", "1.png", "2.png"]
    assert size == (8, 9)
    assert [int(image[0, 0, 0]) for image in images] == [71, 72, 73]
    Image.new("RGB", (8, 9)).save(tmp_path / "0.png")
    with pytest.raises(ValueError, match="hash differs"):
        _load_train_rgb(tmp_path)

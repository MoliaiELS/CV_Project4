import json
from pathlib import Path
import shutil

import numpy as np
from PIL import Image
import pytest

from p4.data import prepare_dataset
from p4.evaluate import evaluate_renders, evaluation_cameras, rendering_metrics


def toy_dataset(tmp_path):
    source = tmp_path / "source"
    (source / "images").mkdir(parents=True)
    centers = [[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, .3]]
    frames = []
    for i, center in enumerate(centers):
        pixels = np.full((24, 32, 3), i * 40, dtype=np.uint8)
        Image.fromarray(pixels).save(source / "images" / f"{i}.png")
        pose = np.eye(4)
        pose[:3, 3] = center
        frames.append({"file_path": f"images/{i}.png", "transform_matrix": pose.tolist()})
    (source / "transforms.json").write_text(json.dumps({"w": 32, "h": 24, "fl_x": 20, "fl_y": 20,
                                                        "camera_convention": "opencv", "frames": frames}))
    split = tmp_path / "split.json"
    split.write_text(json.dumps({"train": ["0.png", "1.png", "2.png"], "test": ["3.png"]}))
    dataset = tmp_path / "prepared"
    manifest = prepare_dataset(source, dataset, "toy", split_file=split)
    return dataset, manifest


def test_held_out_cameras_train_only_transform_and_leakage_rejected(tmp_path):
    dataset, manifest = toy_dataset(tmp_path)
    with np.load(dataset / "reference.npz") as ref:
        poses = ref["c2w"][:3].copy()
        poses[:, :3, 3] = poses[:, :3, 3] * 2 + [1, 2, 3]
        scene = {"names": ref["names"][:3], "c2w": poses, "K": ref["K"][:3],
                 "points": np.array([[0., 0, 3]]), "colors": np.array([[255, 0, 0]], dtype=np.uint8)}
    scene_path = tmp_path / "scene.npz"
    np.savez(scene_path, **scene)
    cameras = evaluation_cameras(dataset, scene_path)
    assert cameras["names"].tolist() == ["images/3.png.png"]
    np.testing.assert_allclose(cameras["c2w"][0, :3, 3], [3, 4, 3.6])
    assert cameras["alignment"]["ate_rmse"] < 1e-10
    scene["names"] = np.array(["images/0.png.png", "images/1.png.png", "images/3.png.png"])
    np.savez(scene_path, **scene)
    with pytest.raises(ValueError, match="non-train"):
        evaluation_cameras(dataset, scene_path)


def test_metric_values_and_exact_test_set(tmp_path):
    pytest.importorskip("skimage")
    prediction = np.zeros((24, 32, 3))
    target = np.full_like(prediction, .5)
    metrics = rendering_metrics(prediction, target)
    assert metrics["psnr"] == pytest.approx(6.020599913279624)
    assert -1 <= metrics["ssim"] <= 1
    perfect = rendering_metrics(target, target)
    assert perfect["psnr"] == 120.
    assert perfect["ssim"] == pytest.approx(1.)
    dataset, manifest = toy_dataset(tmp_path)
    renders = tmp_path / "renders"
    (renders / "images").mkdir(parents=True)
    name = "images/3.png.png"
    shutil.copy2(dataset / name, renders / name)
    result = evaluate_renders(dataset, renders, lpips=False)
    assert result["test_frames"] == 1
    assert result["psnr"] == 120
    assert result["lpips"] is None
    assert not result["complete"]
    assert "partial" in result["lpips_status"]
    Image.new("RGB", (32, 24)).save(renders / "extra.png")
    with pytest.raises(ValueError, match="Render set mismatch"):
        evaluate_renders(dataset, renders, lpips=False)

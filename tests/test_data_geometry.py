import json
from pathlib import Path
import struct

import numpy as np
from PIL import Image
import pytest

from p4.data import load_manifest, prepare_dataset, read_colmap_model
from p4.geometry import (align_training_cameras, fit_similarity, invert_poses,
                         project_points, transform_points, transform_poses)


def make_nerf_source(path, count=6):
    path.mkdir()
    images = path / "images"
    images.mkdir()
    centers = [[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0], [0, 0, 1], [1, 0, 1]]
    frames = []
    for i in range(count):
        Image.new("RGB", (80, 40), (i * 20, 30, 40)).save(images / f"frame_{i}.png")
        pose = np.eye(4)
        pose[:3, 3] = centers[i]
        frames.append({"file_path": f"images/frame_{i}.png", "transform_matrix": pose.tolist()})
    source = {"w": 80, "h": 40, "fl_x": 60., "fl_y": 70., "cx": 40., "cy": 20., "frames": frames}
    (path / "transforms.json").write_text(json.dumps(source))
    return source


def test_explicit_split_resize_and_axis_conversion(tmp_path):
    source = tmp_path / "source"
    original = make_nerf_source(source)
    split = tmp_path / "split.json"
    split.write_text(json.dumps({"train": ["frame_0.png", "frame_1.png", "frame_2.png", "frame_4.png"],
                                 "test": ["frame_3.png", "frame_5.png"]}))
    out = tmp_path / "prepared"
    manifest = prepare_dataset(source, out, "toy", max_size=40, split_file=split)
    assert manifest["split_source"]["kind"] == "explicit"
    assert manifest["frames"][0]["width"] == 40
    assert manifest["frames"][0]["height"] == 20
    assert manifest["frames"][0]["sha256"]
    assert load_manifest(out)["split_hash"] == manifest["split_hash"]
    with np.load(out / "reference.npz") as reference:
        np.testing.assert_allclose(reference["K"][0], [[30, 0, 20], [0, 35, 10], [0, 0, 1]])
        np.testing.assert_allclose(reference["c2w"][0], np.diag([1, -1, -1, 1]))
        assert reference["names"][0] == "images/frame_0.png.png"
    # Intrinsic scaling preserves projective image position under resize.
    point = np.array([[.2, -.1, 2.]])
    oldK = np.array([[60, 0, 40], [0, 70, 20], [0, 0, 1.]])
    newK = oldK.copy()
    newK[:2] *= .5
    before, depth = project_points(point, np.eye(4), oldK)
    after, _ = project_points(point, np.eye(4), newK)
    np.testing.assert_allclose(after, before / 2)
    assert depth[0] > 0


def test_no_silent_fallback_and_split_overlap(tmp_path):
    source = tmp_path / "source"
    make_nerf_source(source)
    with pytest.raises(ValueError, match="No official split"):
        prepare_dataset(source, tmp_path / "a", "toy")
    manifest = prepare_dataset(source, tmp_path / "b", "toy", test_every=3)
    assert manifest["split_source"]["kind"] == "fallback"
    assert [f["id"] for f in manifest["frames"] if f["split"] == "test"] == ["frame_0.png", "frame_3.png"]
    split = tmp_path / "overlap.json"
    split.write_text(json.dumps({"train": ["frame_0.png"], "test": ["frame_0.png"]}))
    with pytest.raises(ValueError, match="overlapping"):
        prepare_dataset(source, tmp_path / "c", "toy", split_file=split)


def test_sim3_and_test_centers_not_used_for_alignment():
    source = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [.4, .2, 1.]])
    angle = .4
    R = np.array([[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])
    truth = {"scale": 2.3, "rotation": R, "translation": np.array([3, -2, .5])}
    target = transform_points(source, truth)
    fit = fit_similarity(source, target)
    np.testing.assert_allclose(fit["rotation"], R, atol=1e-12)
    np.testing.assert_allclose(fit["scale"], 2.3, atol=1e-12)
    poses = np.tile(np.eye(4), (4, 1, 1))
    poses[:, :3, 3] = source
    estimated = transform_poses(poses, truth)
    estimated[3, :3, 3] = [900, 800, 700]  # held-out pose must have no effect
    result = align_training_cameras(["a", "b", "c", "test"], estimated,
                                    ["a", "b", "c", "test"], poses, ["a", "b", "c"])
    np.testing.assert_allclose(result["reference_to_estimate"]["rotation"], R, atol=1e-12)
    assert result["ate_rmse"] < 1e-12
    assert result["train_names"] == ["a", "b", "c"]
    restored = invert_poses(transform_poses(poses, truth)) @ transform_poses(poses, truth)
    np.testing.assert_allclose(restored, np.tile(np.eye(4), (4, 1, 1)), atol=1e-12)


def test_sim3_rejects_unobservable_rotation():
    line = np.array([[0., 0, 0], [1, 0, 0], [2, 0, 0]])
    with pytest.raises(ValueError, match="Degenerate"):
        fit_similarity(line, line * 2)


def test_colmap_text_and_binary_read_identically(tmp_path):
    text = tmp_path / "text"
    binary = tmp_path / "binary"
    text.mkdir()
    binary.mkdir()
    (text / "cameras.txt").write_text("# comment\n1 PINHOLE 80 40 60 70 40 20\n")
    (text / "images.txt").write_text("# comment\n1 1 0 0 0 -1 -2 -3 1 image.png\n\n")
    with (binary / "cameras.bin").open("wb") as stream:
        stream.write(struct.pack("<QiiQQdddd", 1, 1, 1, 80, 40, 60, 70, 40, 20))
    with (binary / "images.bin").open("wb") as stream:
        stream.write(struct.pack("<Qidddddddi", 1, 1, 1, 0, 0, 0, -1, -2, -3, 1))
        stream.write(b"image.png\0")
        stream.write(struct.pack("<Q", 0))
    a, b = read_colmap_model(text), read_colmap_model(binary)
    np.testing.assert_allclose(a[0][1]["K"], b[0][1]["K"])
    np.testing.assert_allclose(a[1]["image.png"]["c2w"], b[1]["image.png"]["c2w"])
    np.testing.assert_allclose(a[1]["image.png"]["c2w"][:3, 3], [1, 2, 3])


def make_course_source(path):
    (path / "rgb").mkdir(parents=True)
    records = []
    for i, translation in enumerate(([1., 2, 3], [2., 2, 3], [1., 3, 3], [2., 3, 4])):
        name = f"frame_{i+1:05d}.png"
        Image.new("RGB", (64, 32), (i * 40, 10, 20)).save(path / "rgb" / name)
        records.append({"cam_id": i+1, "cam_quat": [0, np.sin(np.pi/4), 0, np.cos(np.pi/4)],
                        "cam_trans": translation, "image_name": name,
                        "fx": .8, "fy": .9, "cx": .5, "cy": .5})
    (path / "cameras.json").write_text(json.dumps(records))
    (path / "intrinsics.json").write_text(json.dumps({"fx": .8, "fy": .9, "cx": .5, "cy": .5}))
    return records


def test_actual_course_rgb_camera_schema_normalized_intrinsics_and_xyzw(tmp_path):
    source = tmp_path / "DL3DV-2"
    make_course_source(source)
    dataset = tmp_path / "prepared"
    result = prepare_dataset(source, dataset, "Dl3dv-2", test_every=4, max_size=32)
    assert result["scene"] == "Dl3dv-2"  # Source folder casing does not become experiment identity.
    assert result["source_format"] == "course_rgb_cameras_json"
    assert result["reference_kind"] == "reference"
    assert "d3a5c0926e5c5bceac362e5d0646e60575cbcd0a" in result["reference_convention_source"]
    assert result["split_source"]["kind"] == "fallback"
    with np.load(dataset / "reference.npz") as reference:
        np.testing.assert_allclose(reference["K"][0], [[25.6, 0, 16], [0, 14.4, 8], [0, 0, 1]])
        np.testing.assert_allclose(reference["c2w"][0, :3, :3], [[0, 0, 1], [0, 1, 0], [-1, 0, 0]], atol=1e-12)
        np.testing.assert_allclose(reference["c2w"][0, :3, 3], [1, 2, 3])
        assert reference["names"][0] == "images/frame_00001.png.png"
    assert len(result["frames"]) == 4


def test_course_partial_archive_rejected_and_split_still_required(tmp_path):
    source = tmp_path / "DL3DV-2"
    make_course_source(source)
    with pytest.raises(ValueError, match="No official split"):
        prepare_dataset(source, tmp_path / "unfrozen", "Dl3dv-2")
    (source / "rgb/frame_00004.png").unlink()
    with pytest.raises(ValueError, match="archive may be incomplete"):
        prepare_dataset(source, tmp_path / "partial", "Dl3dv-2", test_every=4)


def test_course_rgb_without_camera_record_rejected(tmp_path):
    source = tmp_path / "DL3DV-2"
    make_course_source(source)
    Image.new("RGB", (64, 32)).save(source / "rgb/extra.png")
    with pytest.raises(ValueError, match="absent from course cameras"):
        prepare_dataset(source, tmp_path / "prepared", "Dl3dv-2", test_every=4)


def test_course_images_schema_and_hidden_notebook_copies(tmp_path):
    source = tmp_path / "Re10k-1"
    make_course_source(source)
    (source / "rgb").rename(source / "images")
    hidden = source / "images/.ipynb_checkpoints"
    hidden.mkdir()
    Image.new("RGB", (64, 32)).save(hidden / "frame_00001-checkpoint.png")
    result = prepare_dataset(source, tmp_path / "prepared", "Re10k-1", test_every=4)
    assert len(result["frames"]) == 4
    assert result["source_format"] == "course_images_cameras_json"


def make_waymo_source(path):
    front = path / "FRONT"
    for folder in ("rgb", "gt", "gt_old", "calib"):
        (front / folder).mkdir(parents=True)
    yy, xx = np.meshgrid(np.arange(32), np.arange(48), indexing="ij")
    pixels = np.stack([(xx * 13 + yy * 7) % 256, (xx * 3 + yy * 11) % 256, (xx * yy) % 256], axis=-1).astype(np.uint8)
    rotation = np.array([[0., 0, 1], [0, 1, 0], [-1, 0, 0]])
    for i, translation in enumerate(([1., 2, 3], [2., 2, 3], [1., 3, 3], [2., 3, 4])):
        name = f"{i:06d}"
        Image.fromarray(pixels).save(front / "rgb" / (name + ".png"))
        pose = np.eye(4)
        pose[:3, :3] = rotation
        pose[:3, 3] = translation
        np.savetxt(front / "gt" / (name + ".txt"), pose)
        old = pose.copy()
        old[:3, 3] += 10000
        np.savetxt(front / "gt_old" / (name + ".txt"), old)
        (front / "calib" / (name + ".txt")).write_text(
            "Tr_velo_to_cam: 1 0 0 9000 0 1 0 8000 0 0 1 7000\n"
            "fx: 31.0 fy: 29.0 cx: 24.0 cy: 16.0\n"
            "k1: 0.15 k2: -0.02 p1: 0.001 p2: 0.002 k3: 0.0\n")
    return front, pixels


def test_actual_waymo_layout_gt_c2w_and_distortion_before_resize(tmp_path):
    cv2 = pytest.importorskip("cv2")
    source = tmp_path / "405841"
    front, pixels = make_waymo_source(source)
    dataset = tmp_path / "prepared"
    # Parent scene directory is accepted and resolves the FRONT camera.
    result = prepare_dataset(source, dataset, "Waymo-405841", test_every=4, max_size=24)
    assert result["source_format"] == "course_waymo_front"
    assert result["source"] == str(front.resolve())
    assert result["reference_kind"] == "reference"
    assert "gt_old" in result["reference_pose_convention"]
    K = np.array([[31., 0, 24], [0, 29., 16], [0, 0, 1]])
    expected = cv2.undistort(pixels, K, np.array([.15, -.02, .001, .002, 0]), None, K)
    expected = np.asarray(Image.fromarray(expected).resize((24, 16), Image.Resampling.LANCZOS))
    with Image.open(dataset / "images/000000.png.png") as canonical:
        np.testing.assert_array_equal(np.asarray(canonical), expected)
    with np.load(dataset / "reference.npz") as reference:
        np.testing.assert_allclose(reference["K"][0], [[15.5, 0, 12], [0, 14.5, 8], [0, 0, 1]])
        np.testing.assert_allclose(reference["c2w"][0, :3, :3], [[0, 0, 1], [0, 1, 0], [-1, 0, 0]])
        # Neither vehicle extrinsic nor gt_old offset is composed into camera c2w.
        np.testing.assert_allclose(reference["c2w"][0, :3, 3], [1, 2, 3])


def test_waymo_missing_rgb_or_calibration_is_explicit_failure(tmp_path):
    source = tmp_path / "405841"
    front, pixels = make_waymo_source(source)
    (front / "rgb/000003.png").unlink()
    with pytest.raises(ValueError, match="coverage differs"):
        prepare_dataset(source, tmp_path / "prepared", "Waymo-405841", test_every=4)
    Image.fromarray(pixels).save(front / "rgb/000003.png")
    (front / "calib/000003.txt").unlink()
    with pytest.raises(ValueError, match="lacks matching gt/calib"):
        prepare_dataset(source, tmp_path / "prepared2", "Waymo-405841", test_every=4)


def test_waymo_nonrigid_reference_pose_rejected(tmp_path):
    source = tmp_path / "405841"
    front, pixels = make_waymo_source(source)
    pose = np.eye(4)
    pose[0, 0] = 2.
    np.savetxt(front / "gt/000002.txt", pose)
    with pytest.raises(ValueError, match="not SO\\(3\\)"):
        prepare_dataset(source, tmp_path / "prepared", "Waymo-405841", test_every=4)

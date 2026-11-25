import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest

from p4.initialize import (
    _read_colmap_text, depth_to_points, preprocessing_geometry,
    run_initializer, train_frames,
)


def test_preprocessing_inverse_recovers_projection_after_crop_pad_resize():
    for method in ("vggt", "vggt-omega"):
        # Odd padding, landscape, portrait and an extreme crop are included.
        sizes = [(961, 540), (541, 960), (1600, 300)]
        a, bounds, hw = preprocessing_geometry(sizes, method)
        canonical_k = np.array([[800., 0, 400], [0, 790, 250], [0, 0, 1]])
        xyz = np.array([.1, -.2, 1.])
        for transform, bound in zip(a, bounds):
            predicted_k = transform @ canonical_k
            recovered = np.linalg.inv(transform) @ predicted_k
            np.testing.assert_allclose(recovered @ xyz, canonical_k @ xyz, atol=1e-10)
            assert 0 <= bound[0] < bound[2] <= hw[1]
            assert 0 <= bound[1] < bound[3] <= hw[0]
    # VGGT square518 geometry uses exact original integer padding.
    a, bounds, _ = preprocessing_geometry([(961, 540)], "vggt")
    assert a[0, 1, 2] == pytest.approx(210 * 518 / 961)
    # Omega crops panoramic input rather than changing the canonical images.
    a, _, _ = preprocessing_geometry([(1600, 300)], "vggt-omega")
    assert a[0, 0, 2] < 0


def test_depth_export_filters_padding_nan_and_negative_and_preserves_colors():
    depth = np.ones((1, 3, 4)) * 2
    confidence = np.arange(12).reshape(1, 3, 4).astype(float)
    depth[0, 1, 1], confidence[0, 1, 2] = -1, np.nan
    rgb = np.arange(36).reshape(1, 3, 4, 3).astype(np.uint8)
    w2c = np.eye(4)[None]
    w2c[0, 0, 3] = -10  # Camera center is at world x=10.
    points, colors, meta = depth_to_points(depth, confidence, w2c, np.eye(3)[None], rgb,
                                           np.array([[1, 0, 3, 3]]), confidence_quantile=0)
    np.testing.assert_allclose(points, [[12, 0, 2], [14, 0, 2], [12, 4, 2], [14, 4, 2]])
    np.testing.assert_array_equal(colors, rgb[0, [0, 0, 2, 2], [1, 2, 1, 2]])
    assert meta["valid_points_before_cap"] == 4


def test_point_limit_is_deterministic_and_applies_identical_point_color_indices():
    rgb = np.arange(300).reshape(1, 10, 10, 3).astype(np.uint8)
    args = (np.ones((1, 10, 10)), np.ones((1, 10, 10)), np.eye(4)[None],
            np.eye(3)[None], rgb, np.array([[0, 0, 10, 10]]))
    points, colors, _ = depth_to_points(*args, max_points=7, seed=9)
    points2, colors2, _ = depth_to_points(*args, max_points=7, seed=9)
    np.testing.assert_array_equal(points, points2)
    np.testing.assert_array_equal(colors, colors2)
    assert len(points) == 7
    np.testing.assert_array_equal(colors, rgb[0, points[:, 1].astype(int), points[:, 0].astype(int)])


def _toy_dataset(tmp_path):
    dataset = tmp_path / "dataset"
    (dataset / "images").mkdir(parents=True)
    frames = []
    for i in range(3):
        Image.new("RGB", (16, 12)).save(dataset / f"images/{i}.png")
        frames.append({"id": str(i), "image": f"images/{i}.png", "split": "test" if i == 1 else "train", "width": 16, "height": 12,
                       "sha256": hashlib.sha256((dataset / f"images/{i}.png").read_bytes()).hexdigest()})
    (dataset / "manifest.json").write_text(json.dumps({"schema_version": 1, "scene": "toy", "frames": frames}))
    # Any reference file contents, including an invalid file, must be irrelevant.
    (dataset / "reference.npz").write_bytes(b"DO NOT OPEN")
    return dataset


@pytest.mark.parametrize("head_offload,backend,strategy", [
    (False, "native", "stock"),
    (True, "native", "cpu_head_features"),
    (False, "offload", "cpu_disk_layers"),
])
def test_only_manifest_train_rgb_reaches_initializer(tmp_path, monkeypatch, head_offload, backend, strategy):
    dataset = _toy_dataset(tmp_path)
    seen = []
    def mocked_neural(dataset, output, frames, *args):
        assert args[-2:] == (head_offload, backend)
        seen.extend(f["image"] for f in frames)
        return {"registered_frames": len(frames), "initial_points": 1}
    monkeypatch.setattr("p4.initialize._run_neural", mocked_neural)
    result = run_initializer(dataset, tmp_path / "init", "vggt",
                             vggt_offload_features=head_offload, neural_backend=backend)
    assert result["memory_strategy"] == strategy
    assert seen == ["images/0.png", "images/2.png"]
    assert result["status"] == "completed"


def test_failed_initialization_keeps_metadata_and_refuses_stale_reuse(tmp_path, monkeypatch):
    dataset = _toy_dataset(tmp_path)
    def fail(*args):
        raise RuntimeError("GPU unavailable")
    monkeypatch.setattr("p4.initialize._run_neural", fail)
    out = tmp_path / "init"
    with pytest.raises(RuntimeError, match="GPU unavailable"):
        run_initializer(dataset, out, "vggt")
    metadata = json.loads((out / "scene.json").read_text())
    assert metadata["status"] == "failed"
    assert "GPU unavailable" in metadata["error"]
    with pytest.raises(FileExistsError):
        run_initializer(dataset, out, "vggt")



def test_colmap_reproduction_flags_reach_native_processes(tmp_path, monkeypatch):
    dataset = _toy_dataset(tmp_path)
    from p4 import initialize

    calls = []
    def run(command, **kwargs):
        calls.append(command)
        if command[1] == "mapper":
            (Path(command[command.index("--output_path") + 1]) / "0").mkdir()
        elif command[1] == "model_converter":
            model = Path(command[command.index("--output_path") + 1])
            (model / "cameras.txt").write_text("1 PINHOLE 16 12 10 11 8 6\n")
            (model / "images.txt").write_text(
                "1 1 0 0 0 0 0 0 1 images/0.png\n\n2 1 0 0 0 -2 0 0 1 images/2.png\n\n"
            )
            (model / "points3D.txt").write_text("1 2 3 4 255 100 0 .01 1 0 2 0\n")
        return SimpleNamespace(stdout="COLMAP 3.11.1")

    monkeypatch.setattr(initialize.shutil, "which", lambda _: "colmap")
    monkeypatch.setattr(initialize.subprocess, "run", run)
    out = tmp_path / "init"
    meta = run_initializer(dataset, out, "colmap", colmap_single_camera=True, colmap_num_threads=8,
                           colmap_exhaustive_block_size=200)
    assert meta["registered_frames"] == 2
    assert meta["colmap_options"] == {"single_camera": True, "num_threads": 8, "exhaustive_block_size": 200}
    assert meta["commands"] == calls[:3]
    assert [json.loads(line) for line in (out / "colmap.log").read_text().splitlines()] == calls[:3]
    assert calls[0][calls[0].index("--ImageReader.single_camera") + 1] == "1"
    assert calls[0][-2:] == ["--SiftExtraction.num_threads", "8"]
    assert calls[1][-4:] == ["--ExhaustiveMatching.block_size", "200", "--SiftMatching.num_threads", "8"]
    assert calls[2][-2:] == ["--Mapper.num_threads", "8"]


def test_colmap_text_opencv_convention_and_empty_observation_line(tmp_path):
    (tmp_path / "cameras.txt").write_text("# Camera\n1 PINHOLE 16 12 10 11 8 6\n")
    (tmp_path / "images.txt").write_text("# Image\n1 1 0 0 0 -2 0 0 1 images/0.png\n\n2 1 0 0 0 0 0 0 1 images/2.png\n0 0 1\n")
    (tmp_path / "points3D.txt").write_text("# points\n1 2 3 4 255 100 0 .01 1 0 2 0\n")
    images, points, colors = _read_colmap_text(tmp_path)
    assert set(images) == {"images/0.png", "images/2.png"}
    np.testing.assert_allclose(images["images/0.png"][0][:3, 3], [2, 0, 0])
    np.testing.assert_allclose(images["images/0.png"][1], [[10, 0, 8], [0, 11, 6], [0, 0, 1]])
    np.testing.assert_array_equal(points, [[2, 3, 4]])
    np.testing.assert_array_equal(colors, [[255, 100, 0]])


def test_manifest_train_paths_cannot_escape_dataset(tmp_path):
    dataset = _toy_dataset(tmp_path)
    manifest = json.loads((dataset / "manifest.json").read_text())
    manifest["frames"][0]["image"] = "../outside.png"
    (tmp_path / "outside.png").write_bytes(b"x")
    (dataset / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        train_frames(dataset)


def test_initializer_checks_only_training_rgb_hashes(tmp_path):
    dataset = _toy_dataset(tmp_path)
    # Test RGB does not participate in initializer preflight or model input.
    (dataset / "images/1.png").write_bytes(b"not used for initialization")
    _, frames = train_frames(dataset)
    assert len(frames) == 2
    Image.new("RGB", (16, 12), "red").save(dataset / "images/0.png")
    with pytest.raises(ValueError, match="RGB hash differs"):
        train_frames(dataset)

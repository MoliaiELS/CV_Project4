import json

import numpy as np
from PIL import Image
import pytest

from p4.common import sha256_file
from p4.initialize import preprocessing_geometry
from scripts.extract_vggt_depth_cache import _load_inputs, _open_features, _prepared_geometry, _validate_gauge


def test_prepared_depth_applies_pixel_transform_without_scaling_z():
    model = np.arange(48, dtype=np.float32).reshape(1, 6, 8) + 3
    affine = np.array([[[2., 0., 1.], [0., 2., 0.], [0., 0., 1.]]])
    depth, confidence = _prepared_geometry(model, model * 2, affine, (3, 2))
    expected = model[:, :3:2, 1:6:2]
    np.testing.assert_array_equal(depth, expected)
    np.testing.assert_array_equal(confidence, expected * 2)
    assert depth.dtype == confidence.dtype == np.float32


def test_cache_schema_is_memory_mapped_and_exact(tmp_path):
    for layer in (4, 11, 17, 23):
        np.save(tmp_path / f"layer_{layer:02d}.npy", np.zeros((1, 2, 6, 2048), dtype=np.float32))
    arrays, fingerprints = _open_features(tmp_path, 2, (14, 14))
    assert all(isinstance(array, np.memmap) and not array.flags.writeable for array in arrays.values())
    assert fingerprints == {path.name: sha256_file(path) for path in tmp_path.iterdir()}
    with pytest.raises(ValueError, match="feature-cache schema"):
        _open_features(tmp_path, 3, (14, 14))


def test_archive_inputs_require_train_rgb_only_and_preserve_gauge(tmp_path):
    dataset = tmp_path / "data"
    dataset.mkdir()
    names = ["first.png", "second.png"]
    frames = []
    for name in names:
        file = dataset / name
        Image.new("RGB", (5, 7)).save(file)
        frames.append({"image": name, "split": "train", "width": 5, "height": 7, "sha256": sha256_file(file)})
    frames.append({"image": "missing-test.png", "split": "test"})
    manifest = dataset / "manifest.json"
    manifest.write_text(json.dumps({"schema_version": 1, "frames": frames}), encoding="utf-8")
    model = tmp_path / "scene.npz"
    c2w = np.stack([np.eye(4)] * 2)
    c2w[1, 0, 3] = 1.
    K = np.stack([np.eye(3)] * 2)
    np.savez(model, names=names, c2w=c2w, K=K, points=np.ones((1, 3)), colors=np.zeros((1, 3), np.uint8))
    transform, bounds, model_size = preprocessing_geometry([(5, 7)] * 2, "vggt")
    meta = {"method": "vggt", "neural_backend": "offload", "status": "completed", "input_names": names,
            "input_frames": 2, "manifest_sha256": sha256_file(manifest), "model_input_hw": model_size,
            "canonical_to_model": transform.tolist(), "valid_model_bounds": bounds.tolist()}
    model.with_suffix(".json").write_text(json.dumps(meta), encoding="utf-8")

    _, original, _, paths, size, _, _, _ = _load_inputs(dataset, model)
    assert paths == [str(dataset / name) for name in names]  # no test image or reference archive exists
    assert size == (5, 7)
    assert _validate_gauge(original, {**original, "K": K * 1.2}) == 1.
    shifted = c2w.copy()
    shifted[1, 0, 3] += .01
    with pytest.raises(ValueError, match="anchor-center gauge"):
        _validate_gauge(original, {**original, "c2w": shifted})

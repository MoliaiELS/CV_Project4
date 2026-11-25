import json

import numpy as np
import pytest

from p4.common import sha256_file
from scripts.train_sparse_depth import _depth_target, _load_depth_maps, depth_losses


def test_depth_cache_checks_digest_geometry_and_exact_order(tmp_path):
    path, receipt_path = tmp_path / "depth_maps.npz", tmp_path / "depth_receipt.json"
    names = ["b.png", "a.png"]
    depth = np.stack([np.full((12, 14), 2.), np.full((12, 14), 3.)])
    valid = np.ones_like(depth)
    np.savez_compressed(path, names=names, depth=depth, confidence=valid,
                        split_hash="set", camera_initializer_sha256="input")
    manifest = {"split_hash": "set", "frames": [{"image": name, "split": "train", "height": 12, "width": 14}
                                                 for name in names]}
    receipt = {"depth_maps_sha256": sha256_file(path), "manifest_sha256": "manifest",
               "camera_initializer_sha256": "input", "training_ids": names, "split_hash": "set",
               "reference_data_used": False, "depth_units": "native VGGT camera-axis Z; not normalized or fit",
               "prepared_hw": [12, 14]}
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    fingerprint = {"manifest_sha256": "manifest", "scene_sha256": "input"}
    actual = _load_depth_maps(path, receipt_path, manifest, {"names": np.array(names)}, fingerprint)
    np.testing.assert_equal(actual[0], depth)
    np.testing.assert_equal(actual[1], valid)
    with pytest.raises(ValueError, match="authenticate"):
        _load_depth_maps(path, receipt_path, manifest, {"names": np.array(names[::-1])}, fingerprint)
    path.write_bytes(path.read_bytes() + b"corrupted")
    with pytest.raises(ValueError, match="authenticate"):
        _load_depth_maps(path, receipt_path, manifest, {"names": np.array(names)}, fingerprint)


def test_depth_target_preserves_camera_z_unit_conversion_and_resize():
    z = np.full((12, 20), 4., dtype=np.float32)
    z[1, 1] = np.nan
    log_z, weight, resize = _depth_target(z, np.ones_like(z), 2., 10)
    assert log_z.shape == weight.shape == (6, 10)
    assert log_z[0, 0] == weight[0, 0] == 0.
    np.testing.assert_allclose(log_z[1:], np.log(2.))
    np.testing.assert_equal(weight[1:], 1.)
    np.testing.assert_equal(resize, np.diag([0.5, 0.5, 1.]))


def test_depth_loss_has_global_scale_and_patch_relative_gradients():
    torch = pytest.importorskip("torch")
    exact = torch.linspace(1., 2., 16 * 16).reshape(16, 16)
    target = exact.log()
    confidence = torch.ones_like(exact)
    alpha = torch.ones_like(exact)
    global_loss, local_loss, support = depth_losses(exact, alpha, target, confidence)
    assert (global_loss.item(), local_loss.item(), support.item()) == (0., 0., 1.)
    scaled = (exact * 2).requires_grad_()
    global_loss, local_loss, _ = depth_losses(scaled, alpha, target, confidence)
    assert global_loss.item() > .5
    assert local_loss.item() < 1e-10
    perturbation = exact.clone()
    perturbation[:3, :3] *= 2.
    perturbation[0, 0] = float("nan")
    perturbation.requires_grad_()
    alpha[0, 0] = 0.
    global_loss, local_loss, support = depth_losses(perturbation, alpha, target, confidence)
    assert local_loss.item() > 0.
    assert support.item() == pytest.approx(255 / 256)
    (global_loss + local_loss).backward()
    assert torch.isfinite(perturbation.grad).all()
    assert perturbation.grad[0, 0].item() == 0.

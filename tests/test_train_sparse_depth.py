import json

import numpy as np
from PIL import Image
import pytest

from p4.common import sha256_file
from scripts.part3_views import Camera
from scripts.sparse_warp_teacher import build_warp_teacher, warp_teacher_loss
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


def test_warp_teacher_covers_each_train_gap_and_records_deterministic_labels(tmp_path):
    names = ["left.png", "center.png", "right.png"]
    cameras = []
    for name, offset in zip(names, (-.02, 0., .02)):
        Image.new("RGB", (16, 16), (120, 80, 40)).save(tmp_path / name)
        pose = np.eye(4)
        pose[0, 3] = offset
        camera_K = np.array([[8., 0., 8.], [0., 8., 8.], [0., 0., 1.]])
        cameras.append(Camera(name, pose, camera_K, 16, 16))
    z = np.full((3, 16, 16), 2., dtype=np.float32)
    confidence = np.ones_like(z)
    views, receipt = build_warp_teacher(tmp_path, cameras, names[::-1], z, confidence, 1.)
    assert [view.camera.name for view in views] == [f"interval_{g:03d}_{f}" for g in range(2) for f in (33, 50, 67)]
    assert receipt["views"] == 6 and receipt["mean_coverage"] > .95
    assert receipt["per_view"][4]["anchors"] == names[1:]
    np.testing.assert_allclose(views[4].rgb[8, 8], np.asarray([120, 80, 40]) / 255., atol=1e-6)
    _, replay = build_warp_teacher(tmp_path, cameras, names, z, confidence, 1.)
    assert replay["teacher_sha256"] == receipt["teacher_sha256"]
    with pytest.raises(ValueError, match="each real training camera"):
        build_warp_teacher(tmp_path, cameras, [*names[:2], "heldout.png"], z, confidence, 1.)


def test_warp_teacher_loss_downweights_tiny_confidence_support():
    torch = pytest.importorskip("torch")
    rgb = torch.ones(4, 4, 3, requires_grad=True)
    target = torch.zeros_like(rgb)
    confidence = torch.zeros(4, 4)
    confidence[0, 0] = .5
    loss = warp_teacher_loss(rgb, target, confidence)
    assert loss.item() == pytest.approx(.5 * .95 / (0.25 * 16))
    loss.backward()
    assert torch.count_nonzero(rgb.grad).item() == 3

import numpy as np
import pytest

from scripts.train_joint_poses import PoseConfig, _load_base, _normalization_poses, _poses_for_export, apply_pose_delta


def test_pose_schedule_has_warmup_and_frozen_tail():
    PoseConfig().validate(7000)
    with pytest.raises(ValueError, match="warmup"):
        PoseConfig(stop=7000).validate(7000)
    with pytest.raises(ValueError, match="finite"):
        PoseConfig(learning_rate=float("nan")).validate(7000)


def test_joint_checkpoint_marker_only_accepts_joint_training_state():
    class Saved:
        def __init__(self, payload):
            self.payload = payload

        def load(self, *args, **kwargs):
            return self.payload

    fingerprint = {"scene_sha256": "known", "manifest_sha256": "data"}
    tagged = {"schema_version": 1, "fingerprint": {**fingerprint, "requires_joint_export": True}}
    with pytest.raises(ValueError, match="incompatible"):
        _load_base(Saved(tagged), "raw.pt", fingerprint)
    assert _load_base(Saved({**tagged, "joint": {}}), "raw.pt", fingerprint)["joint"] == {}


def test_pose_update_and_export_use_the_original_scene_frame():
    torch = pytest.importorskip("torch")
    poses = np.broadcast_to(np.eye(4), (2, 4, 4)).copy()
    poses[:, :3, 3] = [[2, 5, 9], [4, 5, 9]]
    normalization = {"origin": [3, 5, 9], "radius": 2.0}
    delta = torch.tensor([[0, 0, 0, 0, 0, 0], [0, 0, np.pi / 2, .1, .2, .3]], dtype=torch.float64)
    exported = _poses_for_export({"c2w": poses}, delta, normalization)
    np.testing.assert_array_equal(exported[0], poses[0])
    np.testing.assert_allclose(exported[1, :3, :3], [[0, -1, 0], [1, 0, 0], [0, 0, 1]], atol=1e-6)
    np.testing.assert_allclose(exported[1, :3, 3], [4.2, 5.4, 9.6])
    normalized = torch.from_numpy(_normalization_poses(poses, normalization)).double()
    assert torch.isfinite(torch.autograd.functional.jacobian(
        lambda d: apply_pose_delta(normalized, d), torch.zeros_like(delta))).all()


def test_rasterizer_has_a_photometric_camera_gradient():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("The pinned rasterizer requires CUDA")
    pytest.importorskip("gsplat")
    from p4.train import _rasterize

    def tensor(values):
        return torch.tensor(values, dtype=torch.float32, device="cuda")

    splats = {
        "means": tensor([[.2, .1, 2.2]]), "scales": tensor([[-2.2, -2.2, -2.2]]),
        "quats": tensor([[1, 0, 0, 0]]), "opacities": tensor([2]),
        "sh0": tensor([[[1, 0, 0]]]), "shN": torch.zeros((1, 0, 3), device="cuda"),
    }
    intrinsics = tensor([[47, 0, 24], [0, 47, 24], [0, 0, 1]])
    base = torch.eye(4, device="cuda")
    x = torch.arange(48, device="cuda")

    def red_centroid(delta):
        rgb, _, _ = _rasterize(splats, apply_pose_delta(base, delta), intrinsics, 48, 48, 0)
        red = rgb[0, :, :, 0]
        return (red * x[None, :]).sum() / red.sum()

    zero = torch.zeros(6, device="cuda", requires_grad=True)
    analytical = torch.autograd.grad(red_centroid(zero), zero)[0][3].item()
    perturbation = tensor([0, 0, 0, .002, 0, 0])
    with torch.no_grad():
        numerical = ((red_centroid(perturbation) - red_centroid(-perturbation)) / .004).item()
    assert analytical < -5
    assert analytical == pytest.approx(numerical, rel=.08)

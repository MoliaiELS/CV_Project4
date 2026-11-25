"""CPU checks of the geometry/protocol used by the CUDA runner."""
from dataclasses import asdict
import json
import subprocess
import sys

import numpy as np
import pytest

from p4.geometry import project_points
from p4.train import TrainConfig, normalized_geometry


def test_normalization_preserves_cameras_and_projected_pixels():
    angles = [.1, -.2, .4]
    poses = np.tile(np.eye(4), (3, 1, 1))
    for i, angle in enumerate(angles):
        poses[i, :3, :3] = [[np.cos(angle), 0, np.sin(angle)], [0, 1, 0], [-np.sin(angle), 0, np.cos(angle)]]
    poses[:, :3, 3] = [[10, 11, -2], [14, 10, -1], [13, 13, -1]]
    points = np.array([[12., 12., 5.], [10.5, 13., 7.], [13.1, 11., 10.]])
    K = np.array([[600., 0, 480], [0, 620., 320], [0, 0, 1]])
    normalized_poses, normalized_points, normalization = normalized_geometry({"c2w": poses, "points": points})
    assert normalization["radius"] > 0
    np.testing.assert_allclose(normalized_poses[:, :3, :3], poses[:, :3, :3], atol=1e-7)
    np.testing.assert_allclose(normalized_poses[:, :3, 3].mean(0), [0, 0, 0], atol=1e-6)
    for before, after in zip(poses, normalized_poses):
        original_pixels, original_depth = project_points(points, before, K)
        normalized_pixels, normalized_depth = project_points(normalized_points, after, K)
        np.testing.assert_allclose(normalized_pixels, original_pixels, atol=2e-4)
        np.testing.assert_allclose(normalized_depth * normalization["radius"], original_depth, atol=1e-6)
    # Function must not mutate the initialization, whose hash binds checkpoint provenance.
    np.testing.assert_array_equal(poses[:, :3, 3], [[10, 11, -2], [14, 10, -1], [13, 13, -1]])


def test_zero_extent_initialization_fails_before_training():
    with pytest.raises(ValueError, match="extent is zero"):
        normalized_geometry({"c2w": np.tile(np.eye(4), (3, 1, 1)), "points": np.ones((4, 3))})


@pytest.mark.parametrize("change", [{"steps": 0}, {"sh_degree": 4}, {"ssim_lambda": 1.1},
                                    {"refine_every": 0}, {"max_gaussians": 3}])
def test_invalid_training_protocol_rejected(change):
    with pytest.raises(ValueError):
        TrainConfig(**change).validate()


def test_cli_help_available_without_cuda_packages():
    result = subprocess.run([sys.executable, "-m", "p4.train", "train", "--help"],
                            capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
    assert "--resume" in result.stdout
    assert "--stop-after" in result.stdout

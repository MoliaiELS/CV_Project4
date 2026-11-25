import math
from statistics import fmean

import pytest

from scripts.compare_sparse_quality import compare


def comparison_input():
    frames = [
        {"id": "a", "image": "a.png", "split": "test", "width": 2, "height": 2},
        {"id": "b", "image": "b.png", "split": "test", "width": 4, "height": 2},
    ]
    manifest = {"scene": "fixture", "split_hash": "fixed", "frames": frames}

    def scores(errors):
        pairs = [{"id": frame["id"], "image": frame["image"], "psnr": -10 * math.log10(error)}
                 for frame, error in zip(frames, errors)]
        return {"scene": "fixture", "split_hash": "fixed", "test_frames": len(frames),
                "psnr": fmean(pair["psnr"] for pair in pairs), "ssim": 0.8, "lpips": 0.2,
                "protocol": {"postprocessing": "none", "test_rgb_in_reconstruction": False,
                             "psnr_mse_floor": 1e-12},
                "per_frame": pairs}

    return manifest, scores([0.04, 0.04]), scores([0.01, 0.05])


def test_nonuniform_frames_can_meet_geometric_but_not_pooled_mse_target():
    manifest, baseline, candidate = comparison_input()
    result = compare(baseline, candidate, manifest)
    assert result["geometric_mse_reduction_percent"] == pytest.approx(44.09830056)
    assert result["pooled_mse_reduction_percent"] == pytest.approx(8.33333333)
    assert result["geometric_mse_target_met"] and not result["pooled_mse_target_met"]


def test_candidate_must_use_the_same_ordered_heldout_frames():
    manifest, baseline, candidate = comparison_input()
    candidate["per_frame"].reverse()
    with pytest.raises(ValueError, match="ordered held-out"):
        compare(baseline, candidate, manifest)

"""Compare raw renders on the same held-out frames in the MSE and PSNR domains."""

import argparse
import json
import math
from pathlib import Path
from statistics import fmean

from p4.common import sha256_file, write_json
from p4.data import load_manifest


def compare(baseline, candidate, manifest, target_percent=30.0):
    """Recover MSE from the evaluator's per-frame PSNR before aggregating."""
    if not 0.0 < target_percent < 100.0:
        raise ValueError("The MSE reduction target must be in (0, 100)")
    frames = [frame for frame in manifest["frames"] if frame["split"] == "test"]
    expected = [(frame["id"], frame["image"]) for frame in frames]
    if not frames or len(set(expected)) != len(expected):
        raise ValueError("The manifest must have distinct held-out entries")
    pixels = [frame["width"] * frame["height"] for frame in frames]
    if min(pixels) <= 0:
        raise ValueError("Held-out frames must have positive pixel counts")

    scores = []
    for record in (baseline, candidate):
        actual = [(frame["id"], frame["image"]) for frame in record["per_frame"]]
        if (record["scene"] != manifest["scene"] or record["split_hash"] != manifest["split_hash"]
                or record["test_frames"] != len(expected) or actual != expected):
            raise ValueError("Both scores must use the exact ordered held-out split")
        protocol = record["protocol"]
        if (protocol["postprocessing"] != "none" or protocol["test_rgb_in_reconstruction"] is not False
                or protocol["psnr_mse_floor"] != 1e-12):
            raise ValueError("Both scores must use the project's raw-render protocol")
        values = [float(row["psnr"]) for row in record["per_frame"]]
        if any(not math.isfinite(value) or value >= 120.0 - 1e-8 for value in values):
            raise ValueError("Recover exact squared errors from raw renders when PSNR hits the evaluator floor")
        frame_mse = [10.0 ** (-value / 10.0) for value in values]
        mean_psnr = fmean(values)
        if not math.isclose(mean_psnr, record["psnr"], rel_tol=1e-10):
            raise ValueError("Reported PSNR does not agree with the per-frame scores")
        scores.append({
            "mean_frame_psnr_db": mean_psnr,
            "geometric_mean_frame_mse": 10.0 ** (-mean_psnr / 10.0),
            "pooled_rgb_mse": math.fsum(error * size for error, size in zip(frame_mse, pixels)) / sum(pixels),
            "ssim": record["ssim"],
            "lpips": record["lpips"],
        })

    before, after = scores
    geometric = 100.0 * (1.0 - after["geometric_mean_frame_mse"] / before["geometric_mean_frame_mse"])
    pooled = 100.0 * (1.0 - after["pooled_rgb_mse"] / before["pooled_rgb_mse"])
    improvement = after["mean_frame_psnr_db"] - before["mean_frame_psnr_db"]
    improved = sum(b["psnr"] > a["psnr"] for a, b in zip(baseline["per_frame"], candidate["per_frame"]))
    return {
        "scene": manifest["scene"], "split_hash": manifest["split_hash"], "test_frames": len(frames),
        "baseline": before, "candidate": after, "psnr_improvement_db": improvement,
        "improved_frames": improved, "geometric_mse_reduction_percent": geometric,
        "pooled_mse_reduction_percent": pooled, "target_reduction_percent": target_percent,
        "mean_psnr_target_db": before["mean_frame_psnr_db"] - 10.0 * math.log10(1.0 - target_percent / 100.0),
        "geometric_mse_target_met": geometric >= target_percent,
        "pooled_mse_target_met": pooled >= target_percent,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("dataset", "baseline", "candidate", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--target-percent", type=float, default=30.0)
    args = parser.parse_args()
    baseline, candidate = [json.loads(path.read_text(encoding="utf-8"))
                           for path in (args.baseline, args.candidate)]
    result = compare(baseline, candidate, load_manifest(args.dataset), args.target_percent)
    result.update(baseline_metrics_sha256=sha256_file(args.baseline),
                  candidate_metrics_sha256=sha256_file(args.candidate))
    write_json(args.output, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

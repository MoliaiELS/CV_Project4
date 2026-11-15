"""Server-only synthetic regression: optimize, save, resume, render and score.

This is NOT a course scene result. Invoke from repository root after setup:
  python scripts/gpu_smoke.py --output runs/gpu_smoke
"""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from PIL import Image

from p4.common import sha256_file, write_json
from p4.train import TrainConfig, train, render, cuda_dependencies


def synthetic_dataset(root):
    root.mkdir(parents=True)
    (root / "images").mkdir()
    poses, Ks, frames = [], [], []
    for i in range(8):
        angle = i * np.pi / 4
        center = np.array([0.35 * np.cos(angle), 0.35 * np.sin(angle), -2.0])
        forward = -center / np.linalg.norm(center)
        right = np.cross(np.array([0., 1., 0.]), forward)
        right /= np.linalg.norm(right)
        down = np.cross(forward, right)
        pose = np.eye(4)
        pose[:3, :3] = np.stack([right, down, forward], axis=1)
        pose[:3, 3] = center
        poses.append(pose)
        Ks.append(np.array([[48., 0., 32.], [0., 48., 32.], [0., 0., 1.]]))
        y, x = np.mgrid[:64, :64]
        rgb = np.stack([x / 63, y / 63, np.full_like(x, 0.4 + 0.02 * i, dtype=float)], axis=-1)
        name = f"images/{i:03d}.png"
        Image.fromarray(np.round(rgb * 255).astype(np.uint8)).save(root / name)
        frames.append({"id": str(i), "image": name, "split": "test" if i in (3, 7) else "train",
                       "width": 64, "height": 64, "sha256": sha256_file(root / name)})
    manifest = {"schema_version": 1, "scene": "SYNTHETIC_GPU_REGRESSION_NOT_COURSE_RESULT", "frames": frames,
                "split_source": "synthetic regression", "split_hash": "synthetic-v1", "reference_kind": "synthetic"}
    write_json(root / "manifest.json", manifest)
    poses, Ks = np.stack(poses), np.stack(Ks)
    np.savez_compressed(root / "reference.npz", names=np.array([f["image"] for f in frames]), c2w=poses, K=Ks)
    rng = np.random.default_rng(123)
    points = rng.uniform(-0.4, 0.4, (160, 3))
    allowed = [i for i, frame in enumerate(frames) if frame["split"] == "train"]
    scene = root / "scene.npz"
    np.savez_compressed(scene, names=np.array([frames[i]["image"] for i in allowed]),
                        c2w=poses[allowed], K=Ks[allowed], points=points.astype(np.float32),
                        colors=rng.integers(20, 230, (160, 3), dtype=np.uint8))
    return scene


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="runs/gpu_smoke")
    args = parser.parse_args()
    root = Path(args.output)
    if root.exists():
        raise SystemExit("Smoke output already exists; choose a new output directory to preserve evidence")
    torch = cuda_dependencies()
    scene = synthetic_dataset(root / "dataset")
    cfg = TrainConfig(steps=40, save_steps=[20, 40], log_every=10)
    train(root / "dataset", scene, root / "continuous", cfg)
    # Compare both continuations from precisely the same optimizer state.
    # Independent CUDA trajectories can diverge before the resume point.
    train(root / "dataset", scene, root / "resumed", cfg,
          resume=root / "continuous/checkpoints/step_000020.pt")
    a = torch.load(root / "continuous/checkpoints/step_000040.pt", map_location="cpu", weights_only=False)
    b = torch.load(root / "resumed/checkpoints/step_000040.pt", map_location="cpu", weights_only=False)
    differences = {key: float((a["splats"][key] - b["splats"][key]).abs().max()) for key in a["splats"]}
    # CUDA atomics can differ slightly; check numerical agreement, not bitwise equality.
    for key in differences:
        torch.testing.assert_close(a["splats"][key], b["splats"][key], atol=3e-4, rtol=3e-4)
    assert a["sampler"] == b["sampler"] and a["cursor"] == b["cursor"] and a["order"] == b["order"]
    assert a["scheduler"] == b["scheduler"]
    render(root / "dataset", scene, root / "resumed/checkpoints/step_000040.pt", root / "renders")
    from p4.evaluate import evaluate_renders
    metrics = evaluate_renders(root / "dataset", root / "renders", lpips=True)
    # A separate short run executes real densification, pruning and opacity reset.
    dense_cfg = TrainConfig(steps=30, save_steps=[30], refine_start=5, refine_stop=25,
                            refine_every=5, reset_every=10, log_every=5)
    train(root / "dataset", scene, root / "densification", dense_cfg)
    result = {"synthetic_only": True, "resume_parameter_max_abs_error": differences,
              "sampler_and_schedule_resume": "passed", "densification_and_reset_execution": "passed",
              "rendering_metrics": metrics, "complete": bool(metrics["complete"])}
    write_json(root / "verification.json", result)
    print(json.dumps(result, indent=2))
    if not metrics["complete"]:
        raise SystemExit("LPIPS assets unavailable; server smoke not complete")


if __name__ == "__main__":
    main()

"""Score a joint fit using the held-out camera alignment of its fixed-pose control."""

import argparse
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image

from p4.common import sha256_file, write_json
from p4.data import load_manifest, load_scene
from p4.evaluate import evaluate_renders, evaluation_cameras
from p4.train import _rasterize, cuda_dependencies, normalized_geometry
from scripts.part3_fusion import _normalized_poses, load_local_checkpoint


def evaluate(dataset, initializer, refined, output):
    dataset, initializer, refined, output = map(Path, (dataset, initializer, refined, output))
    original, updated = (load_scene(path) for path in (initializer, refined / "scene.npz"))
    if any(not np.array_equal(original[key], updated[key]) for key in ("names", "K", "points", "colors")):
        raise ValueError("Alignment diagnostic requires the same initializer apart from its learned train poses")
    if not np.array_equal(original["c2w"][0], updated["c2w"][0]):
        raise ValueError("The frozen train-camera anchor differs")
    receipt = json.loads((refined / "export.json").read_text(encoding="utf-8"))
    if sha256_file(initializer) != receipt["original_scene_sha256"]:
        raise ValueError("The original alignment initializer differs from the joint-run receipt")
    model = load_local_checkpoint(dataset, refined / "scene.npz", refined / "checkpoint.pt",
                                  receipt["checkpoint_sha256"])
    if model.normalization != normalized_geometry(original)[2]:
        raise ValueError("The learned model no longer shares the original world normalization")
    cameras = evaluation_cameras(dataset, initializer)
    poses = _normalized_poses(cameras["c2w"], model.normalization)
    frames = {item["image"]: item for item in load_manifest(dataset)["frames"] if item["split"] == "test"}
    render_dir = output / "renders"
    torch = cuda_dependencies()
    torch.cuda.synchronize()
    started = time.perf_counter()
    with torch.inference_mode():
        for name, pose, K in zip(cameras["names"], poses, cameras["K"]):
            frame = frames[str(name)]
            if sha256_file(dataset / str(name)) != frame["sha256"]:
                raise ValueError(f"Held-out image changed: {name}")
            rgb, _, _ = _rasterize(model.splats, torch.as_tensor(pose, device="cuda"),
                                   torch.as_tensor(K, dtype=torch.float32, device="cuda"),
                                   frame["width"], frame["height"], model.sh_degree)
            pixels = (rgb[0].clamp(0, 1).cpu().numpy() * 255).round().astype(np.uint8)
            path = render_dir / Path(str(name)).with_suffix(".png")
            path.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(pixels).save(path)
    torch.cuda.synchronize()
    record = {**cameras["alignment"], "step": model.provenance["step"], "count": len(poses),
              "checkpoint_sha256": model.checkpoint_sha256, "raw_gaussian_renders": True,
              "render_seconds": time.perf_counter() - started,
              "alignment_source_scene_sha256": receipt["original_scene_sha256"],
              "fitted_scene_sha256": receipt["scene_sha256"]}
    write_json(render_dir / "render.json", record)
    metrics = evaluate_renders(dataset, render_dir)
    if not metrics["complete"]:
        raise RuntimeError("Incomplete fixed-alignment diagnostic")
    write_json(output / "metrics.json", metrics)
    print(f"Fixed-alignment joint render PSNR {metrics['psnr']:.6f}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("dataset", "initializer", "refined", "output"):
        parser.add_argument(f"--{flag}", type=Path, required=True)
    args = parser.parse_args()
    evaluate(args.dataset, args.initializer, args.refined, args.output)


if __name__ == "__main__":
    main()

"""Shared vanilla 3DGS backend using the pinned gsplat 1.5.3 public API.

No camera or appearance optimization. Checkpoints include optimizer, strategy,
sampling and random states; save occurs AFTER a completed optimization step.
CUDA imports are lazy so CPU data tools and --help do not require torch.
"""
import argparse
from dataclasses import asdict, dataclass, fields
import json
import math
from pathlib import Path
import random
import time

import numpy as np
from PIL import Image

from p4.common import sha256_file, source_version, write_json


@dataclass
class TrainConfig:
    steps: int = 7000
    seed: int = 42
    save_steps: tuple = (500, 1000, 2000, 4000, 7000)
    sh_degree: int = 3
    sh_degree_interval: int = 1000
    ssim_lambda: float = 0.2
    refine_start: int = 500
    refine_stop: int = 6000
    refine_every: int = 100
    reset_every: int = 3000
    max_gaussians: int = 3000000
    log_every: int = 50

    def validate(self):
        if self.steps < 1 or self.sh_degree_interval < 1 or self.log_every < 1:
            raise ValueError("steps, sh_degree_interval, log_every must be positive")
        if not 0 <= self.sh_degree <= 3 or not 0 <= self.ssim_lambda <= 1:
            raise ValueError("Invalid SH degree or SSIM loss weight")
        if not 0 <= self.refine_start < self.refine_stop or self.refine_every < 1 or self.reset_every < 1:
            raise ValueError("Invalid densification schedule")
        if self.max_gaussians < 4:
            raise ValueError("max_gaussians must be >= 4")


def cuda_dependencies():
    import torch
    import gsplat
    if not torch.cuda.is_available():
        raise RuntimeError("3DGS requires an NVIDIA CUDA GPU; use scripts/setup_server.sh on the server.")
    if gsplat.__version__ != "1.5.3":
        raise RuntimeError(f"Expected gsplat 1.5.3, got {gsplat.__version__}; run server setup.")
    return torch


def ssim_loss_value(prediction, target):
    """Differentiable 11x11 Gaussian-window SSIM, sigma=1.5, valid pixels.

    Equivalent window conventions are used by the offline scikit-image metric.
    Images are NCHW float RGB in [0, 1]; no clipping in the training gradient.
    """
    import torch
    import torch.nn.functional as F
    coordinates = torch.arange(11, device=prediction.device, dtype=prediction.dtype) - 5
    kernel = torch.exp(-coordinates.square() / (2 * 1.5**2))
    kernel = kernel / kernel.sum()
    window = (kernel[:, None] * kernel[None, :]).expand(3, 1, 11, 11).contiguous()
    def blur(x):
        return F.conv2d(x, window, groups=3)
    a, b = blur(prediction), blur(target)
    va, vb = blur(prediction.square()) - a.square(), blur(target.square()) - b.square()
    covariance = blur(prediction * target) - a * b
    return (((2 * a * b + 0.01**2) * (2 * covariance + 0.03**2)) /
            ((a.square() + b.square() + 0.01**2) * (va + vb + 0.03**2))).mean()


def normalized_geometry(scene):
    """A train-camera similarity normalization; references never enter this step."""
    poses = scene["c2w"].astype(np.float32).copy()
    origin = poses[:, :3, 3].mean(axis=0)
    radius = float(np.max(np.linalg.norm(poses[:, :3, 3] - origin, axis=1)))
    if not np.isfinite(radius) or radius < 1e-8:
        raise ValueError("Camera extent is zero; cannot normalize scene")
    poses[:, :3, 3] = (poses[:, :3, 3] - origin) / radius
    points = (scene["points"].astype(np.float32) - origin) / radius
    return poses, points, {"origin": origin.tolist(), "radius": radius}


def training_fingerprint(dataset, scene_path, manifest):
    for frame in manifest["frames"]:
        if frame["split"] == "train" and sha256_file(Path(dataset) / frame["image"]) != frame.get("sha256"):
            raise ValueError(f"Prepared training image changed: {frame['image']}")
    return {"manifest_sha256": sha256_file(Path(dataset) / "manifest.json"),
            "scene_sha256": sha256_file(scene_path),
            "package_source_sha256": source_version()["package_source_sha256"]}


def _new_parameters(points, colors, degree, device):
    import torch
    from scipy.spatial import cKDTree
    distances = cKDTree(points).query(points, k=4)[0][:, 1:]
    scales = np.sqrt(np.square(distances).mean(axis=1)).clip(1e-7)
    n = len(points)
    sh = torch.zeros(n, (degree + 1)**2, 3, device=device)
    sh[:, 0] = (torch.as_tensor(colors / 255.0, dtype=torch.float32, device=device) - 0.5) / 0.28209479177387814
    values = {
        "means": torch.as_tensor(points, dtype=torch.float32, device=device),
        "scales": torch.as_tensor(np.log(scales), dtype=torch.float32, device=device)[:, None].repeat(1, 3),
        "quats": torch.rand(n, 4, device=device),
        "opacities": torch.full((n,), math.log(0.1 / 0.9), device=device),
        "sh0": sh[:, :1], "shN": sh[:, 1:]
    }
    return torch.nn.ParameterDict({k: torch.nn.Parameter(v.contiguous()) for k, v in values.items()})


def _optimizers(splats):
    import torch
    rates = dict(means=1.6e-4 * 1.1, scales=5e-3, quats=1e-3, opacities=5e-2, sh0=2.5e-3, shN=2.5e-3 / 20)
    return {k: torch.optim.Adam([{"params": p, "lr": rates[k], "name": k}], eps=1e-15)
            for k, p in splats.items()}


def _rasterize(splats, c2w, K, width, height, degree):
    import torch
    from gsplat import rasterization
    return rasterization(means=splats["means"], quats=splats["quats"],
                         scales=splats["scales"].exp(), opacities=splats["opacities"].sigmoid(),
                         colors=torch.cat([splats["sh0"], splats["shN"]], dim=1),
                         viewmats=torch.linalg.inv(c2w[None]), Ks=K[None],
                         width=width, height=height, sh_degree=degree, packed=False,
                         near_plane=0.01, far_plane=1e10, rasterize_mode="classic")


def train(dataset, scene_path, output, config=None, resume=None, stop_after=None):
    from gsplat.strategy import DefaultStrategy
    from p4.data import load_manifest, load_scene
    cfg = config or TrainConfig()
    cfg.validate()
    torch = cuda_dependencies()
    dataset, scene_path, output = Path(dataset), Path(scene_path), Path(output)
    manifest = load_manifest(dataset)
    scene = load_scene(scene_path)
    expected = {f["image"] for f in manifest["frames"] if f["split"] == "train"}
    if set(scene["names"].tolist()) != expected:
        raise ValueError("Training requires all and only train images to be registered; partial initialization is a failure.")
    if len(scene["points"]) < 4:
        raise ValueError("Need at least four initialization points")
    output.mkdir(parents=True, exist_ok=True)
    if not resume and (output / "config.json").exists():
        raise FileExistsError(f"Existing training output: {output}; explicitly resume or use another run directory")
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed_all(cfg.seed)
    device = "cuda"
    poses, points, normalization = normalized_geometry(scene)
    fingerprint = training_fingerprint(dataset, scene_path, manifest)
    if resume:
        saved = torch.load(resume, map_location=device, weights_only=False)  # Only trust your own checkpoints.
        if saved["schema_version"] != 1 or saved["fingerprint"] != fingerprint or saved["config"] != asdict(cfg):
            raise ValueError("Resume mismatch: checkpoint schema, dataset, initialization or frozen config changed")
        splats = torch.nn.ParameterDict({k: torch.nn.Parameter(v) for k, v in saved["splats"].items()})
        normalization = saved["normalization"]
    else:
        saved = None
        splats = _new_parameters(points, scene["colors"], cfg.sh_degree, device)
    optimizers = _optimizers(splats)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizers["means"], gamma=0.01 ** (1 / cfg.steps))
    strategy = DefaultStrategy(refine_start_iter=cfg.refine_start, refine_stop_iter=cfg.refine_stop,
                               refine_every=cfg.refine_every, reset_every=cfg.reset_every, verbose=False)
    strategy.check_sanity(splats, optimizers)
    state = strategy.initialize_state(scene_scale=1.1)
    sampler = np.random.default_rng(cfg.seed)
    order, cursor, start, elapsed = [], 0, 0, 0.0
    if saved:
        for k, opt in optimizers.items():
            opt.load_state_dict(saved["optimizers"][k])
        scheduler.load_state_dict(saved["scheduler"])
        state = saved["strategy"]
        sampler.bit_generator.state = saved["sampler"]
        order, cursor, start, elapsed = saved["order"], saved["cursor"], saved["step"], saved["training_seconds"]
        random.setstate(saved["python_rng"])
        np.random.set_state(saved["numpy_rng"])
        torch.set_rng_state(saved["torch_rng"].cpu())
        torch.cuda.set_rng_state_all([s.cpu() for s in saved["cuda_rng"]])
        log_path = output / "train.jsonl"
        if log_path.exists():
            # Discard log entries beyond the explicitly restored state. Keep a
            # record of the abandoned continuation instead of duplicating steps.
            lines = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines() if line]
            if any(row["step"] > start for row in lines):
                archive = output / f"train_before_resume_{time.time_ns()}.jsonl"
                log_path.rename(archive)
                log_path.write_text("".join(json.dumps(row) + "\n" for row in lines if row["step"] <= start), encoding="utf-8")
    stop = cfg.steps if stop_after is None else min(cfg.steps, int(stop_after))
    if stop < start:
        raise ValueError("stop_after is before checkpoint step")
    write_json(output / "config.json", asdict(cfg))
    write_json(output / "provenance.json", {**source_version(), **fingerprint, "normalization": normalization,
               "torch": torch.__version__, "cuda": torch.version.cuda, "gsplat": "1.5.3",
               "gpu": torch.cuda.get_device_name(), "pose_optimization": False, "appearance_optimization": False,
               "strategy_patch": "gsplat 1.5.3 opacity-reset boolean precedence fix; reset after post-backward at zero-based steps 3000,... below refine_stop",
               "initial_gaussians": len(points), "seed": cfg.seed})
    torch.cuda.reset_peak_memory_stats()
    poses_t = torch.as_tensor(poses, device=device)
    Ks = torch.as_tensor(scene["K"], dtype=torch.float32, device=device)
    torch.cuda.synchronize()
    tick = time.perf_counter()
    for step in range(start, stop):
        if cursor >= len(order):
            order, cursor = sampler.permutation(len(poses)).tolist(), 0
        index = order[cursor]
        cursor += 1
        image = np.asarray(Image.open(dataset / str(scene["names"][index])).convert("RGB"), dtype=np.float32) / 255
        height, width = image.shape[:2]
        if min(height, width) < 11:
            raise ValueError("Training images must be at least 11 pixels on both axes for SSIM")
        pixels = torch.as_tensor(image, device=device)[None]
        degree = min(step // cfg.sh_degree_interval, cfg.sh_degree)
        rendered, _, info = _rasterize(splats, poses_t[index], Ks[index], width, height, degree)
        strategy.step_pre_backward(splats, optimizers, state, step, info)
        l1 = (rendered - pixels).abs().mean()
        ssim = ssim_loss_value(rendered.permute(0, 3, 1, 2), pixels.permute(0, 3, 1, 2))
        loss = (1 - cfg.ssim_lambda) * l1 + cfg.ssim_lambda * (1 - ssim)
        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite loss at step {step + 1}")
        loss.backward()
        for opt in optimizers.values():
            opt.step()
            opt.zero_grad(set_to_none=True)
        scheduler.step()
        strategy.step_post_backward(splats, optimizers, state, step, info, packed=False)
        # Pinned upstream default.py uses `== 0 & step > 0`, a chained
        # comparison that is always false. Apply the intended shared 3DGS reset.
        if step < cfg.refine_stop and step > 0 and step % cfg.reset_every == 0:
            from gsplat.strategy.ops import reset_opa
            reset_opa(params=splats, optimizers=optimizers, state=state, value=strategy.prune_opa * 2)
        if len(splats["means"]) > cfg.max_gaussians:
            raise RuntimeError("Gaussian resource ceiling exceeded; run failed. Raise the SAME ceiling for all methods before formal runs.")
        completed = step + 1
        if completed % cfg.log_every == 0 or completed == stop:
            torch.cuda.synchronize()
            row = {"step": completed, "loss": loss.item(), "l1": l1.item(), "ssim": ssim.item(),
                   "num_gaussians": len(splats["means"]), "training_seconds": elapsed + time.perf_counter() - tick,
                   "peak_cuda_gib": torch.cuda.max_memory_allocated() / 1024**3}
            with (output / "train.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(row) + "\n")
            print(json.dumps(row), flush=True)
        if completed in cfg.save_steps or completed == stop:
            torch.cuda.synchronize()
            payload = {"schema_version": 1, "step": completed, "config": asdict(cfg), "fingerprint": fingerprint,
                       "splats": splats.state_dict(), "optimizers": {k: v.state_dict() for k, v in optimizers.items()},
                       "scheduler": scheduler.state_dict(), "strategy": state, "normalization": normalization,
                       "sampler": sampler.bit_generator.state, "order": order, "cursor": cursor,
                       "python_rng": random.getstate(), "numpy_rng": np.random.get_state(),
                       "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state_all(),
                       "training_seconds": elapsed + time.perf_counter() - tick,
                       "sh_degree_active": min(step // cfg.sh_degree_interval, cfg.sh_degree)}
            checkpoint = output / "checkpoints" / f"step_{completed:06d}.pt"
            checkpoint.parent.mkdir(exist_ok=True)
            temp = checkpoint.with_suffix(".tmp")
            torch.save(payload, temp)
            temp.replace(checkpoint)
    return output


def render(dataset, scene_path, checkpoint, output):
    """Evaluate with ONE train-derived global alignment; no per-test pose fitting."""
    from p4.evaluate import evaluation_cameras
    from p4.data import load_manifest
    torch = cuda_dependencies()
    cameras = evaluation_cameras(dataset, scene_path)
    names, poses, Ks = cameras["names"], cameras["c2w"].copy(), cameras["K"]
    metadata = cameras["alignment"]
    saved = torch.load(checkpoint, map_location="cuda", weights_only=False)
    manifest = load_manifest(dataset)
    if saved["fingerprint"] != training_fingerprint(dataset, scene_path, manifest):
        raise ValueError("Render checkpoint does not match dataset and initializer")
    origin = np.array(saved["normalization"]["origin"])
    radius = saved["normalization"]["radius"]
    poses[:, :3, 3] = (poses[:, :3, 3] - origin) / radius
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    torch.cuda.synchronize()
    tick = time.perf_counter()
    with torch.no_grad():
        for name, pose, K in zip(names, poses, Ks):
            with Image.open(Path(dataset) / str(name)) as image:
                width, height = image.size
            rgb, _, _ = _rasterize(saved["splats"], torch.as_tensor(pose, dtype=torch.float32, device="cuda"),
                                   torch.as_tensor(K, dtype=torch.float32, device="cuda"), width, height,
                                   saved["sh_degree_active"])
            pixels = (rgb[0].clamp(0, 1).cpu().numpy() * 255).round().astype(np.uint8)
            destination = output / Path(str(name)).with_suffix(".png")
            destination.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(pixels).save(destination)
    torch.cuda.synchronize()
    write_json(output / "render.json", {**metadata, "step": saved["step"], "checkpoint_sha256": sha256_file(checkpoint),
               "render_seconds": time.perf_counter() - tick, "count": len(names), "raw_gaussian_renders": True})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest="command", required=True)
    training = subs.add_parser("train")
    training.add_argument("--dataset", required=True)
    training.add_argument("--scene", required=True, help="Initializer scene.npz")
    training.add_argument("--output", required=True)
    training.add_argument("--config", default="configs/part1.json")
    training.add_argument("--resume")
    training.add_argument("--stop-after", type=int, help="Pilot / interruption test only; preserves original schedule")
    rendering = subs.add_parser("render")
    for flag in ["dataset", "scene", "checkpoint", "output"]:
        rendering.add_argument("--" + flag, required=True)
    args = parser.parse_args()
    if args.command == "train":
        raw = json.loads(Path(args.config).read_text(encoding="utf-8"))
        cfg = TrainConfig(**raw.get("training", raw))
        train(args.dataset, args.scene, args.output, cfg, args.resume, args.stop_after)
    else:
        render(args.dataset, args.scene, args.checkpoint, args.output)


if __name__ == "__main__":
    main()

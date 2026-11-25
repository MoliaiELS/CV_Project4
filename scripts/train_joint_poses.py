"""Continue a standard Gaussian fit while refining train-camera extrinsics."""

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import random
import tempfile
import time

import numpy as np
from PIL import Image

from p4.common import sha256_file, source_version, write_json
from p4.data import load_manifest, load_scene
from p4.train import TrainConfig, _optimizers, _rasterize, cuda_dependencies, ssim_loss_value, training_fingerprint


@dataclass(frozen=True)
class PoseConfig:
    start: int = 1000
    stop: int = 6000
    learning_rate: float = 2e-4
    translation_prior: float = 0.05
    rotation_prior: float = 0.01

    def validate(self, steps):
        values = (self.learning_rate, self.translation_prior, self.rotation_prior)
        if not 0 < self.start < self.stop < steps:
            raise ValueError("Pose refinement must follow warmup and precede final fixed-pose training")
        if not np.isfinite(values).all() or min(values) <= 0:
            raise ValueError("Pose learning rate and prior weights must be finite and positive")


def apply_pose_delta(c2w, delta):
    """Compose a right-local rotation and translation with one or more base poses."""
    import torch

    x, y, z = delta[..., :3].unbind(-1)
    zero = torch.zeros_like(x)
    skew = torch.stack((zero, -z, y, z, zero, -x, -y, x, zero), dim=-1)
    rotation = torch.matrix_exp(skew.reshape(*x.shape, 3, 3))
    base_rotation = c2w[..., :3, :3]
    translation = c2w[..., :3, 3:4] + base_rotation @ delta[..., 3:, None]
    upper = torch.cat((base_rotation @ rotation, translation), dim=-1)
    return torch.cat((upper, c2w[..., 3:4, :]), dim=-2)


def _normalization_poses(poses, normalization):
    poses = np.array(poses, dtype=np.float32, copy=True)
    poses[:, :3, 3] = (poses[:, :3, 3] - np.asarray(normalization["origin"], np.float32)) / normalization["radius"]
    return poses


def _write_checkpoint(torch, path, payload):
    temporary = path.with_suffix(".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def _restore_adam(optimizers, saved):
    for name, optimizer in optimizers.items():
        optimizer.load_state_dict(saved["optimizers"][name])
        for state in optimizer.state.values():
            if "step" in state:
                state["step"] = state["step"].cpu()


def _load_base(torch, path, fingerprint, expected_sha256=None):
    if expected_sha256 is not None and sha256_file(path) != expected_sha256:
        raise ValueError(f"Checkpoint checksum mismatch: {path}")
    saved = torch.load(path, map_location="cpu", weights_only=False)
    if saved["schema_version"] != 1 or saved["fingerprint"] != fingerprint:
        raise ValueError("Checkpoint is incompatible with the original dataset, initializer, or backend")
    return saved


def _poses_for_export(scene, deltas, normalization):
    """Return refined cameras in the original initializer's unnormalized frame."""
    import torch

    base = torch.from_numpy(_normalization_poses(scene["c2w"], normalization)).double()
    with torch.no_grad():
        poses = apply_pose_delta(base, deltas.cpu().double()).numpy()
    poses[:, :3, 3] = poses[:, :3, 3] * normalization["radius"] + np.asarray(normalization["origin"])
    # Preserve the original cameras exactly when their delta was frozen at zero.
    untouched = (deltas.cpu().numpy() == 0).all(axis=1)
    poses[untouched] = scene["c2w"][untouched]
    return poses


def train_joint(dataset, initializer, warmup, output, pose_config=None, *, resume=None, stop_after=None):
    """Warm up with p4.train, then continue its state with differentiable train poses."""
    from gsplat.strategy import DefaultStrategy

    torch = cuda_dependencies()
    dataset, initializer, warmup, output = map(Path, (dataset, initializer, warmup, output))
    pose_config = pose_config or PoseConfig()
    manifest = load_manifest(dataset)
    scene = load_scene(initializer)
    names = [str(name) for name in scene["names"]]
    if set(names) != {row["image"] for row in manifest["frames"] if row["split"] == "train"}:
        raise ValueError("Joint training requires all and only registered train frames")
    fingerprint = training_fingerprint(dataset, initializer, manifest)
    warmup_sha = sha256_file(warmup)
    saved = _load_base(torch, Path(resume) if resume else warmup, fingerprint, None if resume else warmup_sha)
    cfg = TrainConfig(**saved["config"])
    cfg.validate()
    pose_config.validate(cfg.steps)
    protocol = {"pose_config": asdict(pose_config), "warmup_sha256": warmup_sha,
                "anchor": names[0], "intrinsics_optimized": False, "reference_used": False}
    if resume:
        if any(saved["joint"].get(key) != value for key, value in protocol.items()):
            raise ValueError("Joint resume does not match the original pose protocol or warmup")
    elif saved["step"] != pose_config.start or "joint" in saved:
        raise ValueError("Provide the vanilla checkpoint at the exact joint warmup boundary")
    start = saved["step"]
    stop = cfg.steps if stop_after is None else min(cfg.steps, stop_after)
    if stop <= start:
        raise ValueError("Requested stop must be after the checkpoint")
    output.mkdir(parents=True, exist_ok=True)
    log_path = output / "train.jsonl"
    if not resume and (output / "protocol.json").exists():
        raise FileExistsError(f"Existing joint run {output}; explicitly resume")
    if resume and log_path.exists():
        rows = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines() if line]
        if any(row["step"] > start for row in rows):
            log_path.rename(output / f"train_before_resume_{time.time_ns()}.jsonl")
            log_path.write_text("".join(json.dumps(row) + "\n" for row in rows if row["step"] <= start),
                                encoding="utf-8")
    write_json(output / "protocol.json", {**source_version(), **protocol, **fingerprint,
               "joint_script_sha256": sha256_file(__file__), "training_config": asdict(cfg),
               "gpu": torch.cuda.get_device_name(), "torch": torch.__version__})

    device = "cuda"
    normalization = saved["normalization"]
    base_poses = torch.as_tensor(_normalization_poses(scene["c2w"], normalization), device=device)
    intrinsics = torch.as_tensor(scene["K"], dtype=torch.float32, device=device)
    splats = torch.nn.ParameterDict({name: torch.nn.Parameter(value.to(device))
                                   for name, value in saved["splats"].items()})
    optimizers = _optimizers(splats)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizers["means"], gamma=0.01 ** (1 / cfg.steps))
    _restore_adam(optimizers, saved)
    scheduler.load_state_dict(saved["scheduler"])
    strategy = DefaultStrategy(refine_start_iter=cfg.refine_start, refine_stop_iter=cfg.refine_stop,
                               refine_every=cfg.refine_every, reset_every=cfg.reset_every, verbose=False)
    strategy.check_sanity(splats, optimizers)
    state = {key: value.to(device) if torch.is_tensor(value) else value for key, value in saved["strategy"].items()}
    deltas = torch.nn.Parameter(torch.zeros((len(names), 6), device=device))
    pose_optimizer = torch.optim.Adam([deltas], lr=pose_config.learning_rate)
    sampler = np.random.default_rng(cfg.seed)
    sampler.bit_generator.state = saved["sampler"]
    order, cursor, elapsed = saved["order"], saved["cursor"], saved["training_seconds"]
    random.setstate(saved["python_rng"])
    np.random.set_state(saved["numpy_rng"])
    torch.set_rng_state(saved["torch_rng"])
    torch.cuda.set_rng_state_all(saved["cuda_rng"])
    pose_updates = 0
    if resume:
        with torch.no_grad():
            deltas.copy_(saved["joint"]["deltas"])
        pose_optimizer.load_state_dict(saved["joint"]["optimizer"])
        pose_updates = saved["joint"]["pose_updates"]
    del saved
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    started = time.perf_counter()
    checkpoint = None
    frozen_poses = None
    for step in range(start, stop):
        if cursor >= len(order):
            order, cursor = sampler.permutation(len(names)).tolist(), 0
        index = order[cursor]
        cursor += 1
        with Image.open(dataset / names[index]) as image:
            pixels = np.asarray(image.convert("RGB"), dtype=np.float32) / 255
        height, width = pixels.shape[:2]
        if min(height, width) < 11:
            raise ValueError("Training images must be at least 11 pixels on both axes for SSIM")
        target = torch.as_tensor(pixels, device=device)[None]
        degree = min(step // cfg.sh_degree_interval, cfg.sh_degree)
        optimize_pose = index != 0 and step < pose_config.stop
        delta = deltas[index]
        if step >= pose_config.stop and frozen_poses is None:
            frozen_poses = apply_pose_delta(base_poses, deltas.detach())
            frozen_poses[0] = base_poses[0]
        if frozen_poses is not None:
            pose = frozen_poses[index]
        else:
            pose = base_poses[index] if index == 0 else apply_pose_delta(base_poses[index], delta)
        rgb, _, info = _rasterize(splats, pose, intrinsics[index], width, height, degree)
        strategy.step_pre_backward(splats, optimizers, state, step, info)
        l1 = (rgb - target).abs().mean()
        ssim = ssim_loss_value(rgb.permute(0, 3, 1, 2), target.permute(0, 3, 1, 2))
        loss = (1 - cfg.ssim_lambda) * l1 + cfg.ssim_lambda * (1 - ssim)
        if optimize_pose:
            loss = loss + pose_config.rotation_prior * delta[:3].square().sum()
            loss = loss + pose_config.translation_prior * delta[3:].square().sum()
        if not torch.isfinite(loss):
            raise RuntimeError(f"Nonfinite loss at step {step + 1}")
        loss.backward()
        if optimize_pose:
            if deltas.grad is None or not torch.isfinite(deltas.grad).all():
                raise RuntimeError(f"Missing or nonfinite pose gradient at step {step + 1}")
            pose_optimizer.step()
            pose_optimizer.zero_grad(set_to_none=True)
            pose_updates += 1
        for optimizer in optimizers.values():
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        scheduler.step()
        strategy.step_post_backward(splats, optimizers, state, step, info, packed=False)
        if 0 < step < cfg.refine_stop and step % cfg.reset_every == 0:
            from gsplat.strategy.ops import reset_opa
            reset_opa(params=splats, optimizers=optimizers, state=state, value=strategy.prune_opa * 2)
        if len(splats["means"]) > cfg.max_gaussians:
            raise RuntimeError("Shared Gaussian resource ceiling exceeded")
        completed = step + 1
        if completed % cfg.log_every == 0 or completed == stop:
            torch.cuda.synchronize()
            row = {"step": completed, "real_view": names[index], "loss": loss.item(), "l1": l1.item(),
                   "ssim": ssim.item(), "pose_updates": pose_updates, "num_gaussians": len(splats["means"]),
                   "training_seconds": elapsed + time.perf_counter() - started,
                   "peak_cuda_gib": torch.cuda.max_memory_allocated() / 1024**3}
            with log_path.open("a", encoding="utf-8") as stream:
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
                       "training_seconds": elapsed + time.perf_counter() - started, "sh_degree_active": degree,
                       "joint": {**protocol, "deltas": deltas.detach(), "optimizer": pose_optimizer.state_dict(),
                                 "pose_updates": pose_updates}}
            checkpoint = output / "checkpoints" / f"step_{completed:06d}.pt"
            checkpoint.parent.mkdir(exist_ok=True)
            _write_checkpoint(torch, checkpoint, payload)
    return checkpoint


def export_joint(dataset, initializer, checkpoint, output):
    """Export the frozen refined train cameras and a matching rendering checkpoint."""
    import torch

    dataset, initializer, checkpoint, output = map(Path, (dataset, initializer, checkpoint, output))
    if output.exists():
        raise FileExistsError(f"Export into a fresh directory: {output}")
    scene = load_scene(initializer)
    manifest = load_manifest(dataset)
    fingerprint = training_fingerprint(dataset, initializer, manifest)
    saved = _load_base(torch, checkpoint, fingerprint)
    joint = saved.pop("joint")
    if saved["step"] != saved["config"]["steps"] or joint["pose_updates"] == 0:
        raise ValueError("Export requires a completed joint fit with pose updates")
    deltas = joint["deltas"]
    if not torch.isfinite(deltas).all() or deltas[0].count_nonzero() or deltas.count_nonzero() == 0:
        raise ValueError("Expected finite pose updates and the unchanged gauge anchor")
    scene["c2w"] = _poses_for_export(scene, deltas, saved["normalization"])
    rotations = np.rad2deg(np.linalg.norm(deltas[:, :3].numpy(), axis=1))
    translations = np.linalg.norm(deltas[:, 3:].numpy(), axis=1)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".joint-export-", dir=output.parent) as temporary:
        staging = Path(temporary) / "export"
        staging.mkdir()
        scene_path, render_checkpoint = staging / "scene.npz", staging / "checkpoint.pt"
        np.savez_compressed(scene_path, **scene)
        load_scene(scene_path)
        saved["fingerprint"] = training_fingerprint(dataset, scene_path, manifest)
        _write_checkpoint(torch, render_checkpoint, saved)
        write_json(staging / "export.json", {**source_version(), **saved["fingerprint"],
                   "checkpoint_sha256": sha256_file(render_checkpoint),
                   "joint_checkpoint_sha256": sha256_file(checkpoint),
                   "original_scene_sha256": fingerprint["scene_sha256"], "warmup_sha256": joint["warmup_sha256"],
                   "step": saved["step"], "pose_updates": joint["pose_updates"], "anchor": joint["anchor"],
                   "max_rotation_delta_degrees": float(rotations.max()),
                   "max_translation_delta_normalized": float(translations.max()),
                   "intrinsics_optimized": False, "reference_used_in_training": False})
        staging.rename(output)
    return output / "scene.npz", output / "checkpoint.pt"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    train = commands.add_parser("train")
    export = commands.add_parser("export")
    for sub in (train, export):
        for flag in ("dataset", "initializer", "output"):
            sub.add_argument(f"--{flag}", type=Path, required=True)
    train.add_argument("--warmup", type=Path, required=True)
    train.add_argument("--resume", type=Path)
    train.add_argument("--stop-after", type=int)
    train.add_argument("--pose-start", type=int, default=PoseConfig.start)
    train.add_argument("--pose-stop", type=int, default=PoseConfig.stop)
    train.add_argument("--pose-lr", type=float, default=PoseConfig.learning_rate)
    export.add_argument("--checkpoint", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "train":
        config = PoseConfig(args.pose_start, args.pose_stop, args.pose_lr)
        print(train_joint(args.dataset, args.initializer, args.warmup, args.output, config,
                          resume=args.resume, stop_after=args.stop_after))
    else:
        print(*export_joint(args.dataset, args.initializer, args.checkpoint, args.output))


if __name__ == "__main__":
    main()

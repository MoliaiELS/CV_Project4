"""Train sparse 3DGS with cached VGGT depth supervision on the real training views."""

import argparse
from dataclasses import asdict
from functools import lru_cache
import json
from pathlib import Path
import random
import time

import numpy as np
from PIL import Image

from p4.common import sha256_file, source_version, write_json
from p4.data import load_manifest, load_scene
from p4.train import (TrainConfig, _new_parameters, _optimizers, _rasterize, cuda_dependencies,
                      normalized_geometry, ssim_loss_value, training_fingerprint)
from scripts.part3_fusion import _CameraSampler, _normalized_poses, _save_checkpoint, _train_log_start


LEGACY_DEPTH_ADAPTER_SHA256 = "f750c958a947bc7465054b9a9eaef1cfd8e54f4d946c6520692e576598a7d0d3"


def _load_depth_maps(path, receipt_path, manifest, scene, fingerprint):
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    digest = sha256_file(path)
    names = scene["names"].tolist()
    if (receipt.get("depth_maps_sha256") != digest
            or receipt.get("manifest_sha256") != fingerprint["manifest_sha256"]
            or receipt.get("camera_initializer_sha256") != fingerprint["scene_sha256"]
            or receipt.get("training_ids") != names
            or receipt.get("split_hash") != manifest["split_hash"]
            or receipt.get("reference_data_used") is not False
            or receipt.get("depth_units") != "native VGGT camera-axis Z; not normalized or fit"):
        raise ValueError("Depth receipt must authenticate the exact train-only dataset and camera-Z maps")
    with np.load(path, allow_pickle=False) as archive:
        if (archive["split_hash"].item() != manifest["split_hash"]
                or archive["camera_initializer_sha256"].item() != fingerprint["scene_sha256"]
                or archive["names"].tolist() != names):
            raise ValueError("Depth cache does not match the ordered sparse training cameras")
        depth, confidence = archive["depth"], archive["confidence"]
    frames = {frame["image"]: frame for frame in manifest["frames"] if frame["split"] == "train"}
    shape = (len(names), *receipt["prepared_hw"])
    if (depth.ndim != 3 or depth.shape != shape or confidence.shape != shape
            or any(depth.shape[1:] != (frames[name]["height"], frames[name]["width"]) for name in names)):
        raise ValueError("Cached depth maps must use the real canonical RGB grid")
    return depth, confidence, digest


def _depth_target(depth, confidence, radius, max_edge):
    """Use nearest native Z measurements and confidence ranks in a reduced grid."""
    height, width = depth.shape
    scale = min(1., max_edge / max(height, width))
    out_h, out_w = max(1, round(height * scale)), max(1, round(width * scale))
    y = ((np.arange(out_h) + 0.5) * height / out_h).astype(int)
    x = ((np.arange(out_w) + 0.5) * width / out_w).astype(int)
    z = np.asarray(depth[np.ix_(y, x)], dtype=np.float32) / radius
    conf = np.asarray(confidence[np.ix_(y, x)], dtype=np.float32)
    valid = np.isfinite(z) & (z > .01) & np.isfinite(conf) & (conf > 0)
    if not valid.any():
        raise ValueError("A training camera has no valid above-near-plane depth")
    weak, strong = np.quantile(conf[valid], (0.30, 0.70))
    if strong > weak:
        weights = np.where(valid, np.clip((conf - weak) / (strong - weak), 0., 1.), 0.)
    else:
        weights = (valid & (conf >= strong)).astype(np.float32)
    log_z = np.log(np.where(valid, z, 1.))
    return log_z, weights.astype(np.float32), np.diag([out_w / width, out_h / height, 1.])


def depth_losses(predicted, alpha, target, confidence, patch=8):
    """Robust log-Z error and patch-relative log-Z error, ignoring weak support."""
    import torch
    from torch.nn import functional as F

    finite = torch.isfinite(predicted) & (predicted > .01)
    valid = finite & (alpha.detach() >= .15) & (confidence > 0)
    weights = confidence * valid
    log_z = torch.where(valid, predicted, torch.ones_like(predicted)).log()
    residual = torch.where(valid, log_z - target, 0.)
    global_error = F.smooth_l1_loss(residual, torch.zeros_like(residual), reduction="none", beta=.12)
    global_loss = (weights * global_error).sum() / weights.sum().clamp_min(.05 * weights.numel())
    if min(residual.shape) < patch:
        return global_loss, global_loss * 0, valid.float().mean()
    blocks = residual.unfold(0, patch, patch).unfold(1, patch, patch)
    block_weights = weights.unfold(0, patch, patch).unfold(1, patch, patch)
    mass = block_weights.sum(dim=(-2, -1), keepdim=True)
    centered = blocks - (block_weights * blocks).sum(dim=(-2, -1), keepdim=True) / mass.clamp_min(1.)
    local_error = F.smooth_l1_loss(centered, torch.zeros_like(centered), reduction="none", beta=.06)
    patch_weights = block_weights * (mass >= .20 * patch * patch)
    local_loss = (local_error * patch_weights).sum() / patch_weights.sum().clamp_min(.05 * patch_weights.numel())
    return global_loss, local_loss, valid.float().mean()


def _render_depth(splats, pose, K, width, height):
    import torch
    from gsplat import rasterization

    means = splats["means"]
    depth, alpha, _ = rasterization(
        means=means, quats=splats["quats"].detach(), scales=splats["scales"].detach().exp(),
        opacities=splats["opacities"].detach().sigmoid(), colors=means.new_zeros((len(means), 1)),
        viewmats=torch.linalg.inv(pose[None]), Ks=K[None], width=width, height=height,
        render_mode="ED", packed=False, near_plane=.01, far_plane=1e10, rasterize_mode="classic",
    )
    return depth[0, ..., 0], alpha[0, ..., 0]


def _rgb(path):
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.float32) / 255.


def train(dataset, scene_path, depth_path, receipt_path, output, cfg, *, depth_weight=.15, depth_edge=192,
          depth_start=200, teacher_weight=0., teacher_start=400, teacher_ramp=600, resume=None, stop_after=None):
    cfg.validate()
    if not np.isfinite(depth_weight) or depth_weight < 0 or depth_edge < 8 or depth_start < 0:
        raise ValueError("Invalid depth weight, maximum render dimension, or warm-up length")
    if not np.isfinite(teacher_weight) or teacher_weight < 0 or teacher_start < 0 or teacher_ramp < 1:
        raise ValueError("Invalid warp teacher weight, start, or ramp length")
    dataset, scene_path, depth_path, receipt_path, output = map(Path, (dataset, scene_path, depth_path, receipt_path, output))
    manifest, scene = load_manifest(dataset), load_scene(scene_path)
    allowed = {frame["image"] for frame in manifest["frames"] if frame["split"] == "train"}
    if set(scene["names"].tolist()) != allowed or len(scene["points"]) < 4:
        raise ValueError("Initialize at least four Gaussians from exactly the sparse training cameras")
    fingerprint = training_fingerprint(dataset, scene_path, manifest)
    depth, confidence, depth_sha = _load_depth_maps(depth_path, receipt_path, manifest, scene, fingerprint)
    poses, points, normalization = normalized_geometry(scene)
    recipe = {"depth_weight": depth_weight, "depth_start": depth_start, "depth_max_edge": depth_edge,
              "depth_maps_sha256": depth_sha, "depth_receipt_sha256": sha256_file(receipt_path),
              "adapter_sha256": sha256_file(__file__), "local_weight": .25, "depth_gradients": "means only"}
    teacher_views, teacher_record = [], None
    if teacher_weight:
        from scripts import sparse_warp_teacher
        from scripts.part3_views import training_cameras

        teacher_views, teacher_record = sparse_warp_teacher.build_warp_teacher(
            dataset, training_cameras(dataset, scene_path), scene["names"].tolist(), depth, confidence,
            normalization["radius"])
        warp_teacher_loss = sparse_warp_teacher.warp_teacher_loss
        recipe["warp_teacher"] = {"weight": teacher_weight, "start": teacher_start, "ramp": teacher_ramp,
                                  "teacher_sha256": teacher_record["teacher_sha256"], "views": len(teacher_views),
                                  "helper_sha256": sha256_file(sparse_warp_teacher.__file__)}
    checkpoint_config = {**asdict(cfg), "depth_supervision": recipe}
    stop = cfg.steps if stop_after is None else min(cfg.steps, stop_after)
    if stop < 1 or (resume is None and (output / "config.json").exists()):
        raise ValueError("Use a positive stop step and a fresh run directory, or resume a trusted checkpoint")

    torch = cuda_dependencies()
    from gsplat.strategy import DefaultStrategy
    from gsplat.strategy.ops import reset_opa

    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed(cfg.seed)
    saved = torch.load(resume, map_location="cpu", weights_only=False) if resume else None
    accepted_configs = [checkpoint_config]
    if not teacher_weight:
        accepted_configs.append({**checkpoint_config, "depth_supervision": {
            **recipe, "adapter_sha256": LEGACY_DEPTH_ADAPTER_SHA256}})
    if saved and (saved["schema_version"] != 1 or saved["fingerprint"] != fingerprint
                  or saved["config"] not in accepted_configs or saved["step"] > stop):
        raise ValueError("Resume mismatch: training protocol, inputs, implementation, or stop step differs")
    splats = (_new_parameters(points, scene["colors"], cfg.sh_degree, "cuda") if saved is None else
              torch.nn.ParameterDict({key: torch.nn.Parameter(value.cuda()) for key, value in saved["splats"].items()}))
    optimizers = _optimizers(splats)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizers["means"], gamma=.01 ** (1 / cfg.steps))
    strategy = DefaultStrategy(refine_start_iter=cfg.refine_start, refine_stop_iter=cfg.refine_stop,
                               refine_every=cfg.refine_every, reset_every=cfg.reset_every, verbose=False)
    strategy.check_sanity(splats, optimizers)
    state = strategy.initialize_state(scene_scale=1.1)
    sampler = _CameraSampler(len(poses), seed=cfg.seed)
    teacher_sampler = None
    if teacher_views:
        teacher_sampler = _CameraSampler(len(teacher_views), saved["teacher_sampler"] if saved else None,
                                         seed=cfg.seed + 1000)
    start, elapsed = 0, 0.
    if saved:
        for key, optimizer in optimizers.items():
            optimizer.load_state_dict(saved["optimizers"][key])
        scheduler.load_state_dict(saved["scheduler"])
        state = {key: value.cuda() if torch.is_tensor(value) else value for key, value in saved["strategy"].items()}
        sampler = _CameraSampler(len(poses), {"rng": saved["sampler"], "order": saved["order"], "cursor": saved["cursor"]})
        start, elapsed = saved["step"], saved["training_seconds"]
        random.setstate(saved["python_rng"])
        np.random.set_state(saved["numpy_rng"])
        torch.set_rng_state(saved["torch_rng"])
        torch.cuda.set_rng_state(saved["cuda_rng"][0])
    del saved

    targets = [_depth_target(z, c, normalization["radius"], depth_edge) for z, c in zip(depth, confidence)]
    depth_Ks = torch.as_tensor(np.stack([target[2] @ K for target, K in zip(targets, scene["K"])]),
                               dtype=torch.float32, device="cuda")
    targets = [(torch.as_tensor(z, device="cuda"), torch.as_tensor(c, device="cuda")) for z, c, _ in targets]
    poses = torch.as_tensor(poses, device="cuda")
    Ks = torch.as_tensor(scene["K"], dtype=torch.float32, device="cuda")
    teacher_tensors = []
    for view in teacher_views:
        teacher_tensors.append((torch.as_tensor(_normalized_poses(view.camera.c2w, normalization), device="cuda"),
                                torch.as_tensor(view.camera.K, dtype=torch.float32, device="cuda"),
                                torch.as_tensor(view.rgb, device="cuda"),
                                torch.as_tensor(view.confidence, device="cuda")))
    training_rgb = lru_cache(maxsize=24)(_rgb)
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "config.json", checkpoint_config)
    write_json(output / "provenance.json", {**source_version(), **fingerprint, **recipe, "normalization": normalization,
               "train_only": True, "torch": torch.__version__, "gsplat": "1.5.3", "gpu": torch.cuda.get_device_name()})
    if teacher_record is not None:
        write_json(output / "warp_teacher.json", {**teacher_record, **recipe["warp_teacher"]})
    log_path = output / "train.jsonl"
    _train_log_start(log_path, start)
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    tick = time.perf_counter()
    for step in range(start, stop):
        index = sampler.next()
        target = training_rgb(dataset / str(scene["names"][index]))
        height, width = target.shape[:2]
        if min(height, width) < 11:
            raise ValueError("Training images must be at least 11 pixels on both axes for SSIM")
        pixels = torch.as_tensor(target, device="cuda")[None]
        degree = min(step // cfg.sh_degree_interval, cfg.sh_degree)
        rgb, _, info = _rasterize(splats, poses[index], Ks[index], width, height, degree)
        strategy.step_pre_backward(splats, optimizers, state, step, info)
        photo = (1 - cfg.ssim_lambda) * (rgb - pixels).abs().mean()
        photo = photo + cfg.ssim_lambda * (1 - ssim_loss_value(rgb.permute(0, 3, 1, 2), pixels.permute(0, 3, 1, 2)))
        photo.backward()
        global_loss = local_loss = support = photo.new_zeros(())
        if depth_weight and step >= depth_start:
            target_z, target_confidence = targets[index]
            depth_h, depth_w = target_z.shape
            z, alpha = _render_depth(splats, poses[index], depth_Ks[index], depth_w, depth_h)
            global_loss, local_loss, support = depth_losses(z, alpha, target_z, target_confidence)
            (depth_weight * (global_loss + recipe["local_weight"] * local_loss)).backward()
        teacher_loss, applied_weight, teacher_name = None, 0., None
        total_loss = photo + global_loss + local_loss
        if teacher_sampler is not None and step >= teacher_start:
            teacher_index = teacher_sampler.next()
            pose, K, color, mask = teacher_tensors[teacher_index]
            virtual_h, virtual_w = mask.shape
            rendered, _, _ = _rasterize(splats, pose, K, virtual_w, virtual_h, degree)
            teacher_loss = warp_teacher_loss(rendered[0], color, mask)
            applied_weight = teacher_weight * min(1., (step - teacher_start + 1) / teacher_ramp)
            (applied_weight * teacher_loss).backward()
            total_loss = total_loss + teacher_loss
            teacher_name = teacher_views[teacher_index].camera.name
        if not torch.isfinite(total_loss).item():
            raise RuntimeError(f"Nonfinite loss at step {step + 1}")
        for optimizer in optimizers.values():
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        scheduler.step()
        strategy.step_post_backward(splats, optimizers, state, step, info, packed=False)
        if 0 < step < cfg.refine_stop and step % cfg.reset_every == 0:
            reset_opa(params=splats, optimizers=optimizers, state=state, value=strategy.prune_opa * 2)
        if len(splats["means"]) > cfg.max_gaussians:
            raise RuntimeError("Gaussian resource ceiling exceeded")
        completed = step + 1
        if completed % cfg.log_every == 0 or completed == stop:
            torch.cuda.synchronize()
            record = {"step": completed, "rgb_loss": photo.item(), "depth_global": global_loss.item(),
                      "depth_local": local_loss.item(), "depth_support": support.item(), "num_gaussians": len(splats["means"]),
                      "training_seconds": elapsed + time.perf_counter() - tick,
                      "peak_cuda_gib": torch.cuda.max_memory_allocated() / 1024**3}
            if teacher_sampler is not None:
                record.update(teacher_weight=applied_weight, teacher_view=teacher_name,
                              teacher_loss=None if teacher_loss is None else teacher_loss.item())
            with log_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record, allow_nan=False) + "\n")
            print(json.dumps(record, allow_nan=False), flush=True)
        if completed in cfg.save_steps or completed == stop:
            torch.cuda.synchronize()
            camera_state = sampler.state()
            payload = {"schema_version": 1, "step": completed, "config": checkpoint_config, "fingerprint": fingerprint,
                       "splats": splats.state_dict(), "optimizers": {key: opt.state_dict() for key, opt in optimizers.items()},
                       "scheduler": scheduler.state_dict(), "strategy": state, "normalization": normalization,
                       "sampler": camera_state["rng"], "order": camera_state["order"], "cursor": camera_state["cursor"],
                       "python_rng": random.getstate(), "numpy_rng": np.random.get_state(),
                       "torch_rng": torch.get_rng_state(), "cuda_rng": [torch.cuda.get_rng_state()],
                       "training_seconds": elapsed + time.perf_counter() - tick, "sh_degree_active": degree}
            if teacher_sampler is not None:
                payload["teacher_sampler"] = teacher_sampler.state()
            checkpoint = output / "checkpoints" / f"step_{completed:06d}.pt"
            checkpoint.parent.mkdir(exist_ok=True)
            _save_checkpoint(checkpoint, payload)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ("dataset", "scene", "depth-maps", "output"):
        parser.add_argument("--" + option, type=Path, required=True)
    parser.add_argument("--depth-receipt", type=Path)
    parser.add_argument("--config", type=Path, default=Path("configs/part1.json"))
    parser.add_argument("--depth-weight", type=float, default=.15)
    parser.add_argument("--depth-edge", type=int, default=192)
    parser.add_argument("--depth-start", type=int, default=200)
    parser.add_argument("--teacher-weight", type=float, default=0., help="Enable fixed train RGB/DPT warp supervision")
    parser.add_argument("--teacher-start", type=int, default=400, help="Zero-based step when warp supervision begins")
    parser.add_argument("--teacher-ramp", type=int, default=600, help="Length of the linear teacher weight ramp")
    parser.add_argument("--stop-after", type=int)
    parser.add_argument("--resume", type=Path, help="Resume one of your own depth-training checkpoints")
    args = parser.parse_args()
    data = json.loads(args.config.read_text(encoding="utf-8"))
    train(args.dataset, args.scene, args.depth_maps, args.depth_receipt or args.depth_maps.with_name("depth_receipt.json"),
          args.output, TrainConfig(**data.get("training", data)), depth_weight=args.depth_weight,
          depth_edge=args.depth_edge, depth_start=args.depth_start, teacher_weight=args.teacher_weight,
          teacher_start=args.teacher_start, teacher_ramp=args.teacher_ramp, resume=args.resume, stop_after=args.stop_after)


if __name__ == "__main__":
    main()

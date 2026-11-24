"""Continue a verified Part 2 3DGS model with real and optional pseudo views."""

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path, PurePosixPath
import tempfile
import time

import numpy as np
from PIL import Image

from p4.common import sha256_file, source_version, write_json
from p4.data import load_manifest, load_scene
from p4.evaluate import evaluate_renders, evaluation_cameras
from p4.train import _optimizers, _rasterize, cuda_dependencies, ssim_loss_value, training_fingerprint
from scripts.part2_handoff import FROZEN_REV, FROZEN_SHA, load_manifest as load_asset_manifest


MODES = ("real_only", "raw_replay", "ungated", "gated")
HISTORICAL_STEP = 7000


@dataclass(frozen=True)
class FusionConfig:
    steps: int = 3000
    seed: int = 42
    synthetic_weight: float = 0.2
    lr_scale: float = 1.0
    log_every: int = 100
    checkpoint_every: int = 500

    def validate(self):
        if min(self.steps, self.log_every, self.checkpoint_every) < 1 or self.seed < 0:
            raise ValueError("Steps, log interval, and checkpoint interval must be positive; seed must be nonnegative")
        if not 0 < self.synthetic_weight <= 1 or not np.isfinite(self.lr_scale) or self.lr_scale <= 0:
            raise ValueError("Expected synthetic weight in (0, 1] and a finite positive learning-rate scale")


@dataclass(frozen=True)
class SyntheticObservation:
    """Fixed synthetic supervision in the unnormalized Part 2 scene frame."""

    name: str
    c2w: np.ndarray
    K: np.ndarray
    raw_rgb: np.ndarray
    enhanced_rgb: np.ndarray
    confidence: np.ndarray


@dataclass(frozen=True)
class FusionModel:
    splats: dict
    normalization: dict
    sh_degree: int
    checkpoint_path: Path
    checkpoint_sha256: str
    provenance: dict
    mode: str | None = None
    fusion_steps: int = 0


def _assets_for_scene(scene, kind):
    inventory = load_asset_manifest()
    if inventory["frozen_revision"] != FROZEN_REV or inventory["part2_package_sha256"] != FROZEN_SHA:
        raise ValueError("Part 2 asset manifest has an unexpected training source")
    case = PurePosixPath("runs/part2_formal", scene, "full_sparse_7000_v1", kind)
    dataset = PurePosixPath("data/part2_sparse" if kind == "sparse" else "data/prepared", scene)
    assets = {row["relative_path"]: row for row in inventory["files"]}
    return {"checkpoint": assets[str(case / "train/checkpoints/step_007000.pt")],
            "initializer": assets[str(case / "init/scene.npz")],
            "manifest": assets[str(dataset / "manifest.json")],
            "reference": assets[str(dataset / "reference.npz")]}


def _verified_checkpoint(path, expected_sha256):
    import torch

    # The historical file also contains NumPy RNG objects. Authenticate the
    # published bytes using the same open descriptor before pickle loading.
    with Path(path).open("rb") as stream:
        digest = hashlib.sha256()
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
        if digest.hexdigest() != expected_sha256:
            raise ValueError(f"Historical Part 2 checkpoint SHA-256 mismatch: {path}")
        stream.seek(0)
        return torch.load(stream, map_location="cpu", weights_only=False)


def _input_fingerprint(dataset, scene_path):
    manifest = load_manifest(dataset)
    fingerprint = training_fingerprint(dataset, scene_path, manifest)
    # train.py's manifest, image, and scene checks are shared with the frozen
    # release. Its package hash refers to the current import and is separate
    # from the historical source hash authenticated by the asset inventory.
    return {name: fingerprint[name] for name in ("manifest_sha256", "scene_sha256")}


def _initial_model(saved, checkpoint, provenance):
    if saved["step"] < saved["config"]["refine_stop"]:
        raise ValueError("Part 3 requires an initial model that has finished densification")
    return FusionModel({name: tensor.cuda() for name, tensor in saved["splats"].items()},
                       saved["normalization"], saved["sh_degree_active"], checkpoint,
                       provenance["training_checkpoint_sha256"], provenance)


def load_part2_checkpoint(dataset, scene_path, checkpoint, expected_sha256=None, *, kind="sparse"):
    """Read a published M0 checkpoint, retaining only rendering tensors on GPU."""
    if kind not in {"sparse", "full"}:
        raise ValueError(f"Unknown Part 2 condition: {kind}")
    dataset, scene_path, checkpoint = Path(dataset), Path(scene_path), Path(checkpoint).resolve()
    scene = load_manifest(dataset)["scene"]
    assets = _assets_for_scene(scene, kind)
    digest = assets["checkpoint"]["sha256"]
    if expected_sha256 is not None and expected_sha256 != digest:
        raise ValueError(f"Requested SHA-256 does not match the published {scene}/{kind} checkpoint")
    inputs = _input_fingerprint(dataset, scene_path)
    if inputs != {"manifest_sha256": assets["manifest"]["sha256"],
                  "scene_sha256": assets["initializer"]["sha256"]}:
        raise ValueError("Prepared data or initializer does not match the published Part 2 artifacts")
    cuda_dependencies()
    saved = _verified_checkpoint(checkpoint, digest)
    if (saved["schema_version"] != 1 or saved["step"] != HISTORICAL_STEP
            or saved["fingerprint"] != {**inputs, "package_source_sha256": FROZEN_SHA}):
        raise ValueError("Part 2 checkpoint schema, source fingerprint, or training inputs do not match")
    provenance = {**inputs, "scene": scene, "condition": kind, "step": saved["step"],
                  "part2_git_commit": FROZEN_REV, "part2_package_sha256": FROZEN_SHA,
                  "part2_checkpoint_sha256": digest, "training_checkpoint_sha256": digest,
                  "reference_sha256": assets["reference"]["sha256"]}
    return _initial_model(saved, checkpoint, provenance)


def load_local_checkpoint(dataset, scene_path, checkpoint, expected_sha256):
    """Load an independently produced p4.train checkpoint and its recorded digest.

    Use only checkpoints produced by the group's own training run. The digest
    is an integrity check and must come from the trusted initial-run receipt.
    """
    dataset, scene_path, checkpoint = Path(dataset), Path(scene_path), Path(checkpoint).resolve()
    manifest = load_manifest(dataset)
    fingerprint = training_fingerprint(dataset, scene_path, manifest)
    cuda_dependencies()
    saved = _verified_checkpoint(checkpoint, expected_sha256)
    if saved["schema_version"] != 1 or saved["fingerprint"] != fingerprint:
        raise ValueError("Local checkpoint is incompatible with the current p4.train implementation or inputs")
    provenance = {**fingerprint, "scene": manifest["scene"], "condition": "local_sparse", "step": saved["step"],
                  "training_checkpoint_sha256": expected_sha256,
                  "reference_sha256": sha256_file(dataset / "reference.npz")}
    return _initial_model(saved, checkpoint, provenance)


def _read_fusion_payload(path):
    import torch

    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("schema_version") != 1 or payload.get("format") != "p4_part3_fusion":
        raise ValueError(f"Unknown Part 3 checkpoint schema: {path}")
    return payload


def _fusion_model(path, digest, payload, splats):
    run = payload["run"]
    return FusionModel(splats, payload["normalization"], payload["sh_degree"], path, digest,
                       run["initial"], run["mode"], run["parent_fusion_steps"] + payload["round_step"])


def load_fusion_checkpoint(path):
    """Load a completed Part 3 checkpoint without retaining optimizer tensors."""
    path = Path(path).resolve()
    cuda_dependencies()
    payload = _read_fusion_payload(path)
    if payload["round_step"] != payload["run"]["config"]["steps"]:
        raise ValueError(f"Part 3 round is incomplete: {path}; re-run fine_tune to resume")
    return _fusion_model(path, sha256_file(path), payload,
                         {key: tensor.cuda() for key, tensor in payload["splats"].items()})


def _validate_camera(c2w, K):
    if c2w.shape != (4, 4) or K.shape != (3, 3):
        raise ValueError("Expected 4x4 c2w and 3x3 intrinsics")
    rotation = c2w[:3, :3]
    if (not np.isfinite(c2w).all() or not np.isfinite(K).all()
            or not np.allclose(c2w[3], [0, 0, 0, 1], atol=1e-5)
            or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-3)
            or not np.isclose(np.linalg.det(rotation), 1, atol=1e-3)
            or np.any(np.diag(K)[:2] <= 0)):
        raise ValueError("Invalid c2w transform or intrinsics")


def _normalized_poses(poses, normalization):
    poses = np.array(poses, dtype=np.float32, copy=True)
    poses[..., :3, 3] = (poses[..., :3, 3] - normalization["origin"]) / normalization["radius"]
    return poses


def render_view(model, c2w, K, width, height):
    """Render float32 RGB, alpha and expected camera-Z depth; depth uses normalized scene units."""
    import torch
    from gsplat import rasterization

    c2w, K = np.asarray(c2w), np.asarray(K)
    _validate_camera(c2w, K)
    if min(width, height) < 1:
        raise ValueError("Image dimensions must be positive")
    device = model.splats["means"].device
    pose = torch.as_tensor(_normalized_poses(c2w, model.normalization), device=device)
    K = torch.as_tensor(K, dtype=torch.float32, device=device)
    splats = model.splats
    with torch.inference_mode():
        rendered, alpha, _ = rasterization(
            means=splats["means"], quats=splats["quats"], scales=splats["scales"].exp(),
            opacities=splats["opacities"].sigmoid(), colors=torch.cat([splats["sh0"], splats["shN"]], dim=1),
            viewmats=torch.linalg.inv(pose[None]), Ks=K[None], width=width, height=height,
            sh_degree=model.sh_degree, packed=False, near_plane=0.01, far_plane=1e10,
            render_mode="RGB+ED", rasterize_mode="classic",
        )
        return {"rgb": rendered[0, ..., :3].clamp(0, 1).cpu().numpy(),
                "depth": rendered[0, ..., 3].cpu().numpy(), "alpha": alpha[0, ..., 0].cpu().numpy()}


def _observation_digest(observations, manifest):
    names = {str(key) for frame in manifest["frames"] for key in (frame["image"], frame["id"])}
    digest = hashlib.sha256()
    for view in observations:
        if not view.name or view.name in names:
            raise ValueError(f"Synthetic view ID is empty or collides with another observation: {view.name}")
        names.add(view.name)
        _validate_camera(view.c2w, view.K)
        if (view.raw_rgb.ndim != 3 or view.raw_rgb.shape[2] != 3
                or view.enhanced_rgb.shape != view.raw_rgb.shape
                or view.confidence.shape != view.raw_rgb.shape[:2]
                or min(view.confidence.shape) < 1):
            raise ValueError(f"Mismatched RGB/confidence shapes for {view.name}")
        for array in (view.raw_rgb, view.enhanced_rgb, view.confidence):
            if not np.isfinite(array).all() or array.min() < 0 or array.max() > 1:
                raise ValueError(f"RGB and confidence must be finite in [0,1]: {view.name}")
        digest.update(json.dumps(view.name).encode())
        for array in (view.c2w, view.K, view.raw_rgb, view.enhanced_rgb, view.confidence):
            digest.update(json.dumps(array.shape).encode())
            digest.update(array.dtype.str.encode())
            digest.update(np.ascontiguousarray(array).view(np.uint8))
    return digest.hexdigest()


class _CameraSampler:
    """Permutation cycles compatible with the original Part 2 sampler."""

    def __init__(self, size, state=None, seed=0):
        self.size = size
        self.rng = np.random.default_rng(seed)
        self.order, self.cursor = [], 0
        if state is not None:
            self.rng.bit_generator.state = state["rng"]
            self.order, self.cursor = state["order"], state["cursor"]
            if sorted(self.order) not in ([], list(range(size))) or not 0 <= self.cursor <= len(self.order):
                raise ValueError("Checkpoint camera sampler differs from its view set")

    def next(self):
        if self.cursor >= len(self.order):
            self.order = self.rng.permutation(self.size).tolist()
            self.cursor = 0
        index = self.order[self.cursor]
        self.cursor += 1
        return index

    def state(self):
        return {"rng": self.rng.bit_generator.state, "order": self.order, "cursor": self.cursor}


def _load_training_state(model, mode, run, path):
    if path.exists():
        payload = _read_fusion_payload(path)
        if payload["run"] != run:
            raise ValueError(f"Resume mismatch (configuration, parent, code, or supervision): {path}")
        return payload
    if model.mode is not None:
        if model.mode != mode:
            raise ValueError("Each Part 3 branch must continue its own previous checkpoint")
        if sha256_file(model.checkpoint_path) != model.checkpoint_sha256:
            raise ValueError("Part 3 parent checkpoint changed since load")
        parent = _read_fusion_payload(model.checkpoint_path)
        return {**parent, "round_step": 0, "training_seconds": 0.0, "synthetic_sampler": None}
    historical = _verified_checkpoint(model.checkpoint_path, model.provenance["training_checkpoint_sha256"])
    numpy_rng = historical["numpy_rng"]
    return {"splats": historical["splats"], "optimizers": historical["optimizers"],
            "real_sampler": {"rng": historical["sampler"], "order": historical["order"],
                             "cursor": historical["cursor"]}, "synthetic_sampler": None,
            "historical_state": {"terminal_scheduler": historical["scheduler"],
                                 "terminal_learning_rates": {name: opt["param_groups"][0]["lr"]
                                                             for name, opt in historical["optimizers"].items()},
                                 "strategy": historical["strategy"], "refine_stop": historical["config"]["refine_stop"],
                                 "ssim_lambda": historical["config"]["ssim_lambda"],
                                 "seed": historical["config"]["seed"], "python_rng": historical["python_rng"],
                                 "numpy_rng": (numpy_rng[0], numpy_rng[1].tolist(), *numpy_rng[2:]),
                                 "torch_rng": historical["torch_rng"], "cuda_rng": historical["cuda_rng"]},
            "round_step": 0, "training_seconds": 0.0}


def _save_checkpoint(path, payload):
    import torch

    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".checkpoint-", suffix=".tmp", delete=False) as file:
        temporary = Path(file.name)
    try:
        torch.save(payload, temporary)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _train_log_start(path, step):
    if not path.exists():
        return
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue  # An interrupted append may leave one partial tail row.
        if row["step"] <= step:
            rows.append(line + "\n")
    path.write_text("".join(rows), encoding="utf-8")


def _rgb(path):
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.float32) / 255


def _validate_inputs(model, dataset, scene_path):
    inputs = _input_fingerprint(dataset, scene_path)
    if any(model.provenance[key] != value for key, value in inputs.items()):
        raise ValueError("Part 3 must use the original Part 2 manifest and initializer")


def fine_tune(model, dataset, scene_path, output, mode, observations=(), config=None):
    """Run or resume one round from M0 or the same branch's previous round.

    Every update uses the same real-view sampler and Part 2 RGB objective.
    All pseudo-view branches add one L1 term drawn with an independent sampler.
    Confidence multiplies per-pixel residuals; its image-wide mean preserves
    downweighting even when the confident region is small.
    """
    cfg = config or FusionConfig()
    cfg.validate()
    if mode not in MODES:
        raise ValueError(f"Expected one of {MODES}: {mode}")
    observations = tuple(observations)
    if mode != "real_only" and not observations:
        raise ValueError(f"{mode} requires synthetic observations")
    if model.mode is not None and model.mode != mode:
        raise ValueError("Each Part 3 branch must continue its own previous checkpoint")
    torch = cuda_dependencies()
    dataset, scene_path, output = Path(dataset), Path(scene_path), Path(output).resolve()
    checkpoint = output / "checkpoint.pt"
    if checkpoint == model.checkpoint_path:
        raise ValueError("Use a new directory for the next fusion round")
    _validate_inputs(model, dataset, scene_path)
    manifest, scene = load_manifest(dataset), load_scene(scene_path)
    run = {"mode": mode, "config": asdict(cfg), "initial": model.provenance,
           "parent_checkpoint_sha256": model.checkpoint_sha256, "parent_fusion_steps": model.fusion_steps,
           "observations_sha256": _observation_digest(observations, manifest),
           "observations": [view.name for view in observations], "n_real_views": len(scene["names"]),
           "runtime_package_sha256": source_version()["package_source_sha256"],
           "adapter_sha256": sha256_file(__file__)}
    if set(scene["names"].tolist()) != {f["image"] for f in manifest["frames"] if f["split"] == "train"}:
        raise ValueError("Fusion training requires all and only the original sparse training views")
    output.mkdir(parents=True, exist_ok=True)
    provenance = output / "fusion.json"
    if provenance.exists() and json.loads(provenance.read_text(encoding="utf-8"))["run"] != run:
        raise ValueError(f"Fusion output directory already belongs to a different round: {output}")
    state = _load_training_state(model, mode, run, checkpoint)
    start, elapsed = state["round_step"], state["training_seconds"]
    if start == cfg.steps:
        return _fusion_model(checkpoint, sha256_file(checkpoint), state,
                             {key: tensor.cuda() for key, tensor in state["splats"].items()})
    historical_state = state["historical_state"]
    splats = torch.nn.ParameterDict({key: torch.nn.Parameter(tensor.cuda())
                                     for key, tensor in state["splats"].items()})
    optimizers = _optimizers(splats)
    for key, optimizer in optimizers.items():
        optimizer.load_state_dict(state["optimizers"][key])
        optimizer.param_groups[0]["lr"] = historical_state["terminal_learning_rates"][key] * cfg.lr_scale
    real = _CameraSampler(len(scene["names"]), state["real_sampler"])
    synthetic = None
    if mode != "real_only":
        synthetic = _CameraSampler(len(observations), state["synthetic_sampler"],
                                   seed=cfg.seed + model.fusion_steps)
    del state
    ssim_weight = historical_state["ssim_lambda"]
    write_json(provenance, {"run": run, "densification": "frozen", "scheduler": "terminal Part 2 rates held fixed",
                           "lr": {key: opt.param_groups[0]["lr"] for key, opt in optimizers.items()},
                           "source": source_version(), "torch": torch.__version__, "gsplat": "1.5.3",
                           "gpu": torch.cuda.get_device_name(), "test_data_used_in_fit": False})
    log_path = output / "train.jsonl"
    _train_log_start(log_path, start)
    real_poses = torch.as_tensor(_normalized_poses(scene["c2w"], model.normalization), device="cuda")
    real_intrinsics = torch.as_tensor(scene["K"], dtype=torch.float32, device="cuda")
    if synthetic is not None:
        synthetic_poses = torch.as_tensor(_normalized_poses([v.c2w for v in observations], model.normalization), device="cuda")
        synthetic_intrinsics = torch.as_tensor(np.stack([v.K for v in observations]), dtype=torch.float32, device="cuda")
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    tick = time.perf_counter()
    for step in range(start + 1, cfg.steps + 1):
        index = real.next()
        target = _rgb(dataset / str(scene["names"][index]))
        height, width = target.shape[:2]
        pixels = torch.as_tensor(target, device="cuda")[None]
        rgb, _, _ = _rasterize(splats, real_poses[index], real_intrinsics[index], width, height, model.sh_degree)
        ssim = ssim_loss_value(rgb.permute(0, 3, 1, 2), pixels.permute(0, 3, 1, 2))
        real_loss = (1 - ssim_weight) * (rgb - pixels).abs().mean() + ssim_weight * (1 - ssim)
        real_loss.backward()
        synthetic_loss = None
        if synthetic is not None:
            pseudo_index = synthetic.next()
            view = observations[pseudo_index]
            target = view.raw_rgb if mode == "raw_replay" else view.enhanced_rgb
            height, width = target.shape[:2]
            rgb, _, _ = _rasterize(splats, synthetic_poses[pseudo_index], synthetic_intrinsics[pseudo_index],
                                   width, height, model.sh_degree)
            residual = (rgb[0] - torch.as_tensor(target, dtype=torch.float32, device="cuda")).abs()
            if mode == "gated":
                residual = residual * torch.as_tensor(view.confidence, dtype=torch.float32, device="cuda")[..., None]
            synthetic_loss = residual.mean()
            (cfg.synthetic_weight * synthetic_loss).backward()
        finite = torch.isfinite(real_loss.detach())
        if synthetic_loss is not None:
            finite = finite & torch.isfinite(synthetic_loss.detach())
        if not finite.item():
            raise RuntimeError(f"Non-finite fusion loss at round step {step}")
        for optimizer in optimizers.values():
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        if step % cfg.log_every == 0 or step == cfg.steps:
            torch.cuda.synchronize()
            row = {"step": step, "fusion_step": model.fusion_steps + step,
                   "real_loss": real_loss.item(), "real_view": str(scene["names"][index]),
                   "synthetic_loss": None if synthetic_loss is None else synthetic_loss.item(),
                   "synthetic_view": None if synthetic_loss is None else view.name,
                   "num_gaussians": len(splats["means"]), "training_seconds": elapsed + time.perf_counter() - tick,
                   "peak_cuda_gib": torch.cuda.max_memory_allocated() / 1024**3}
            with log_path.open("a", encoding="utf-8") as log:
                log.write(json.dumps(row, allow_nan=False) + "\n")
            print(json.dumps(row, allow_nan=False), flush=True)
        if step % cfg.checkpoint_every == 0 or step == cfg.steps:
            torch.cuda.synchronize()
            payload = {"schema_version": 1, "format": "p4_part3_fusion", "run": run, "round_step": step,
                       "training_seconds": elapsed + time.perf_counter() - tick, "normalization": model.normalization,
                       "sh_degree": model.sh_degree, "splats": splats.state_dict(),
                       "optimizers": {key: opt.state_dict() for key, opt in optimizers.items()},
                       "real_sampler": real.state(), "synthetic_sampler": None if synthetic is None else synthetic.state(),
                       "historical_state": historical_state}
            _save_checkpoint(checkpoint, payload)
    return _fusion_model(checkpoint, sha256_file(checkpoint), payload, splats.state_dict())


def evaluate_model(model, dataset, scene_path, output, lpips=True):
    """Evaluate the raw model on the frozen Part 2 held-out split and train-fit Sim3."""
    import torch

    dataset, scene_path, output = Path(dataset), Path(scene_path), Path(output)
    _validate_inputs(model, dataset, scene_path)
    if sha256_file(dataset / "reference.npz") != model.provenance["reference_sha256"]:
        raise ValueError("Reference cameras differ from the published Part 2 reference")
    cameras = evaluation_cameras(dataset, scene_path)
    poses = _normalized_poses(cameras["c2w"], model.normalization)
    render_dir = output / "renders"
    manifest = load_manifest(dataset)
    test_sha256 = {frame["image"]: frame["sha256"] for frame in manifest["frames"] if frame["split"] == "test"}
    torch.cuda.synchronize()
    tick = time.perf_counter()
    with torch.inference_mode():
        for name, pose, K in zip(cameras["names"], poses, cameras["K"]):
            name = str(name)
            if sha256_file(dataset / name) != test_sha256[name]:
                raise ValueError(f"Held-out image changed: {name}")
            with Image.open(dataset / name) as image:
                width, height = image.size
            rgb, _, _ = _rasterize(model.splats, torch.as_tensor(pose, device="cuda"),
                                   torch.as_tensor(K, dtype=torch.float32, device="cuda"),
                                   width, height, model.sh_degree)
            pixels = (rgb[0].clamp(0, 1).cpu().numpy() * 255).round().astype(np.uint8)
            destination = render_dir / Path(name).with_suffix(".png")
            destination.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(pixels).save(destination)
    torch.cuda.synchronize()
    render_record = {**cameras["alignment"], "render_seconds": time.perf_counter() - tick,
                     "checkpoint_sha256": model.checkpoint_sha256, "fusion_steps": model.fusion_steps,
                     "count": len(cameras["names"]), "raw_gaussian_renders": True}
    write_json(render_dir / "render.json", render_record)
    metrics = evaluate_renders(dataset, render_dir, lpips=lpips)
    metrics.update({"alignment": cameras["alignment"], "mode": model.mode, "fusion_steps": model.fusion_steps,
                    "checkpoint_sha256": model.checkpoint_sha256, "initial": model.provenance,
                    "render_seconds": render_record["render_seconds"]})
    write_json(output / "metrics.json", metrics)
    return metrics

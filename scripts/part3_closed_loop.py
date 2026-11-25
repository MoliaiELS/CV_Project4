"""Run two rounds of diffusion-assisted reconstruction with matched controls."""

import argparse
import csv
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from p4.common import sha256_file, source_version, write_json
from p4.data import load_manifest
from scripts import part3_fusion as fusion
from scripts.part3_restore import DifixRestorer
from scripts.part3_views import (confidence_map, interpolate_views, read_rgb, save_rgb,
                                 select_views, training_cameras)


COURSE_SCENES = ("Re10k-1", "Dl3dv-2", "Waymo-405841")
ARMS = (*fusion.MODES, "fixed_gated")


def _render(model, camera):
    return fusion.render_view(model, camera.c2w, camera.K, camera.width, camera.height)


def _read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _source_hash():
    digest = hashlib.sha256()
    for name in ("part3_closed_loop.py", "part3_fusion.py", "part3_restore.py", "part3_views.py"):
        digest.update(bytes.fromhex(sha256_file(Path(__file__).with_name(name))))
    return digest.hexdigest()


def _plan_and_generate(model, candidates, dataset, output, arm, round_index, args, restorer):
    if arm == "real_only":
        return [], {"planning_seconds": 0., "generation_seconds": 0., "diffix": None, "observations": []}
    tick = time.perf_counter()
    policy = "fixed" if arm == "fixed_gated" else "active"
    chosen, scores = select_views(candidates, lambda camera: _render(model, camera), args.views, policy)
    planning_seconds = time.perf_counter() - tick
    tick = time.perf_counter()
    observations, records = [], []
    for index, target in enumerate(chosen):
        camera = target.camera
        folder = output / camera.name
        raw = _render(model, camera)
        if arm == "raw_replay":
            enhanced, confidence, stats = raw["rgb"], np.ones_like(raw["alpha"]), {}
        else:
            save_rgb(folder / "raw.png", raw["rgb"])
            diffusion_seed = args.seed + round_index * 1000 + index
            restored = restorer.restore(folder / "raw.png", dataset / target.reference.name, seed=diffusion_seed)
            enhanced = np.asarray(restored, dtype=np.float32) / 255.
            anchor_inputs = [(anchor, _render(model, anchor), read_rgb(dataset / anchor.name))
                             for anchor in target.anchors]
            confidence, stats = confidence_map(target, raw, enhanced, anchor_inputs, model.normalization["radius"])
        observation = fusion.SyntheticObservation(
            camera.name, np.asarray(camera.c2w, dtype=np.float64), np.asarray(camera.K, dtype=np.float64),
            raw["rgb"], enhanced, confidence)
        observations.append(observation)
        save_rgb(folder / "raw.png", observation.raw_rgb)
        save_rgb(folder / "enhanced.png", observation.enhanced_rgb)
        save_rgb(folder / "alpha.png", raw["alpha"])
        save_rgb(folder / "confidence.png", confidence)
        save_rgb(folder / "reference.png", read_rgb(dataset / target.reference.name))
        records.append({
            "target": camera.name, "anchors": [anchor.name for anchor in target.anchors],
            "reference": target.reference.name, "c2w": camera.c2w.tolist(), "K": camera.K.tolist(),
            "width": camera.width, "height": camera.height, "fraction": target.fraction,
            "seed": None if arm == "raw_replay" else diffusion_seed,
            "parent_checkpoint_sha256": model.checkpoint_sha256, **stats,
        })
    return observations, {
        "policy": policy, "candidates": scores, "observations": records, "planning_seconds": planning_seconds,
        "generation_seconds": time.perf_counter() - tick, "diffix": None if restorer is None else restorer.provenance(),
    }


def _setup_round(model, candidates, dataset, output, arm, round_index, args, model_dir):
    # Reuse exact float arrays after interruption; PNG files serve as evidence.
    manifest = output / "observations.json"
    if manifest.exists():
        record = _read_json(manifest)
        if record["parent_checkpoint_sha256"] != model.checkpoint_sha256:
            raise ValueError("Pseudo observations belong to a different parent Gaussian model")
        observations = []
        for view in record["observations"]:
            with np.load(output / view["target"] / "training_arrays.npz", allow_pickle=False) as arrays:
                observations.append(fusion.SyntheticObservation(
                    view["target"], np.array(view["c2w"]), np.array(view["K"]),
                    arrays["raw"], arrays["enhanced"], arrays["confidence"]))
        return observations, record
    restorer = None
    if arm not in {"real_only", "raw_replay"}:
        restorer = DifixRestorer(model_dir=model_dir, dtype=args.precision, max_edge=args.max_edge, seed=args.seed)
    observations, record = _plan_and_generate(model, candidates, dataset, output, arm, round_index, args, restorer)
    if restorer is not None:
        import torch

        del restorer
        torch.cuda.empty_cache()
    for view in observations:
        np.savez_compressed(output / view.name / "training_arrays.npz", raw=view.raw_rgb,
                            enhanced=view.enhanced_rgb, confidence=view.confidence)
    record["parent_checkpoint_sha256"] = model.checkpoint_sha256
    write_json(manifest, record)
    return observations, record


def _evaluate(model, dataset, scene, output):
    path = output / "metrics.json"
    if path.exists():
        record = _read_json(path)
        if record.get("checkpoint_sha256") == model.checkpoint_sha256 and record.get("complete"):
            return record
    record = fusion.evaluate_model(model, dataset, scene, output)
    if not record["complete"] or record["lpips"] is None:
        raise RuntimeError(f"Complete LPIPS evaluation is required: {record.get('lpips_status')}")
    return record


def _write_comparison(output):
    rows = []
    records = sorted(output.glob("*/baseline/stage.json")) + sorted(output.glob("*/*/round*/stage.json"))
    for path in records:
        rows.append(_read_json(path))
    if rows:
        destination = output / "comparison.csv"
        temporary = destination.with_suffix(".tmp")
        with temporary.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        temporary.replace(destination)


def _stage_record(scene, arm, round_index, metrics, output, run_root, generation, seconds):
    observations = generation.get("observations", [])
    fusion_path = output / "fusion.json"
    update = _read_json(fusion_path) if fusion_path.exists() else {}
    train_log = output / "train.jsonl"
    if train_log.exists():
        update = {**update, **json.loads(train_log.read_text(encoding="utf-8").splitlines()[-1])}
    return {
        "scene": scene, "arm": arm, "round": round_index, "train_frames": metrics.get("train_frames"),
        "test_frames": metrics["test_frames"], "split_hash": metrics["split_hash"],
        "psnr": metrics["psnr"], "ssim": metrics["ssim"], "lpips": metrics["lpips"],
        "views": len(observations), "mean_confidence": (float(np.mean([v["mean_confidence"] for v in observations]))
                                                         if observations and "mean_confidence" in observations[0] else None),
        "planning_seconds": generation.get("planning_seconds", 0.),
        "generation_seconds": generation.get("generation_seconds", 0.),
        "training_seconds": update.get("training_seconds", 0.),
        "render_seconds": metrics.get("render_seconds"), "peak_cuda_gib": update.get("peak_cuda_gib"),
        "stage_seconds": seconds, "checkpoint_sha256": metrics["checkpoint_sha256"],
        "path": output.relative_to(run_root).as_posix(),
    }


def _run_scene(args, name, dataset, scene, checkpoint, expected_sha256=None):
    scene_root = args.output / name
    cameras = training_cameras(dataset, scene)
    candidates = interpolate_views(cameras)
    config = fusion.FusionConfig(steps=args.steps, seed=args.seed, synthetic_weight=args.synthetic_weight)
    run_config = {**asdict(config), "views": args.views, "precision": args.precision, "max_edge": args.max_edge,
                  "implementation_sha256": _source_hash(), "source": source_version()}
    spec = scene_root / "protocol.json"
    if spec.exists():
        previous = _read_json(spec)
        if {k: v for k, v in previous.items() if k != "source"} != {k: v for k, v in run_config.items() if k != "source"}:
            raise ValueError(f"Protocol changed for {scene_root}. Choose a new output directory.")
    else:
        write_json(spec, run_config)
    if expected_sha256:
        m0 = fusion.load_local_checkpoint(dataset, scene, checkpoint, expected_sha256)
    else:
        m0 = fusion.load_part2_checkpoint(dataset, scene, checkpoint)
    baseline = scene_root / "baseline"
    elapsed = time.perf_counter()
    metrics = _evaluate(m0, dataset, scene, baseline)
    metrics["train_frames"] = len(cameras)
    if not (baseline / "stage.json").exists():
        record = _stage_record(name, "sparse", 0, metrics, baseline, args.output, {}, time.perf_counter() - elapsed)
        write_json(baseline / "stage.json", record)
    _write_comparison(args.output)
    for arm in args.arms:
        root = scene_root / arm
        if (root / "round2/stage.json").exists():
            continue
        model = m0
        for round_index in (1, 2):
            output = root / f"round{round_index}"
            stage = output / "stage.json"
            if stage.exists():
                model = fusion.load_fusion_checkpoint(output / "checkpoint.pt")
                continue
            start = time.perf_counter()
            print(f"{name} {arm} round {round_index} starting", flush=True)
            observations, generation = _setup_round(model, candidates, dataset, output, arm, round_index, args, args.model_dir)
            model = fusion.fine_tune(model, dataset, scene, output, "gated" if arm == "fixed_gated" else arm,
                                     observations, config)
            metrics = _evaluate(model, dataset, scene, output)
            metrics["train_frames"] = len(cameras)
            record = _stage_record(name, arm, round_index, metrics, output, args.output, generation,
                                   time.perf_counter() - start)
            write_json(stage, record)
            _write_comparison(args.output)
            print(f"{name} {arm} round {round_index} PSNR {metrics['psnr']:.4f} "
                  f"SSIM {metrics['ssim']:.4f} LPIPS {metrics['lpips']:.4f}", flush=True)
        if args.keep_checkpoints != "all":
            (root / "round1/checkpoint.pt").unlink(missing_ok=True)
        if args.keep_checkpoints == "none" or (args.keep_checkpoints == "gated" and arm != "gated"):
            (root / "round2/checkpoint.pt").unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets-root", type=Path, help="Root of the archived Part 2 assets")
    parser.add_argument("--scene", choices=COURSE_SCENES, action="append")
    parser.add_argument("--supplemental", type=Path, help="JSON describing an independently trained M0")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--arms", choices=ARMS, nargs="+", default=list(fusion.MODES))
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--views", type=int, default=4)
    parser.add_argument("--synthetic-weight", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--precision", choices=("float32", "float16", "bfloat16"), default="float16")
    parser.add_argument("--max-edge", type=int, default=768)
    parser.add_argument("--keep-checkpoints", choices=("all", "final", "gated", "none"), default="all")
    args = parser.parse_args()
    args.output = args.output.resolve()
    if args.assets_root is None and args.supplemental is None:
        parser.error("Provide --assets-root for the course scenes or --supplemental for an independent scene.")
    if args.assets_root is not None:
        for name in args.scene or COURSE_SCENES:
            run = args.assets_root / "runs/part2_formal" / name / "full_sparse_7000_v1/sparse"
            _run_scene(args, name, args.assets_root / "data/part2_sparse" / name, run / "init/scene.npz",
                       run / "train/checkpoints/step_007000.pt")
    if args.supplemental is not None:
        supplemental = _read_json(args.supplemental)
        dataset = Path(supplemental["dataset"])
        _run_scene(args, load_manifest(dataset)["scene"], dataset, Path(supplemental["initializer"]),
                   Path(supplemental["checkpoint"]), supplemental["checkpoint_sha256"])


if __name__ == "__main__":
    main()

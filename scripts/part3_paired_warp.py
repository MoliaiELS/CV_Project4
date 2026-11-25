"""Fit and score paired real-only, naive, and confidence-aware two-round reconstructions."""

import argparse
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from p4.common import sha256_file, source_version, write_json
from p4.data import load_manifest
from scripts import part3_fusion as fusion
from scripts import part3_progressive as progressive
from scripts.part3_closed_loop import _evaluate, _read_json, _stage_record, _write_comparison
from scripts.part3_restore import MODEL_ID, MODEL_REVISION, PRECISIONS
from scripts.part3_views import read_rgb, training_cameras


ARMS = {"real_only": "real_only", "warp_naive": "ungated", "warp_gated": "gated"}
ROUNDS = (1, 2)


def _digest(record):
    payload = json.dumps(record, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode()).hexdigest()


def _bind(path, record):
    if path.exists():
        if _read_json(path) != record:
            raise ValueError(f"Recorded protocol or artifact changed: {path}")
    else:
        write_json(path, record)


def _m0_inputs(path):
    receipt = _read_json(path)
    dataset, initializer, checkpoint = (Path(receipt[key]).resolve()
                                        for key in ("dataset", "initializer", "checkpoint"))
    digest = receipt["checkpoint_sha256"]
    if "initializer_sha256" in receipt and sha256_file(initializer) != receipt["initializer_sha256"]:
        raise ValueError("M0 camera initializer differs from its receipt")
    return dataset, initializer, checkpoint, digest


def _matched_rgb(observation, left, right):
    # For the confidence-aware arm these rejected pixels have zero loss. The
    # naive arm receives the same RGB target with uniform pixel weights.
    if left.shape != observation.enhanced_rgb.shape or right.shape != left.shape:
        raise ValueError(f"Restored image sizes differ for {observation.name}")
    restored = (left + right) * 0.5
    rgb = np.where(observation.confidence[..., None] > 0, observation.enhanced_rgb, restored)
    return np.ascontiguousarray(rgb, dtype=np.float32)


def _plan_targets(m0, cameras, scene_root, views):
    plans = {}
    for round_index in ROUNDS:
        folder = scene_root / "plans" / f"round{round_index}"
        candidates, record = progressive._round_plan(m0, cameras, folder, round_index, views)
        camera_records = [{"name": view.camera.name, "anchors": [camera.name for camera in view.anchors],
                           "K": view.camera.K.tolist(), "c2w": view.camera.c2w.tolist(),
                           "width": view.camera.width, "height": view.camera.height} for view in candidates]
        plans[round_index] = {"policy": "shared_from_m0", "round": round_index,
                              "m0_checkpoint_sha256": m0.checkpoint_sha256,
                              "chosen": record["chosen"], "candidates": record["candidates"],
                              "planning_seconds": record["planning_seconds"],
                              "geometry_sha256": _digest(camera_records)}
        _bind(folder / "fixed_targets.json", {**plans[round_index], "geometry": camera_records})
    return plans


def _observations(model, cameras, dataset, folder, plan, args, warp_depth, manifest):
    generation_plan = {**plan, "parent_checkpoint_sha256": model.checkpoint_sha256}
    _bind(folder / "plan.json", generation_plan)
    observations, _ = progressive._setup_round(model, cameras, dataset, folder, "warp_gated",
                                               plan["round"], args, warp_depth)
    if [view.name for view in observations] != plan["chosen"]:
        raise ValueError("Generated view order differs from the common round plan")
    shared = []
    assets = {}
    for view in observations:
        paths = [folder / view.name / f"restored_{side}.png" for side in ("left", "right")]
        assets[view.name] = [sha256_file(path) for path in paths]
        rgb = _matched_rgb(view, *[read_rgb(path) for path in paths])
        shared.append(replace(view, enhanced_rgb=rgb))
    receipt = {**generation_plan, "observations_sha256": fusion._observation_digest(shared, manifest),
               "generation_receipt_sha256": sha256_file(folder / "observations.json"),
               "restored_png_sha256": assets, "recipe": "confidence target where supported; mean restoration elsewhere"}
    _bind(folder / "matched_supervision.json", receipt)
    return shared, receipt


def _checkpoint_receipt(path, arm, parent_sha, observation_sha, cfg, initial, round_index):
    payload = fusion._read_fusion_payload(path)
    run = payload["run"]
    if (payload["round_step"] != cfg.steps or run["parent_fusion_steps"] != (round_index - 1) * cfg.steps
            or run["parent_checkpoint_sha256"] != parent_sha or run["observations_sha256"] != observation_sha
            or run["mode"] != ARMS[arm] or run["config"] != asdict(cfg) or run["initial"] != initial):
        raise ValueError(f"Checkpoint does not match the paired-run protocol: {path}")
    return {"checkpoint_sha256": sha256_file(path), "parent_checkpoint_sha256": parent_sha,
            "observations_sha256": observation_sha, "round_updates": cfg.steps,
            "cumulative_updates": initial["step"] + round_index * cfg.steps,
            "real_sampler_sha256": _digest(payload["real_sampler"]), "real_views": run["n_real_views"],
            "training_seconds": payload["training_seconds"], "peak_cuda_gib": payload["peak_cuda_gib"],
            "gaussians": len(payload["splats"]["means"])}


def _verify_pair(stages, plan_targets):
    for round_index in ROUNDS:
        rows = [stages[arm][str(round_index)] for arm in ARMS]
        if len({(row["real_sampler_sha256"], row["real_views"], row["round_updates"]) for row in rows}) != 1:
            raise ValueError(f"Real image sampler diverged between arms in round {round_index}")
        for row in rows[1:]:
            if row["targets"] != plan_targets[str(round_index)]:
                raise ValueError(f"Synthetic camera list differs between arms in round {round_index}")
    first = [stages[arm]["1"] for arm in ("warp_naive", "warp_gated")]
    if (first[0]["generation"] != first[1]["generation"]
            or first[0]["observations_sha256"] != first[1]["observations_sha256"]):
        raise ValueError("Naive and confidence-aware arms did not share exact first-round pseudo RGB")


def fit(args):
    dataset, initializer, checkpoint, digest = _m0_inputs(args.m0)
    manifest = load_manifest(dataset)
    scene_root = args.output / manifest["scene"]
    cfg = fusion.FusionConfig(steps=args.steps, seed=args.seed, synthetic_weight=args.synthetic_weight)
    cfg.validate()
    cameras = training_cameras(dataset, initializer)
    if not 1 <= args.views < len(cameras) or args.max_edge < 8:
        raise ValueError("Requested pseudo views must fit distinct sparse-camera intervals")
    m0 = fusion.load_local_checkpoint(dataset, initializer, checkpoint, digest)
    warp_depth = progressive._load_warp_depth(args.warp_depth, cameras, dataset, initializer) if args.warp_depth else None
    protocol = {"schema_version": 1, "scene": manifest["scene"], "split_hash": manifest["split_hash"],
                "initial": m0.provenance, "m0_receipt_sha256": sha256_file(args.m0),
                "m0_normalization": m0.normalization, "arms": ARMS, "rounds": list(ROUNDS),
                "fusion": asdict(cfg), "views": args.views, "precision": args.precision, "max_edge": args.max_edge,
                "difix_model": MODEL_ID, "difix_revision": MODEL_REVISION,
                "adapter_sha256": sha256_file(__file__), "backend_sha256": progressive._implementation_sha256(),
                "warp_depth_sha256": sha256_file(args.warp_depth) if args.warp_depth else None}
    _bind(scene_root / "protocol.json", protocol)
    if not (scene_root / "source.json").exists():
        write_json(scene_root / "source.json", source_version())
    plans = _plan_targets(m0, cameras, scene_root, args.views)
    matched_first = _observations(m0, cameras, dataset, scene_root / "generation/shared/round1",
                                  plans[1], args, warp_depth, manifest)
    empty_observations_sha = fusion._observation_digest([], manifest)
    stages = {arm: {} for arm in ARMS}
    for arm, mode in ARMS.items():
        model = m0
        for round_index in ROUNDS:
            output = scene_root / arm / f"round{round_index}"
            generation = None
            observations = []
            observation_sha = empty_observations_sha
            if arm != "real_only":
                generation = scene_root / "generation" / ("shared" if round_index == 1 else arm) / f"round{round_index}"
                observations, receipt = (matched_first if round_index == 1 else
                                         _observations(model, cameras, dataset, generation, plans[round_index],
                                                       args, warp_depth, manifest))
                observation_sha = receipt["observations_sha256"]
            parent_sha = model.checkpoint_sha256
            print(f"{manifest['scene']} {arm} round {round_index} fit", flush=True)
            model = fusion.fine_tune(model, dataset, initializer, output, mode, observations, cfg)
            stage = _checkpoint_receipt(model.checkpoint_path, arm, parent_sha, observation_sha, cfg,
                                        m0.provenance, round_index)
            if stage["checkpoint_sha256"] != model.checkpoint_sha256:
                raise ValueError(f"Checkpoint changed while recording fit lineage: {output}")
            stage.update(generation=generation.relative_to(scene_root).as_posix() if generation else None,
                         matched_supervision_sha256=sha256_file(generation / "matched_supervision.json") if generation else None,
                         targets=[view.name for view in observations])
            _bind(output / "fit.json", stage)
            stages[arm][str(round_index)] = stage
    targets = {str(index): plan["chosen"] for index, plan in plans.items()}
    _verify_pair(stages, targets)
    _bind(scene_root / "fit.json", {"protocol_sha256": sha256_file(scene_root / "protocol.json"),
                                    "initial_updates": m0.provenance["step"], "targets": targets, "stages": stages})
    return scene_root


def _check_fit(scene_root, m0_path, dataset, initializer, checkpoint, digest):
    fit_receipt = _read_json(scene_root / "fit.json")
    protocol_path = scene_root / "protocol.json"
    protocol = _read_json(protocol_path)
    if (fit_receipt["protocol_sha256"] != sha256_file(protocol_path)
            or protocol["m0_receipt_sha256"] != sha256_file(m0_path)
            or protocol["initial"]["manifest_sha256"] != sha256_file(dataset / "manifest.json")
            or protocol["initial"]["scene_sha256"] != sha256_file(initializer)
            or protocol["initial"]["training_checkpoint_sha256"] != digest or sha256_file(checkpoint) != digest):
        raise ValueError("Frozen fit inputs or protocol differ from the completed fit receipt")
    _verify_pair(fit_receipt["stages"], fit_receipt["targets"])
    plans = {r: _read_json(scene_root / "plans" / f"round{r}" / "fixed_targets.json") for r in ROUNDS}
    for r, plan in plans.items():
        if (plan["round"] != r or plan["m0_checkpoint_sha256"] != digest
                or plan["chosen"] != fit_receipt["targets"][str(r)] or plan["geometry_sha256"] != _digest(plan["geometry"])):
            raise ValueError(f"Fixed target geometry changed before evaluating round {r}")
    for arm in ARMS:
        parent = digest
        for round_index in ROUNDS:
            stage = fit_receipt["stages"][arm][str(round_index)]
            output = scene_root / arm / f"round{round_index}"
            if (stage != _read_json(output / "fit.json") or stage["parent_checkpoint_sha256"] != parent
                    or stage["round_updates"] != protocol["fusion"]["steps"]
                    or stage["cumulative_updates"] != protocol["initial"]["step"] + round_index * stage["round_updates"]
                    or stage["checkpoint_sha256"] != sha256_file(output / "checkpoint.pt")):
                raise ValueError(f"Fit lineage changed before held-out evaluation: {output}")
            if stage["generation"]:
                folder = scene_root / stage["generation"]
                link = folder / "matched_supervision.json"
                generation = _read_json(link)
                if (sha256_file(link) != stage["matched_supervision_sha256"]
                        or generation["parent_checkpoint_sha256"] != parent
                        or generation["geometry_sha256"] != plans[round_index]["geometry_sha256"]
                        or generation["observations_sha256"] != stage["observations_sha256"]
                        or generation["generation_receipt_sha256"] != sha256_file(folder / "observations.json")):
                    raise ValueError(f"Generation lineage changed before held-out evaluation: {folder}")
            parent = stage["checkpoint_sha256"]
    return fit_receipt, protocol


def evaluate(args):
    dataset, initializer, checkpoint, digest = _m0_inputs(args.m0)
    manifest = load_manifest(dataset)
    scene_root = args.output / manifest["scene"]
    completed, protocol = _check_fit(scene_root, args.m0, dataset, initializer, checkpoint, digest)
    fit_sha = sha256_file(scene_root / "fit.json")
    train_count = sum(frame["split"] == "train" for frame in manifest["frames"])
    for arm, round_index in [("sparse", 0), *((arm, r) for arm in ARMS for r in ROUNDS)]:
        started = time.perf_counter()
        if round_index:
            stage = completed["stages"][arm][str(round_index)]
            output = scene_root / arm / f"round{round_index}"
            model = fusion.load_fusion_checkpoint(output / "checkpoint.pt")
            generation = _read_json(scene_root / stage["generation"] / "observations.json") if stage["generation"] else {}
        else:
            stage, generation = {}, {}
            output = scene_root / "baseline"
            model = fusion.load_local_checkpoint(dataset, initializer, checkpoint, digest)
        if model.provenance != protocol["initial"] or model.normalization != protocol["m0_normalization"]:
            raise ValueError(f"Checkpoint changed initializer provenance or normalization: {output}")
        metrics = _evaluate(model, dataset, initializer, output)
        metrics["train_frames"] = train_count
        record = _stage_record(manifest["scene"], arm, round_index, metrics, output, args.output,
                               generation, time.perf_counter() - started)
        record.update(cumulative_updates=completed["initial_updates"] + model.fusion_steps,
                      round_updates=stage.get("round_updates", 0), gaussians=stage.get("gaussians", len(model.splats["means"])),
                      fit_receipt_sha256=fit_sha, generation=stage.get("generation"),
                      training_seconds=stage.get("training_seconds", 0.), peak_cuda_gib=stage.get("peak_cuda_gib"))
        if (output / "stage.json").exists():
            previous = _read_json(output / "stage.json")
            record["stage_seconds"] = previous["stage_seconds"]
        _bind(output / "stage.json", record)
        del model
        print(f"{manifest['scene']} {arm} round {round_index} PSNR {metrics['psnr']:.4f} "
              f"SSIM {metrics['ssim']:.4f} LPIPS {metrics['lpips']:.4f}", flush=True)
    _write_comparison(args.output)
    return scene_root


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    train = commands.add_parser("fit", help="Fit all six stages before opening the held-out evaluator")
    score = commands.add_parser("evaluate", help="Evaluate a completed, verified fit")
    for command in (train, score):
        command.add_argument("--m0", type=Path, required=True, help="Dataset, initializer, and SHA-verified checkpoint receipt")
        command.add_argument("--output", type=Path, required=True)
    train.add_argument("--model-dir", type=Path, required=True)
    train.add_argument("--warp-depth", type=Path, help="Optional authenticated training-view depth maps in the M0 scene frame")
    train.add_argument("--steps", type=int, default=5000)
    train.add_argument("--views", type=int, default=8)
    train.add_argument("--seed", type=int, default=42)
    train.add_argument("--synthetic-weight", type=float, default=0.4)
    train.add_argument("--precision", choices=PRECISIONS, default="float16")
    train.add_argument("--max-edge", type=int, default=768)
    args = parser.parse_args()
    args.output = args.output.resolve()
    (fit if args.command == "fit" else evaluate)(args)


if __name__ == "__main__":
    main()

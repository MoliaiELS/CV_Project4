"""Run two progressive Part 3 rounds with dual-reference restoration and controls."""

import argparse
from dataclasses import asdict
import hashlib
from pathlib import Path
import time

import numpy as np

from p4.common import sha256_file, source_version, write_json
from p4.data import load_manifest
from scripts import part3_fusion as fusion
from scripts.part3_closed_loop import (COURSE_SCENES, _evaluate, _read_json, _render,
                                       _source_hash, _stage_record, _write_comparison)
from scripts.part3_progressive_views import (fuse_restorations, progressive_candidates,
                                             select_progressive_views)
from scripts.part3_restore import DifixRestorer, MODEL_ID, MODEL_REVISION, PRECISIONS
from scripts.part3_views import read_rgb, save_rgb, training_cameras


ARMS = {"real_only": "real_only", "progressive_raw": "raw_replay", "progressive_gated": "gated"}


def _implementation_sha256():
    digest = hashlib.sha256(bytes.fromhex(_source_hash()))
    for path in (Path(__file__), Path(__file__).with_name("part3_progressive_views.py")):
        digest.update(bytes.fromhex(sha256_file(path)))
    return digest.hexdigest()


def _round_plan(model, cameras, output, round_index, count):
    candidates = progressive_candidates(cameras, round_index)
    by_name = {candidate.camera.name: candidate for candidate in candidates}
    path = output / "plan.json"
    if path.exists():
        record = _read_json(path)
    else:
        tick = time.perf_counter()
        chosen, scores = select_progressive_views(
            candidates, lambda camera: _render(model, camera), count, model.normalization["radius"])
        record = {"policy": "progressive", "round": round_index,
                  "parent_checkpoint_sha256": model.checkpoint_sha256,
                  "chosen": [candidate.camera.name for candidate in chosen], "candidates": scores,
                  "planning_seconds": time.perf_counter() - tick}
        write_json(path, record)
    names = record["chosen"]
    if (record["parent_checkpoint_sha256"] != model.checkpoint_sha256 or record["round"] != round_index
            or len(names) != count or len(set(names)) != count or set(names) - by_name.keys()):
        raise ValueError(f"Plan does not match the current parent and requested round: {path}")
    return [by_name[name] for name in names], record


def _observation_from_cache(folder, record):
    arrays_path = folder / "training_arrays.npz"
    if sha256_file(arrays_path) != record["training_arrays_sha256"]:
        raise ValueError(f"Cached pseudo-view arrays failed verification: {arrays_path}")
    with np.load(arrays_path, allow_pickle=False) as arrays:
        return fusion.SyntheticObservation(record["target"], np.asarray(record["c2w"]), np.asarray(record["K"]),
                                           arrays["raw"], arrays["enhanced"], arrays["confidence"])


def _generate_view(model, target, dataset, folder, restorer, seeds):
    camera = target.camera
    raw = _render(model, camera)
    save_rgb(folder / "raw.png", raw["rgb"])
    save_rgb(folder / "alpha.png", raw["alpha"])
    anchor_rgb = [read_rgb(dataset / anchor.name) for anchor in target.anchors]
    for side, image in zip(("left", "right"), anchor_rgb):
        save_rgb(folder / f"reference_{side}.png", image)
    if restorer is None:
        enhanced, confidence, stats = raw["rgb"], np.ones_like(raw["alpha"]), {}
    else:
        restored = []
        for side, anchor, seed in zip(("left", "right"), target.anchors, seeds):
            image = restorer.restore(folder / "raw.png", dataset / anchor.name, seed=seed)
            pixels = np.asarray(image, dtype=np.float32) / 255.
            save_rgb(folder / f"restored_{side}.png", pixels)
            restored.append(pixels)
        anchor_inputs = [(anchor, _render(model, anchor), pixels)
                         for anchor, pixels in zip(target.anchors, anchor_rgb)]
        enhanced, confidence, stats = fuse_restorations(
            target, raw, tuple(restored), anchor_inputs, model.normalization["radius"])
    save_rgb(folder / "enhanced.png", enhanced)
    save_rgb(folder / "confidence.png", confidence)
    arrays = folder / "training_arrays.npz"
    temporary = arrays.with_suffix(".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, raw=raw["rgb"], enhanced=enhanced, confidence=confidence)
    temporary.replace(arrays)
    return {"target": camera.name, "anchors": [anchor.name for anchor in target.anchors],
            "c2w": camera.c2w.tolist(), "K": camera.K.tolist(), "width": camera.width, "height": camera.height,
            "fraction": target.fraction, "parent_checkpoint_sha256": model.checkpoint_sha256,
            "seeds": list(seeds), "training_arrays_sha256": sha256_file(arrays), **stats}


def _setup_round(model, cameras, dataset, output, arm, round_index, args):
    if arm == "real_only":
        return [], {"planning_seconds": 0., "generation_seconds": 0., "diffix": None, "observations": []}
    chosen, plan = _round_plan(model, cameras, output, round_index, args.views)
    manifest = output / "observations.json"
    if manifest.exists():
        record = _read_json(manifest)
        if (record["parent_checkpoint_sha256"] != model.checkpoint_sha256 or record["arm"] != arm
                or [item["target"] for item in record["observations"]] != plan["chosen"]):
            raise ValueError(f"Pseudo-observation receipt does not match the round plan: {manifest}")
        return [_observation_from_cache(output / item["target"], item) for item in record["observations"]], record

    observations, records = [], []
    restorer, difix = None, None
    for index, target in enumerate(chosen):
        target_dir = output / target.camera.name
        receipt = target_dir / "observation.json"
        seed = args.seed + round_index * 1000 + 2 * index
        seeds = (seed, seed + 1) if arm == "progressive_gated" else ()
        if receipt.exists():
            item = _read_json(receipt)
            if (item["target"] != target.camera.name or item["parent_checkpoint_sha256"] != model.checkpoint_sha256
                    or item["seeds"] != list(seeds) or item["arm"] != arm):
                raise ValueError(f"Cached view does not match the round plan: {receipt}")
        else:
            if seeds and restorer is None:
                restorer = DifixRestorer(model_dir=args.model_dir, dtype=args.precision,
                                         max_edge=args.max_edge, seed=args.seed)
            tick = time.perf_counter()
            item = _generate_view(model, target, dataset, target_dir, restorer, seeds)
            item.update(arm=arm, generation_seconds=time.perf_counter() - tick,
                        difix=restorer.provenance() if restorer is not None else None)
            write_json(receipt, item)
        observations.append(_observation_from_cache(target_dir, item))
        difix = item["difix"]
        records.append({key: value for key, value in item.items() if key != "difix"})
    if restorer is not None:
        import torch

        del restorer
        torch.cuda.empty_cache()
    record = {**plan, "arm": arm, "observations": records, "diffix": difix,
              "generation_seconds": sum(item["generation_seconds"] for item in records)}
    write_json(manifest, record)
    return observations, record


def _run_scene(args, name, dataset, scene, checkpoint, expected_sha256=None):
    scene_root = args.output / name
    cameras = training_cameras(dataset, scene)
    config = fusion.FusionConfig(steps=args.steps, seed=args.seed, synthetic_weight=args.synthetic_weight)
    config.validate()
    run = {**asdict(config), "views": args.views, "precision": args.precision, "max_edge": args.max_edge,
           "policy": "progressive", "implementation_sha256": _implementation_sha256(),
           "difix_model": MODEL_ID, "difix_revision": MODEL_REVISION}
    spec = scene_root / "protocol.json"
    if spec.exists():
        if {key: value for key, value in _read_json(spec).items() if key != "source"} != run:
            raise ValueError(f"Protocol differs from the recorded scene. Choose a new output: {scene_root}")
    else:
        write_json(spec, {**run, "source": source_version()})
    if expected_sha256 is None:
        m0 = fusion.load_part2_checkpoint(dataset, scene, checkpoint)
    else:
        m0 = fusion.load_local_checkpoint(dataset, scene, checkpoint, expected_sha256)
    baseline = scene_root / "baseline"
    start = time.perf_counter()
    metrics = _evaluate(m0, dataset, scene, baseline)
    metrics["train_frames"] = len(cameras)
    baseline_record = _stage_record(name, "sparse", 0, metrics, baseline, args.output, {}, time.perf_counter() - start)
    if not (baseline / "stage.json").exists():
        write_json(baseline / "stage.json", baseline_record)
    _write_comparison(args.output)

    for arm in args.arms:
        root = scene_root / arm
        if not (root / "round2/stage.json").exists():
            model = m0
            for round_index in (1, 2):
                output = root / f"round{round_index}"
                stage = output / "stage.json"
                if stage.exists():
                    model = fusion.load_fusion_checkpoint(output / "checkpoint.pt")
                    if _read_json(stage)["checkpoint_sha256"] != model.checkpoint_sha256:
                        raise ValueError(f"Completed stage checkpoint failed verification: {stage}")
                    continue
                print(f"{name} {arm} round {round_index} starting", flush=True)
                start = time.perf_counter()
                observations, receipt = _setup_round(model, cameras, dataset, output, arm, round_index, args)
                model = fusion.fine_tune(model, dataset, scene, output, ARMS[arm], observations, config)
                metrics = _evaluate(model, dataset, scene, output)
                metrics["train_frames"] = len(cameras)
                write_json(stage, _stage_record(name, arm, round_index, metrics, output, args.output,
                                                receipt, time.perf_counter() - start))
                _write_comparison(args.output)
                print(f"{name} {arm} round {round_index} PSNR {metrics['psnr']:.4f} "
                      f"SSIM {metrics['ssim']:.4f} LPIPS {metrics['lpips']:.4f}", flush=True)
        if args.keep_checkpoints != "all":
            (root / "round1/checkpoint.pt").unlink(missing_ok=True)
        if args.keep_checkpoints == "none" or (args.keep_checkpoints == "gated" and arm != "progressive_gated"):
            (root / "round2/checkpoint.pt").unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets-root", type=Path, help="Read-only root of published Part 2 course assets")
    parser.add_argument("--scene", choices=COURSE_SCENES, action="append")
    parser.add_argument("--supplemental", type=Path, help="JSON describing an independently trained M0")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--arms", choices=tuple(ARMS), nargs="+", default=list(ARMS))
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--views", type=int, default=8)
    parser.add_argument("--synthetic-weight", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--precision", choices=PRECISIONS, default="float16")
    parser.add_argument("--max-edge", type=int, default=768)
    parser.add_argument("--keep-checkpoints", choices=("all", "final", "gated", "none"), default="all")
    args = parser.parse_args()
    args.output = args.output.resolve()
    if args.assets_root is None and args.supplemental is None:
        parser.error("provide --assets-root or --supplemental")
    if args.views < 1 or args.max_edge < 8:
        parser.error("--views must be positive and --max-edge must be at least 8")
    if args.assets_root is not None:
        for name in args.scene or COURSE_SCENES:
            case = args.assets_root / "runs/part2_formal" / name / "full_sparse_7000_v1/sparse"
            _run_scene(args, name, args.assets_root / "data/part2_sparse" / name, case / "init/scene.npz",
                       case / "train/checkpoints/step_007000.pt")
    if args.supplemental is not None:
        supplemental = _read_json(args.supplemental)
        dataset = Path(supplemental["dataset"])
        _run_scene(args, load_manifest(dataset)["scene"], dataset, Path(supplemental["initializer"]),
                   Path(supplemental["checkpoint"]), supplemental["checkpoint_sha256"])


if __name__ == "__main__":
    main()

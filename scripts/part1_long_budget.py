"""Compare frozen Part 1 initializers under one longer fixed-pose 3DGS schedule."""

import argparse
import csv
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import shutil
from statistics import median
import subprocess
import sys
from tempfile import TemporaryDirectory
import time

import numpy as np

from p4.common import sha256_file, source_version, write_json
from p4.data import load_manifest, load_scene
from p4.evaluate import evaluate_renders
from p4.train import TrainConfig


SCENES = ("Dl3dv-2", "Re10k-1", "Waymo-405841")
METHODS = ("colmap", "vggt", "vggt-omega")
COMPARISON_STEP = 7000


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def preflight(dataset_root, archive, scenes, methods):
    """Verify the archived inputs, including matched real-camera sampler order."""
    cases = []
    historical_config = json.loads(json.dumps(asdict(TrainConfig())))
    for scene in scenes:
        dataset = dataset_root / scene
        archived_dataset = archive / "datasets" / scene
        manifest = load_manifest(dataset)
        digests = {name: sha256_file(dataset / name) for name in ("manifest.json", "reference.npz")}
        if any(digest != sha256_file(archived_dataset / name) for name, digest in digests.items()):
            raise ValueError(f"{scene}: use the original Part 1 replay manifest and reference arrays")
        for frame in manifest["frames"]:
            if sha256_file(dataset / frame["image"]) != frame["sha256"]:
                raise ValueError(f"{scene}: replay RGB hash mismatch for {frame['image']}")
        train_names = {frame["image"] for frame in manifest["frames"] if frame["split"] == "train"}
        holdout = [frame["image"] for frame in manifest["frames"] if frame["split"] == "test"]
        if len(holdout) < 3:
            raise ValueError(f"{scene}: at least three held-out views are required")
        matched_names = None
        for method in methods:
            prior = archive / "cases" / scene / method
            initializer = prior / "init" / "scene.npz"
            provenance = read_json(prior / "train" / "provenance.json")
            config = read_json(prior / "train" / "config.json")
            if config != historical_config:
                raise ValueError(f"{scene}/{method}: historical Part 1 training config differs")
            initializer_hash = sha256_file(initializer)
            if initializer_hash != provenance["scene_sha256"] or digests["manifest.json"] != provenance["manifest_sha256"]:
                raise ValueError(f"{scene}/{method}: archived initializer or manifest fingerprint differs")
            names = load_scene(initializer)["names"].tolist()
            if set(names) != train_names or (matched_names is not None and names != matched_names):
                raise ValueError(f"{scene}/{method}: training camera membership or order differs")
            matched_names = names
            original = read_json(prior / "metrics" / "step_007000.json")
            if original["split_hash"] != manifest["split_hash"] or [f["image"] for f in original["per_frame"]] != holdout:
                raise ValueError(f"{scene}/{method}: original holdout differs")
            cases.append({
                "scene": scene, "method": method, "dataset": str(dataset), "initializer": str(initializer),
                "initializer_sha256": initializer_hash, "original_training_source": provenance["git_commit"],
                "manifest_sha256": digests["manifest.json"], "reference_sha256": digests["reference.npz"],
                "split_hash": manifest["split_hash"], "train_frames": len(names), "test_frames": len(holdout),
                "train_order_sha256": hashlib.sha256(json.dumps(names).encode()).hexdigest(),
                "sample_images": [holdout[index] for index in (0, len(holdout) // 2, len(holdout) - 1)],
                "historical_7000": {metric: original[metric] for metric in ("psnr", "ssim", "lpips")},
            })
    return cases


def run_module(module, command, **flags):
    arguments = [sys.executable, "-m", module, command]
    for name, value in flags.items():
        arguments.extend((f"--{name}", str(value)))
    subprocess.run(arguments, check=True)


def train_windows(log, log_every):
    """Summarize observed training losses in fixed 1,000-step windows."""
    windows = []
    samples_per_window = 1000 // log_every
    for offset in range(0, len(log), samples_per_window):
        window = log[offset:offset + samples_per_window]
        windows.append({
            "end_step": window[-1]["step"],
            "logged_train_samples": len(window),
            "median_train_loss": median(row["loss"] for row in window),
        })
    return windows


def verify_completed(destination, case):
    record = read_json(destination / "record.json")
    if record["case"] != case:
        raise ValueError(f"Completed case identity differs at {destination}")
    for path, digest in record["artifact_sha256"].items():
        if sha256_file(destination / path) != digest:
            raise ValueError(f"Completed artifact changed: {destination / path}")
    return record


def run_case(case, config, output, scratch, keep_final):
    case_path = Path(case["scene"]) / case["method"]
    destination, work = output / case_path, scratch / case_path
    if not work.resolve().is_relative_to(scratch.resolve()):
        raise ValueError(f"Scratch case escapes experiment scratch: {work}")
    training = work / "train"
    checkpoints = [training / "checkpoints" / f"step_{step:06d}.pt" for step in config.save_steps]
    if (destination / "record.json").is_file():
        record = verify_completed(destination, case)
        if keep_final and sha256_file(checkpoints[-1]) != record["rows"][-1]["checkpoint_sha256"]:
            raise ValueError(f"Retained checkpoint changed at {checkpoints[-1]}")
    else:
        work.mkdir(parents=True, exist_ok=True)
        dataset, initializer = Path(case["dataset"]), Path(case["initializer"])
        config_path = work / "config.json"
        write_json(config_path, asdict(config))
        if not checkpoints[-1].is_file():
            resume = {"resume": checkpoints[0]} if checkpoints[0].is_file() else {}
            if not resume and (training / "config.json").exists():
                training.rename(work / f"interrupted_before_checkpoint_{time.time_ns()}")
            run_module("p4.train", "train", dataset=dataset, scene=initializer,
                       output=training, config=config_path, **resume)
        if read_json(training / "config.json") != read_json(config_path):
            raise ValueError(f"Scratch training config differs at {training}")

        destination.mkdir(parents=True, exist_ok=True)
        artifacts = []
        for name in ("provenance.json", "train.jsonl"):
            shutil.copyfile(training / name, destination / name)
            artifacts.append(Path(name))
        log = [json.loads(line) for line in (destination / "train.jsonl").read_text(encoding="utf-8").splitlines()]
        if [row["step"] for row in log] != list(range(config.log_every, config.steps + 1, config.log_every)):
            raise ValueError(f"Incomplete or duplicate training steps in {training}")
        rows = []
        for step, checkpoint in zip(config.save_steps, checkpoints):
            stage = destination / f"step_{step:06d}"
            with TemporaryDirectory(prefix="renders_", dir=work) as render_dir:
                renders = Path(render_dir)
                run_module("p4.train", "render", dataset=dataset, scene=initializer,
                           checkpoint=checkpoint, output=renders)
                render = read_json(renders / "render.json")
                metrics = evaluate_renders(dataset, renders)
                if not (render["step"] == step and render["raw_gaussian_renders"]
                        and not render["test_pose_refinement"] and metrics["complete"]
                        and metrics["test_frames"] == case["test_frames"]
                        and metrics["split_hash"] == case["split_hash"]):
                    raise ValueError(f"Evaluation is incomplete or inconsistent for {stage}")
                write_json(stage / "metrics.json", {**metrics, "render": render})
                artifacts.append((stage / "metrics.json").relative_to(destination))
                for sample in case["sample_images"]:
                    name = Path(sample).with_suffix(".png")
                    target = stage / "samples" / name
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(renders / name, target)
                    artifacts.append(target.relative_to(destination))
            training_row = log[step // config.log_every - 1]
            rows.append({
                "scene": case["scene"], "method": case["method"], "step": step,
                **{key: metrics[key] for key in ("test_frames", "split_hash", "psnr", "ssim", "lpips")},
                **{key: training_row[key] for key in ("num_gaussians", "training_seconds", "peak_cuda_gib")},
                "ate_rmse_reference_units": render["ate_rmse"],
                "checkpoint_sha256": render["checkpoint_sha256"],
            })
        record = {
            "case": case, "rows": rows, "train_loss_windows": train_windows(log, config.log_every),
            "artifact_sha256": {path.as_posix(): sha256_file(destination / path) for path in artifacts},
        }
        write_json(destination / "record.json", record)

    for path in checkpoints[:-1] if keep_final else checkpoints:
        path.unlink(missing_ok=True)
    return record["rows"]


def write_comparison(output, rows):
    draft = output / "comparison.csv.tmp"
    with draft.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    draft.replace(output / "comparison.csv")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets-root", type=Path, required=True, help="Exact Part 1 replay datasets")
    parser.add_argument("--archive-root", type=Path, required=True, help="Extracted original Part 1 handoff")
    parser.add_argument("--output", type=Path, required=True, help="Persistent compact results")
    parser.add_argument("--scratch", type=Path, required=True, help="Private scratch root for active checkpoints")
    parser.add_argument("--scenes", nargs="+", choices=SCENES, default=SCENES)
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=METHODS)
    parser.add_argument("--steps", type=int, default=30000)
    parser.add_argument("--check-inputs", action="store_true", help="Validate inputs without creating outputs or using CUDA")
    parser.add_argument("--keep-final", action="store_true", help="Retain final checkpoints under scratch")
    args = parser.parse_args()
    config = TrainConfig(steps=args.steps, save_steps=(COMPARISON_STEP, args.steps))
    if args.steps <= COMPARISON_STEP or args.steps % config.log_every:
        parser.error(f"--steps must exceed {COMPARISON_STEP} and be divisible by {config.log_every}")
    config.validate()
    cases = preflight(args.datasets_root.resolve(), args.archive_root.resolve(),
                      list(dict.fromkeys(args.scenes)), list(dict.fromkeys(args.methods)))
    if args.check_inputs:
        print(json.dumps(cases, indent=2))
        return

    output = args.output.resolve()
    scratch = args.scratch.resolve() / hashlib.sha256(str(output).encode()).hexdigest()[:16]
    if output == scratch or output.is_relative_to(scratch) or scratch.is_relative_to(output):
        parser.error("--output and the experiment scratch directory must be separate")
    protocol = {
        "config": json.loads(json.dumps(asdict(config))), "cases": cases,
        "source": source_version(), "driver_sha256": sha256_file(Path(__file__)),
        "keep_final_checkpoints": args.keep_final,
        "budget_description": "Fresh long schedule. The means learning-rate decay uses the final budget from step 1.",
        "evaluation": "Fixed 7,000-step intermediate and final endpoints, scored after training; no holdout selection.",
    }
    protocol_path = output / "protocol.json"
    if protocol_path.exists() and read_json(protocol_path) != protocol:
        raise ValueError(f"Existing experiment protocol differs at {protocol_path}; use a new output directory")
    write_json(protocol_path, protocol)
    rows = []
    for case in cases:
        rows.extend(run_case(case, config, output, scratch, args.keep_final))
        write_comparison(output, rows)


if __name__ == "__main__":
    main()

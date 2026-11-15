"""Run and summarize the nine Part 1 conditions in isolated subprocesses."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import time

from p4.common import sha256_file, source_version, write_json


def conditions(config):
    return [(scene, method) for scene in config["scenes"] for method in config["methods"]]


def signature(config, dataset, method=None, pilot_steps=None):
    manifest = json.loads((dataset / "manifest.json").read_text(encoding="utf-8"))
    for frame in manifest["frames"]:
        actual = sha256_file(dataset / frame["image"])
        if actual != frame.get("sha256"):
            raise ValueError(f"Prepared image changed: {frame['image']}; prepare a fresh dataset")
    assets = {}
    for current in ([method] if method else config["methods"]):
        if current == "colmap":
            continue
        weights = Path(config["initialization"][current.replace("-", "_") + "_weights"])
        assets[current] = sha256_file(weights) if weights.is_file() else "missing"
    payload = {"config": config, "manifest": sha256_file(dataset / "manifest.json"),
               "source": source_version()["package_source_sha256"], "weights": assets, "pilot_steps": pilot_steps,
               "reference": sha256_file(dataset / "reference.npz") if (dataset / "reference.npz").exists() else None}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def stage_outputs(stage, run, config, dataset, pilot_steps):
    """Expected files and semantic checks before a successful stage receipt."""
    if stage == "initialize":
        from p4.data import load_scene
        load_scene(run / "init/scene.npz")
        metadata = json.loads((run / "init/scene.json").read_text(encoding="utf-8"))
        if metadata.get("status") != "completed" or metadata.get("manifest_sha256") != sha256_file(dataset / "manifest.json"):
            raise ValueError("Initializer metadata incomplete or mismatched")
        return [run / "init/scene.npz", run / "init/scene.json"]
    if stage == "train":
        steps = [pilot_steps] if pilot_steps else sorted(set(config["training"]["save_steps"] + [config["training"]["steps"]]))
        return [run / "train/checkpoints" / f"step_{step:06d}.pt" for step in steps] + [run / "train/provenance.json", run / "train/config.json", run / "train/train.jsonl"]
    step = int(stage.split("_")[-1])
    if stage.startswith("render_"):
        render = run / "renders" / f"step_{step:06d}"
        metadata = json.loads((render / "render.json").read_text(encoding="utf-8"))
        if metadata.get("step") != step or not metadata.get("raw_gaussian_renders"):
            raise ValueError("Render metadata incomplete")
        manifest = json.loads((dataset / "manifest.json").read_text(encoding="utf-8"))
        return [render / "render.json"] + [render / Path(f["image"]).with_suffix(".png") for f in manifest["frames"] if f["split"] == "test"]
    path = run / "metrics" / f"step_{step:06d}.json"
    metrics = json.loads(path.read_text(encoding="utf-8"))
    if not metrics.get("complete") or any(not isinstance(metrics.get(k), (int, float)) or not math.isfinite(metrics[k]) for k in ["psnr", "ssim", "lpips"]):
        raise ValueError("Mandatory PSNR/SSIM/LPIPS evaluation incomplete; check LPIPS model availability")
    return [path]


def stage_receipt_valid(receipt, command, identity):
    try:
        data = json.loads(receipt.read_text(encoding="utf-8"))
        # Interpreter locations and path separators differ after a server sync.
        # Exact source/input/output hashes, rather than the local venv path,
        # determine whether the result is the same frozen experiment.
        def canonical(tokens):
            return ["<python>"] + [str(token).replace("\\", "/") for token in tokens[1:]] if tokens else []
        if canonical(data.get("command", [])) != canonical(command) or data.get("signature") != identity or not data.get("artifacts"):
            return False
        return all(Path(path.replace("\\", "/")).is_file() and sha256_file(path.replace("\\", "/")) == digest for path, digest in data["artifacts"].items())
    except (OSError, ValueError):
        return False


def stage_commands(config, scene, method, run, pilot_steps=None):
    dataset = Path(config["dataset_root"]) / scene
    init = config["initialization"]
    cmd = [sys.executable, "-m", "p4.initialize", "--dataset", str(dataset), "--output", str(run / "init"),
           "--method", method, "--seed", str(config["training"]["seed"]),
           "--max-points", str(init["max_points"]), "--confidence-quantile", str(init["confidence_quantile"])]
    if method == "colmap":
        cmd += ["--matcher", init["matcher"]]
    else:
        key = method.replace("-", "_")
        cmd += ["--repo", init[key + "_repo"], "--weights", init[key + "_weights"]]
        if method == "vggt" and init.get("vggt_offload_features", False):
            cmd.append("--vggt-offload-features")
    commands = [("initialize", cmd)]
    train = [sys.executable, "-m", "p4.train", "train", "--dataset", str(dataset),
             "--scene", str(run / "init/scene.npz"), "--output", str(run / "train"),
             "--config", str(run / "frozen_config.json")]
    if pilot_steps:
        train += ["--stop-after", str(pilot_steps)]
    commands.append(("train", train))
    steps = ([pilot_steps] if pilot_steps else sorted(set(config["training"]["save_steps"] + [config["training"]["steps"]])))
    for step in steps:
        render = run / "renders" / f"step_{step:06d}"
        commands.append((f"render_{step}", [sys.executable, "-m", "p4.train", "render", "--dataset", str(dataset),
                         "--scene", str(run / "init/scene.npz"), "--checkpoint",
                         str(run / "train/checkpoints" / f"step_{step:06d}.pt"), "--output", str(render)]))
        commands.append((f"evaluate_{step}", [sys.executable, "-m", "p4.evaluate", "renders", "--dataset", str(dataset),
                         "--render-dir", str(render), "--output", str(run / "metrics" / f"step_{step:06d}.json")]))
    return commands


def run_matrix(config_path, scenes=None, methods=None, dry_run=False, pilot_steps=None):
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    if pilot_steps is not None and not 0 < pilot_steps <= config["training"]["steps"]:
        raise ValueError("pilot-steps must lie between 1 and the frozen training budget")
    failures = 0
    for scene, method in conditions(config):
        if scenes and scene not in scenes or methods and method not in methods:
            continue
        root = Path(config["output_root"])
        if pilot_steps:
            root = root.parent / (root.name + "_pilot")
        run = root / scene / method
        commands = stage_commands(config, scene, method, run, pilot_steps)
        if dry_run:
            print(f"\n{scene} / {method} [{ 'pilot' if pilot_steps else 'formal' }]")
            for stage, cmd in commands:
                print(stage + ": " + subprocess.list2cmdline(cmd))
            continue
        dataset = Path(config["dataset_root"]) / scene
        if not (dataset / "manifest.json").exists():
            run.mkdir(parents=True, exist_ok=True)
            write_json(run / "status.json", {"status": "pending", "reason": "Prepared dataset missing", "scene": scene, "method": method})
            print(f"PENDING {scene}/{method}: prepare dataset first", flush=True)
            failures += 1
            continue
        identity = signature(config, dataset, method, pilot_steps)
        if (run / "status.json").exists():
            old = json.loads((run / "status.json").read_text(encoding="utf-8"))
            if old.get("signature") and old["signature"] != identity:
                raise ValueError(f"Frozen protocol or inputs changed for {run}; use a new output_root")
            if old.get("status") == "completed" and all(stage_receipt_valid(run / "logs" / (s + ".done.json"), c, identity) for s, c in commands):
                print(f"COMPLETED {scene}/{method}; verified frozen signature, skipping", flush=True)
                continue
        run.mkdir(parents=True, exist_ok=True)
        write_json(run / "frozen_config.json", config)
        status = {"status": "running", "scene": scene, "method": method, "signature": identity,
                  "pilot": bool(pilot_steps), "steps": pilot_steps or config["training"]["steps"],
                  "provenance": source_version(), "stages": []}
        tick = time.perf_counter()
        upstream_changed = False
        try:
            for stage, cmd in commands:
                status["stage"] = stage
                write_json(run / "status.json", status)
                log_path = run / "logs" / (stage + ".log")
                log_path.parent.mkdir(exist_ok=True)
                stage_tick = time.perf_counter()
                # Only reuse stages whose exact command completed successfully before.
                receipt = log_path.with_suffix(".done.json")
                if not upstream_changed and stage_receipt_valid(receipt, cmd, identity):
                    status["stages"].append({"stage": stage, "reused": True})
                    continue
                upstream_changed = True
                # Preserve failed work rather than mixing artifacts on retries.
                # Trained state can instead be resumed explicitly with p4.train.
                stage_output = run / ("init" if stage == "initialize" else "train") if stage in {"initialize", "train"} else None
                if stage_output is not None and stage_output.exists() and any(stage_output.iterdir()):
                    archive = run / "attempts" / (time.strftime("%Y%m%d_%H%M%S") + "_" + stage)
                    archive.parent.mkdir(exist_ok=True)
                    stage_output.rename(archive)
                print(f"RUN {scene}/{method} {stage} -> {log_path}", flush=True)
                with log_path.open("w", encoding="utf-8") as log:
                    log.write(subprocess.list2cmdline(cmd) + "\n")
                    log.flush()
                    result = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, check=False)
                if result.returncode:
                    raise RuntimeError(f"{stage} exited {result.returncode}; see {log_path}")
                artifacts = stage_outputs(stage, run, config, dataset, pilot_steps)
                if any(not p.is_file() or not p.stat().st_size for p in artifacts):
                    raise RuntimeError(f"Missing or empty artifacts after {stage}")
                elapsed = time.perf_counter() - stage_tick
                write_json(receipt, {"command": cmd, "seconds": elapsed, "signature": identity,
                                     "artifacts": {p.as_posix(): sha256_file(p) for p in artifacts}})
                status["stages"].append({"stage": stage, "seconds": elapsed})
            final_step = pilot_steps or config["training"]["steps"]
            metrics = json.loads((run / "metrics" / f"step_{final_step:06d}.json").read_text(encoding="utf-8"))
            if metrics.get("lpips") is None or not metrics.get("complete"):
                raise RuntimeError("LPIPS unavailable: metrics incomplete; install metric weights and rerun evaluation")
            status["status"] = "completed"
        except (RuntimeError, OSError, ValueError) as error:
            status["status"], status["reason"] = "failed", str(error)
            print(f"FAILED {scene}/{method}: {error}", flush=True)
            failures += 1
        finally:
            status["wall_seconds_this_attempt"] = time.perf_counter() - tick
            write_json(run / "status.json", status)
    return failures


def summarize(config_path, output):
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    root, output = Path(config["output_root"]), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for scene, method in conditions(config):
        run = root / scene / method
        row = dict(scene=scene, method=method, status="pending", psnr=None, ssim=None, lpips=None,
                   ate_rmse=None, initialization_seconds=None, training_seconds=None,
                   initial_gaussians=None, final_gaussians=None, peak_cuda_gib=None, reason="Not run")
        if (run / "status.json").exists():
            status = json.loads((run / "status.json").read_text(encoding="utf-8"))
            row.update(status=status["status"], reason=status.get("reason", ""))
        valid = False
        if row["status"] == "completed":
            try:
                identity = signature(config, Path(config["dataset_root"]) / scene, method)
                commands = stage_commands(config, scene, method, run)
                valid = status.get("signature") == identity and all(stage_receipt_valid(run / "logs" / (s + ".done.json"), c, identity) for s, c in commands)
            except (OSError, ValueError):
                valid = False
            if not valid:
                row.update(status="pending", reason="Completed artifacts missing, modified or protocol changed; rerun verification")
        if not valid:
            rows.append(row)
            continue
        metric_path = run / "metrics" / f"step_{config['training']['steps']:06d}.json"
        if metric_path.exists():
            metrics = json.loads(metric_path.read_text(encoding="utf-8"))
            row.update({k: metrics.get(k) for k in ["psnr", "ssim", "lpips"]})
        if (run / "init/scene.json").exists():
            metadata = json.loads((run / "init/scene.json").read_text(encoding="utf-8"))
            row["initialization_seconds"] = metadata.get("total_seconds", metadata.get("initialization_seconds"))
        if (run / "train/provenance.json").exists():
            provenance = json.loads((run / "train/provenance.json").read_text(encoding="utf-8"))
            row["initial_gaussians"] = provenance.get("initial_gaussians")
        if (run / "train/train.jsonl").exists():
            log = [json.loads(line) for line in (run / "train/train.jsonl").read_text().splitlines() if line]
            if log:
                row.update(training_seconds=log[-1]["training_seconds"], final_gaussians=log[-1]["num_gaussians"],
                           peak_cuda_gib=max(r["peak_cuda_gib"] for r in log))
        render_meta = run / "renders" / f"step_{config['training']['steps']:06d}" / "render.json"
        if render_meta.exists():
            meta = json.loads(render_meta.read_text(encoding="utf-8"))
            row["ate_rmse"] = meta.get("ate_rmse", meta.get("ate_rmse_reference_units"))
        rows.append(row)
    with (output / "coverage.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    write_json(output / "coverage.json", rows)
    completed = sum(r["status"] == "completed" for r in rows)
    text = [f"# Part 1 actual coverage: {completed}/{len(rows)} completed", "",
            "Pending and failed entries are not measured results. Never fill gaps using historical project numbers.", "",
            "| Scene | Initializer | Status | PSNR | SSIM | LPIPS | Reason |",
            "| --- | --- | --- | --- | --- | --- | --- |"]
    for row in rows:
        text.append("| " + " | ".join(str(row[k]) if row[k] is not None else "pending" for k in
                    ["scene", "method", "status", "psnr", "ssim", "lpips", "reason"]) + " |")
    (output / "coverage.md").write_text("\n".join(text) + "\n", encoding="utf-8")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return rows
    for scene in config["scenes"]:
        figure, axis = plt.subplots(figsize=(6, 4))
        count = 0
        for method in config["methods"]:
            if not any(row["scene"] == scene and row["method"] == method and row["status"] == "completed" for row in rows):
                continue
            pairs = []
            for path in sorted((root / scene / method / "metrics").glob("step_*.json")):
                metric = json.loads(path.read_text(encoding="utf-8"))
                if metric.get("psnr") is not None:
                    pairs.append((int(path.stem.split("_")[-1]), metric["psnr"]))
            if pairs:
                axis.plot(*zip(*pairs), marker="o", label=method)
                count += 1
        if count:
            axis.set(xlabel="Completed optimization steps", ylabel="Test PSNR (dB)", title=scene)
            axis.legend()
            figure.tight_layout()
            figure.savefig(output / (scene + "_convergence.png"), dpi=180)
        plt.close(figure)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest="command", required=True)
    run = subs.add_parser("run")
    run.add_argument("--config", default="configs/part1.json")
    run.add_argument("--scenes", nargs="+")
    run.add_argument("--methods", nargs="+")
    run.add_argument("--dry-run", action="store_true")
    run.add_argument("--pilot-steps", type=int)
    summary = subs.add_parser("summarize")
    summary.add_argument("--config", default="configs/part1.json")
    summary.add_argument("--output", default="results/part1")
    args = parser.parse_args()
    if args.command == "run":
        raise SystemExit(bool(run_matrix(args.config, args.scenes, args.methods, args.dry_run, args.pilot_steps)))
    summarize(args.config, args.output)


if __name__ == "__main__":
    main()

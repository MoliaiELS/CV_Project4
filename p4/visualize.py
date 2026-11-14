"""Build same-view Part 1 panels from verified completed formal experiments.

Missing experiments receive text tiles, never invented renders or scores. If
the prepared dataset is absent, no comparison image is produced for that scene.
Trajectories use a single training-camera Sim3 and reference trajectory units.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import textwrap

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

from p4.common import sha256_file, write_json
from p4.data import load_manifest, load_scene
from p4.evaluate import evaluation_cameras
from p4.experiment import signature, stage_commands, stage_receipt_valid
from p4.geometry import transform_points


METHODS = ("colmap", "vggt", "vggt-omega")
LABELS = {"colmap": "COLMAP + 3DGS", "vggt": "VGGT + 3DGS", "vggt-omega": "VGGT-Omega + 3DGS"}
COLORS = {"colmap": "#3f73b4", "vggt": "#d8812d", "vggt-omega": "#3e9567"}


def _font(size):
    for path in ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "C:/Windows/Fonts/segoeui.ttf"):
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            pass
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def _verified_run(config, scene, method, dataset, run):
    """Use the same frozen-input and artifact receipts as the matrix runner."""
    if method not in config["methods"]:
        return False, "pending: method is not configured"
    try:
        status = json.loads((run / "status.json").read_text(encoding="utf-8"))
    except FileNotFoundError:
        return False, "pending: experiment has not run"
    except (OSError, ValueError) as exc:
        return False, f"pending: unreadable status ({type(exc).__name__})"
    if status.get("status") != "completed":
        state = status.get("status", "pending")
        return False, f"{state}: {status.get('reason', 'no verified complete result')}"
    if status.get("pilot") or status.get("steps") != config["training"]["steps"]:
        return False, "pending: only a pilot or different training budget is available"
    try:
        identity = signature(config, dataset, method)
        if status.get("signature") != identity:
            return False, "pending: frozen protocol, source, weights or data changed"
        for stage, command in stage_commands(config, scene, method, run):
            if not stage_receipt_valid(run / "logs" / (stage + ".done.json"), command, identity):
                return False, f"pending: {stage} artifacts are missing, modified or unverified"
    except (OSError, ValueError, KeyError) as exc:
        return False, f"pending: verification failed ({type(exc).__name__}: {exc})"
    return True, "completed: frozen signature and all artifact hashes verified"


def _selected_frames(manifest, max_frames):
    frames = [f for f in manifest["frames"] if f["split"] == "test"]
    if not frames:
        raise ValueError("No held-out test frames in prepared manifest")
    indices = np.linspace(0, len(frames) - 1, min(max_frames, len(frames)), dtype=int)
    return [frames[i] for i in indices]


def _checked_rgb(dataset, frame):
    path = dataset / frame["image"]
    if sha256_file(path) != frame.get("sha256"):
        raise ValueError(f"Prepared held-out RGB changed: {frame['image']}")
    with Image.open(path) as source:
        image = source.convert("RGB")
    if image.size != (frame["width"], frame["height"]):
        raise ValueError(f"Manifest image dimensions differ: {frame['image']}")
    return image


def _tile(image, size):
    # Only display thumbnails are resized; experiment renders and metric inputs
    # remain untouched. Every column of a row uses the same thumbnail transform.
    return ImageOps.pad(image, size, method=Image.Resampling.LANCZOS, color="#111924", centering=(.5, .5))


def _comparison(config, scene, dataset, frames, methods, destination, tile_width):
    sources = [_checked_rgb(dataset, frame) for frame in frames]
    heights = [max(96, min(420, round(tile_width * image.height / image.width))) for image in sources]
    gap, margin, header, row_header, footer = 10, 20, 86, 30, 48
    width = 2 * margin + 4 * tile_width + 3 * gap
    height = header + sum(h + row_header + gap for h in heights) + footer
    canvas = Image.new("RGB", (width, height), "#f1f4f7")
    draw = ImageDraw.Draw(canvas)
    draw.text((margin, 12), f"{scene} | held-out views | {config['training']['steps']} optimization steps",
              fill="#152238", font=_font(22))
    headings = ("Held-out RGB (GT)", *(LABELS[method] for method in METHODS))
    for col, label in enumerate(headings):
        draw.text((margin + col * (tile_width + gap), 50), label, fill="#27374d", font=_font(18))
    top = header
    for frame, source, tile_height in zip(frames, sources, heights):
        identifier = str(frame["id"])
        draw.text((margin, top + 2), "Frame: " + (identifier if len(identifier) <= 110 else identifier[:107] + "..."),
                  fill="#41556c", font=_font(15))
        top += row_header
        canvas.paste(_tile(source, (tile_width, tile_height)), (margin, top))
        for col, method in enumerate(METHODS, 1):
            left = margin + col * (tile_width + gap)
            valid, reason = methods[method]["verified"], methods[method]["reason"]
            if valid:
                render = Path(config["output_root"]) / scene / method / "renders" / f"step_{config['training']['steps']:06d}" / Path(frame["image"]).with_suffix(".png")
                with Image.open(render) as raw:
                    image = raw.convert("RGB")
                if image.size != source.size:
                    raise ValueError(f"Verified render resolution differs from held-out RGB: {render}")
                canvas.paste(_tile(image, (tile_width, tile_height)), (left, top))
            else:
                draw.rectangle((left, top, left + tile_width - 1, top + tile_height - 1), fill="#e1e7ee", outline="#cbd5e1")
                state = reason.split(":", 1)[0].upper()
                draw.text((left + 16, top + 15), state, fill="#5a6778", font=_font(22))
                words = textwrap.wrap(reason.split(":", 1)[-1].strip(), width=max(20, (tile_width - 32) // 8))
                for line_no, line in enumerate(words[:max(1, (tile_height - 52) // 20)]):
                    draw.text((left + 16, top + 49 + line_no * 20), line, fill="#5a6778", font=_font(14))
        top += tile_height + gap
    draw.text((margin, height - 35), "Same held-out camera per row. Display thumbnails only; no restoration, exposure fitting or test-pose refinement.",
              fill="#41556c", font=_font(14))
    canvas.save(destination)


def _trajectory(config, scene, dataset, manifest, methods, destination):
    available = [method for method in METHODS if methods[method]["verified"]]
    if not available:
        return "skipped: no verified completed reconstruction"
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return "skipped: matplotlib is not installed"
    try:
        with np.load(dataset / "reference.npz", allow_pickle=False) as reference:
            ref = dict(zip(reference["names"].astype(str), reference["c2w"]))
        names = [f["image"] for f in manifest["frames"] if f["split"] == "train" and f["image"] in ref]
        if len(names) < 3:
            return "skipped: fewer than three train reference cameras"
        centers = np.stack([ref[name][:3, 3] for name in names])
        aligned = {}
        notes = []
        for method in available:
            initializer = Path(config["output_root"]) / scene / method / "init/scene.npz"
            cameras = evaluation_cameras(dataset, initializer)
            alignment = cameras["alignment"]
            estimated = load_scene(initializer)
            lookup = dict(zip(estimated["names"].astype(str), estimated["c2w"]))
            matched = [name for name in names if name in lookup]
            aligned[method] = (transform_points(np.stack([lookup[name][:3, 3] for name in matched]),
                                                alignment["estimate_to_reference"]), alignment["ate_rmse"])
        if not aligned:
            return "skipped: train-reference trajectory alignment unavailable"
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
        dimensions = ((0, 1, "X", "Y"), (0, 2, "X", "Z"), (1, 2, "Y", "Z"))
        for axis, (a, b, label_a, label_b) in zip(axes, dimensions):
            axis.plot(centers[:, a], centers[:, b], color="#172d44", linewidth=2.2, label="Reference (train)")
            for method, (points, ate) in aligned.items():
                axis.plot(points[:, a], points[:, b], color=COLORS[method], linewidth=1.3,
                          label=f"{method}: ATE {ate:.4g}")
            axis.set(xlabel=f"{label_a} (reference units)", ylabel=f"{label_b} (reference units)")
            axis.set_aspect("equal", adjustable="datalim")
            axis.grid(alpha=.2)
        axes[0].legend(fontsize=8)
        fig.suptitle(f"{scene} | train trajectories | one train-derived Sim3 per method", fontsize=12)
        fig.text(.5, .015, f"Reference kind: {manifest.get('reference_kind', 'reference')}. ATE is metric only if reference coordinates are metric.",
                 ha="center", fontsize=9)
        fig.tight_layout(rect=(0, .05, 1, .93))
        fig.savefig(destination, dpi=180)
        plt.close(fig)
        return "created: train-only global Sim3; units inherited from reference"
    except (OSError, ValueError, KeyError) as exc:
        return f"skipped: {type(exc).__name__}: {exc}"


def create_visualizations(config_path="configs/part1.json", output="results/part1", max_frames=3, tile_width=384):
    if not 1 <= max_frames <= 3 or not 160 <= tile_width <= 1024:
        raise ValueError("Require 1<=max_frames<=3 and 160<=tile_width<=1024")
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    summary = {"schema_version": 1, "formal_only": True, "max_frames_per_scene": max_frames,
               "selection": "manifest test order; evenly spaced first/middle/last indices",
               "display": "same camera per row; source aspect ratio preserved; raw files untouched", "scenes": []}
    for scene in config["scenes"]:
        # Exact known artifact names only; remove old derived pictures when a
        # previously complete run no longer verifies. No recursive deletion.
        slug = re.sub(r"[^A-Za-z0-9_.-]", "_", scene)
        panel, trajectory = output / f"{slug}_comparison.png", output / f"{slug}_trajectory.png"
        panel.unlink(missing_ok=True)
        trajectory.unlink(missing_ok=True)
        dataset = Path(config["dataset_root"]) / scene
        entry = {"scene": scene, "comparison": None, "trajectory": None, "methods": {}}
        try:
            manifest = load_manifest(dataset)
            frames = _selected_frames(manifest, max_frames)
            for method in METHODS:
                run = Path(config["output_root"]) / scene / method
                valid, reason = _verified_run(config, scene, method, dataset, run)
                entry["methods"][method] = {"verified": valid, "reason": reason}
            _comparison(config, scene, dataset, frames, entry["methods"], panel, tile_width)
            entry.update(comparison=str(panel), selected_frame_ids=[f["id"] for f in frames],
                         selected_images=[f["image"] for f in frames])
            entry["trajectory_note"] = _trajectory(config, scene, dataset, manifest, entry["methods"], trajectory)
            if trajectory.exists():
                entry["trajectory"] = str(trajectory)
            if not any(m["verified"] for m in entry["methods"].values()):
                entry["note"] = "No verified reconstruction results. Panel contains actual held-out RGB and pending/failed text tiles only."
        except (OSError, ValueError, KeyError) as exc:
            entry["note"] = f"skipped: {type(exc).__name__}: {exc}"
        summary["scenes"].append(entry)
    write_json(output / "visualizations.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/part1.json")
    parser.add_argument("--output", default="results/part1")
    parser.add_argument("--max-frames", type=int, default=3)
    parser.add_argument("--tile-width", type=int, default=384)
    args = parser.parse_args()
    summary = create_visualizations(args.config, args.output, args.max_frames, args.tile_width)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

"""Build paper figures and optional held-out demo videos from completed Part 3 runs."""

import argparse
import csv
from dataclasses import dataclass
import json
from pathlib import Path
import shutil
import subprocess

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

from p4.common import sha256_file, write_json
from p4.data import load_manifest
from scripts.part3_views import training_cameras


COURSE_SCENES = ("Re10k-1", "Dl3dv-2", "Waymo-405841")
METRICS = ("psnr", "ssim", "lpips")
ARMS = {
    "real_only": ("Real only", "#345e80", "--", "s"),
    "raw_replay": ("Raw replay", "#78658b", ":", "v"),
    "ungated": ("Ungated", "#be5c20", "-.", "^"),
    "fixed_gated": ("Fixed + gated", "#696969", "--", "D"),
    "gated": ("Active + gated", "#0b725d", "-", "o"),
}
HELD_OUT = (("sparse", 0, "$M_0$"), ("real_only", 2, "Real only R2"),
            ("gated", 1, "Gated R1"), ("gated", 2, "Gated R2"))


@dataclass(frozen=True)
class Stage:
    directory: Path
    record: dict
    metrics: dict

    def render(self, name):
        return self.directory / "renders" / Path(name).with_suffix(".png")


def _json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _stages(root, selected):
    with (root / "comparison.csv").open(newline="", encoding="utf-8") as stream:
        comparison = csv.DictReader(stream)
        fields = comparison.fieldnames
        rows = [row for row in comparison if not selected or row["scene"] in selected]
    stages = {}
    for row in rows:
        key = (row["scene"], row["arm"], int(row["round"]))
        if key in stages:
            raise ValueError(f"Duplicate stage {key}")
        directory = root / row["path"]
        record = _json(directory / "stage.json")
        metrics = _json(directory / "metrics.json")
        if ((record["scene"], record["arm"], record["round"]) != key
                or metrics["split_hash"] != row["split_hash"]
                or metrics["checkpoint_sha256"] != row["checkpoint_sha256"]
                or not metrics["complete"]
                or any(float(row[name]) != metrics[name] or metrics[name] != record[name] for name in METRICS)
                or not np.isfinite([metrics[name] for name in METRICS]).all()):
            raise ValueError(f"Metrics, stage record, and comparison disagree for {key}")
        if len(metrics["per_frame"]) != int(row["test_frames"]):
            raise ValueError(f"Incorrect held-out frame count for {key}")
        stages[key] = Stage(directory, record, metrics)
    found = {k[0] for k in stages}
    if not found or (selected and found != set(selected)):
        raise ValueError(f"Missing scene results: {sorted(set(selected or ()) - found)}")
    scenes = [name for name in COURSE_SCENES if name in found] + sorted(found - set(COURSE_SCENES))
    for scene in scenes:
        expected_arms = ("real_only", "raw_replay", "ungated", "gated") if scene in COURSE_SCENES else ("real_only", "gated")
        present_arms = {arm for name, arm, step in stages if name == scene and step}
        required = [(scene, "sparse", 0), *((scene, arm, step) for arm in sorted(set(expected_arms) | present_arms)
                                          for step in (1, 2))]
        absent = [key for key in required if key not in stages]
        if absent:
            raise ValueError(f"Run is incomplete: {absent}")
        baseline = stages[scene, "sparse", 0].metrics
        ids = [(f["id"], f["image"]) for f in baseline["per_frame"]]
        for (name, _, _), stage in stages.items():
            if name != scene:
                continue
            if stage.metrics["split_hash"] != baseline["split_hash"] or ids != [(f["id"], f["image"]) for f in stage.metrics["per_frame"]]:
                raise ValueError(f"Inconsistent held-out frames in {stage.directory}")
    return fields, rows, stages, scenes


def _scene_inputs(scenes, assets, supplemental):
    inputs = {}
    if supplemental:
        receipt = _json(supplemental)
        dataset = Path(receipt["dataset"])
        inputs[load_manifest(dataset)["scene"]] = dataset, Path(receipt["initializer"])
    for scene in scenes:
        if scene in COURSE_SCENES and assets is not None:
            root = assets / "runs/part2_formal" / scene / "full_sparse_7000_v1/sparse"
            inputs[scene] = assets / "data/part2_sparse" / scene, root / "init/scene.npz"
        if scene not in inputs:
            raise ValueError(f"No image or initializer source for {scene}; supply --assets-root or --supplemental")
    return inputs


def _plot_metric(ax, stages, scene, metric, *, change=False):
    handles = {}
    baseline = stages[scene, "sparse", 0].metrics[metric]
    for arm, (name, color, style, marker) in ARMS.items():
        if (scene, arm, 2) not in stages:
            continue
        values = np.asarray([baseline, *(stages[scene, arm, step].metrics[metric] for step in (1, 2))])
        if change:
            values -= baseline
        handles[arm], = ax.plot((0, 1, 2), values, color=color, linestyle=style, marker=marker,
                                label=name, linewidth=1.25, markersize=3.8)
    ax.set(xlim=(-.12, 2.12), xticks=[0, 1, 2], xticklabels=["M0", "R1", "R2"])
    ax.tick_params(axis="both", labelsize=7, length=2, color="#9aa2a7")
    ax.grid(axis="y", color="#e3e6e8", linewidth=.5)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#9aa2a7")
    ax.ticklabel_format(axis="y", style="plain", useOffset=False)
    return handles


def _metric_legend(figure, handles):
    figure.legend([handles[key] for key in ARMS if key in handles],
                  [handles[key].get_label() for key in ARMS if key in handles],
                  loc="upper center", bbox_to_anchor=(0.52, .993), ncols=len(handles),
                  fontsize=7.5, frameon=False, handletextpad=.5, columnspacing=1.2)


def _closed_loop(stages, scenes, destination):
    figure, axes = plt.subplots(len(scenes), 3, figsize=(7.1, 1.46 * len(scenes) + 0.7), squeeze=False)
    handles = {}
    for row, scene in enumerate(scenes):
        for col, (metric, label) in enumerate(zip(METRICS, ("PSNR (dB) ↑", "SSIM ↑", "LPIPS ↓"))):
            ax = axes[row, col]
            handles.update(_plot_metric(ax, stages, scene, metric))
            if row == 0:
                ax.set_title(label, fontsize=8.8, fontweight="bold", pad=4)
            if col == 0:
                ax.text(-0.38, 0.5, scene, rotation=90, transform=ax.transAxes, ha="center", va="center", fontsize=8)
    _metric_legend(figure, handles)
    height = figure.get_figheight()
    figure.subplots_adjust(left=.13, right=.99, bottom=max(.065, .26 / height), top=1 - .57 / height,
                           wspace=.44, hspace=.39)
    figure.savefig(destination, dpi=184, facecolor="white")
    plt.close(figure)


def _closed_loop_psnr(stages, scenes, destination):
    figure, axes = plt.subplots(1, len(scenes), figsize=(7.1, 1.98), squeeze=False)
    handles = {}
    for ax, scene in zip(axes[0], scenes):
        handles.update(_plot_metric(ax, stages, scene, "psnr", change=True))
        ax.axhline(0, color="#acb4ba", linewidth=.7, zorder=0)
        ax.set_title(scene, fontsize=8.5, fontweight="bold", pad=3)
    axes[0, 0].set_ylabel("Δ PSNR (dB)", fontsize=8)
    _metric_legend(figure, handles)
    figure.subplots_adjust(left=.125, right=.99, bottom=.16, top=.74, wspace=.46)
    figure.savefig(destination, dpi=184, facecolor="white")
    plt.close(figure)


def _frame_examples(stages, scenes):
    examples = []
    for scene in scenes:
        before = stages[scene, "sparse", 0].metrics["per_frame"]
        after = stages[scene, "gated", 2].metrics["per_frame"]
        for base, enhanced in zip(before, after):
            examples.append({"scene": scene, "id": base["id"], "image": base["image"],
                             **{f"delta_{m}": enhanced[m] - base[m] for m in METRICS}})
    return sorted(examples, key=lambda row: (row["delta_psnr"], row["scene"], str(row["id"])))


def _read_preview(path, edge=580, grayscale=False):
    with Image.open(path) as original:
        image = original.convert("L" if grayscale else "RGB")
        image.thumbnail((edge, edge), Image.Resampling.LANCZOS)
        return np.asarray(image)


def _heldout_figure(stages, inputs, examples, destination):
    figure, axes = plt.subplots(len(examples), 5, figsize=(7.12, 1.66 * len(examples)), squeeze=False)
    for row, example in enumerate(examples):
        scene, name = example["scene"], example["image"]
        inputs_for_row = [("Ground truth", inputs[scene][0] / name, None)]
        for arm, round_index, title in HELD_OUT:
            stage = stages[scene, arm, round_index]
            metrics = next(frame for frame in stage.metrics["per_frame"] if frame["image"] == name)
            inputs_for_row.append((title, stage.render(name), metrics["psnr"]))
        for col, (title, path, psnr) in enumerate(inputs_for_row):
            ax = axes[row, col]
            ax.imshow(_read_preview(path))
            ax.axis("off")
            suffix = f"  {psnr:.2f} dB" if psnr is not None else ""
            ax.set_title(title + suffix, fontsize=7.6, pad=3)
        axes[row, 0].text(0, -0.10, f"{scene} · {example['id']} · Gated R2 − M0 {example['delta_psnr']:+.2f} dB",
                          transform=axes[row, 0].transAxes, fontsize=7.6, ha="left", va="top")
    figure.subplots_adjust(left=.006, right=.994, bottom=.11, top=.93, wspace=.02, hspace=.37)
    figure.savefig(destination, dpi=185, facecolor="white")
    plt.close(figure)


def _pseudo_figure(stages, scene, destination):
    records = [(_json(stages[scene, "gated", round_index].directory / "observations.json"),
                stages[scene, "gated", round_index].directory) for round_index in (1, 2)]
    examples = []
    figure, axes = plt.subplots(2, 4, figsize=(7.12, 3.18), squeeze=False)
    for row, (record, directory) in enumerate(records):
        observations = record["observations"]
        # Illustrate confidence at both ends of the support range, using only
        # training evidence available during each round.
        order = sorted(observations, key=lambda v: (v["verified_fraction"], v["target"]))
        chosen = order[-1] if row == 0 else order[0]
        examples.append({"scene": scene, "round": row + 1, **chosen})
        for col, (label, filename) in enumerate(zip(("Training reference", "Raw virtual render", "Difix-ref output", "Confidence"),
                                                    ("reference.png", "raw.png", "enhanced.png", "confidence.png"))):
            ax = axes[row, col]
            preview = _read_preview(directory / chosen["target"] / filename, grayscale=col == 3)
            if col == 3:
                confidence = ax.imshow(preview / 255, cmap="viridis", vmin=0, vmax=1)
            else:
                ax.imshow(preview)
            ax.axis("off")
            if row == 0:
                ax.set_title(label, fontsize=8, pad=3)
        axes[row, 0].text(0, -.10, f"R{row + 1} · {chosen['target']} · {chosen['verified_fraction']:.0%} reprojected support",
                          transform=axes[row, 0].transAxes, fontsize=8, ha="left", va="top")
    scale = figure.add_axes((.805, .065, .13, .014))
    colorbar = figure.colorbar(confidence, cax=scale, orientation="horizontal", ticks=[0, 1])
    colorbar.ax.tick_params(labelsize=7.5, length=1.5, pad=1)
    colorbar.outline.set_linewidth(.4)
    figure.subplots_adjust(left=.006, right=.994, bottom=.115, top=.93, wspace=.025, hspace=.22)
    figure.savefig(destination, dpi=185, facecolor="white")
    plt.close(figure)
    return examples


def _trajectory_figure(stages, scenes, inputs, destination):
    figure, axes = plt.subplots(1, len(scenes), figsize=(7.12, 1.85), squeeze=False)
    handles = {}
    for ax, scene in zip(axes[0], scenes):
        cameras = training_cameras(*inputs[scene])
        centers = np.stack([camera.c2w[:3, 3] for camera in cameras])
        centroid = centers.mean(0)
        _, _, axes_3d = np.linalg.svd(centers - centroid, full_matrices=False)
        axes_2d = axes_3d[:2].T
        trajectory = (centers - centroid) @ axes_2d
        line, = ax.plot(*trajectory.T, "o-", label="Real anchors", color="#4b5861", markersize=2.2, linewidth=.8)
        handles["Real anchors"] = line
        marks = ([("fixed_gated", 1, "Fixed", "D", "#858585")] if (scene, "fixed_gated", 1) in stages else [])
        marks += [("gated", 1, "Active R1", "o", "#b65a1f"), ("gated", 2, "Active R2", "+", "#087560")]
        for arm, round_index, label, shape, tint in marks:
            record = _json(stages[scene, arm, round_index].directory / "observations.json")
            virtual = np.asarray([view["c2w"] for view in record["observations"]])[:, :3, 3]
            positions = (virtual - centroid) @ axes_2d
            line, = ax.plot(*positions.T, linestyle="none", marker=shape, markerfacecolor="none",
                            color=tint, markeredgewidth=1.3, markersize=5, label=label)
            handles[label] = line
        ax.set_title(scene, fontsize=8.5, pad=2)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color("#dce1e4")
    figure.legend(handles.values(), handles.keys(), frameon=False, ncols=len(handles), fontsize=7.5,
                  loc="upper center", bbox_to_anchor=(.5, .995), columnspacing=1.4)
    figure.subplots_adjust(left=.012, right=.988, top=.73, bottom=.07, wspace=.12)
    figure.savefig(destination, dpi=185, facecolor="white")
    plt.close(figure)


def _video_encoder(override):
    if override:
        return str(override)
    installed = shutil.which("ffmpeg")
    if installed:
        return installed
    try:
        import imageio_ffmpeg
    except ImportError as exc:
        raise RuntimeError("Install FFmpeg or imageio-ffmpeg to build the optional videos") from exc
    return imageio_ffmpeg.get_ffmpeg_exe()


def _video_frame(scene, frame, stages, dataset, fonts):
    width, height = 480, 248
    image = Image.new("RGB", (2 * width, 2 * height + 76), "#121b22")
    draw = ImageDraw.Draw(image)
    draw.text((12, 8), f"{scene}   /   Held-out frame {frame['id']}", fill="white", font=fonts[0])
    name = frame["image"]
    panels = [("Ground truth", dataset / name),
              ("Sparse M0", stages[scene, "sparse", 0].render(name)),
              ("Real-only R2", stages[scene, "real_only", 2].render(name)),
              ("Active + gated R2", stages[scene, "gated", 2].render(name))]
    for index, (label, path) in enumerate(panels):
        y, x = 44 + (index // 2) * (height + 12), (index % 2) * width
        draw.text((x + 12, y - 3), label, fill="#e3edf2", font=fonts[1])
        with Image.open(path) as original:
            preview = ImageOps.contain(original.convert("RGB"), (width - 14, height - 27), Image.Resampling.LANCZOS)
            image.paste(preview, (x + (width - preview.width) // 2, y + 16 + (height - 27 - preview.height) // 2))
    return image


def _write_video(executable, path, scenes, stages, inputs, fps):
    from matplotlib.font_manager import findfont

    fonts = [ImageFont.truetype(findfont("DejaVu Sans"), size) for size in (20, 15)]
    temporary = path.with_suffix(".partial.mp4")
    command = [executable, "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
               "-s", "960x572", "-r", str(fps), "-i", "-", "-an", "-c:v", "libx264",
               "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(temporary)]
    try:
        with subprocess.Popen(command, stdin=subprocess.PIPE) as encoder:
            for scene in scenes:
                for frame in stages[scene, "sparse", 0].metrics["per_frame"]:
                    encoder.stdin.write(_video_frame(scene, frame, stages, inputs[scene][0], fonts).tobytes())
            encoder.stdin.close()
            if encoder.wait() != 0:
                raise RuntimeError(f"FFmpeg failed to encode {path}")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def summarize(run_root, output, *, assets_root=None, supplemental=None, selected=None,
              videos_output=None, ffmpeg=None, fps=5):
    fields, rows, stages, scenes = _stages(run_root, selected)
    inputs = _scene_inputs(scenes, assets_root, supplemental)
    figures = output / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    with (output / "comparison.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    with (output / "per_frame_metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["scene", "arm", "round", "id", "image", *METRICS],
                                lineterminator="\n")
        writer.writeheader()
        for (scene, arm, step), stage in stages.items():
            for frame in stage.metrics["per_frame"]:
                writer.writerow({"scene": scene, "arm": arm, "round": step, **frame})
    _closed_loop(stages, scenes, figures / "closed_loop.png")
    _closed_loop_psnr(stages, [scene for scene in scenes if scene in COURSE_SCENES] or scenes,
                      figures / "closed_loop_psnr.png")
    examples = _frame_examples(stages, scenes)
    course_examples = [item for item in examples if item["scene"] in COURSE_SCENES] or examples
    extrema = [course_examples[-1], course_examples[0]] if len(course_examples) > 1 else course_examples
    _heldout_figure(stages, inputs, extrema, figures / "heldout_comparison.png")
    scene_extrema = {}
    for scene in scenes:
        candidates = [item for item in examples if item["scene"] == scene]
        scene_extrema[scene] = [candidates[-1], candidates[0]] if len(candidates) > 1 else candidates
        _heldout_figure(stages, inputs, scene_extrema[scene], figures / f"heldout_{scene}.png")
    pseudo_examples = _pseudo_figure(stages, scenes[0], figures / "pseudo_views.png")
    _trajectory_figure(stages, scenes, inputs, figures / "active_trajectory.png")
    videos = []
    if videos_output:
        videos_output.mkdir(parents=True, exist_ok=True)
        executable = _video_encoder(ffmpeg)
        for scene in scenes:
            path = videos_output / f"{scene}_heldout.mp4"
            _write_video(executable, path, [scene], stages, inputs, fps)
            videos.append(path)
        if len(scenes) > 1:
            path = videos_output / "part3_heldout_demo.mp4"
            _write_video(executable, path, scenes, stages, inputs, fps)
            videos.append(path)
    write_json(output / "visual_evidence.json", {
        "scene_order": scenes,
        "compact_psnr": "Held-out mean PSNR change relative to M0. The scene panels use independent y-axis limits.",
        "heldout_selection": "Largest and smallest Gated R2 minus M0 per-frame PSNR; use the course scenes for the main figure when available. For illustration after evaluation only.",
        "heldout_examples": extrema, "heldout_per_scene": scene_extrema,
        "pseudo_selection": "First scene, most reprojected target support in R1 and least in R2. Confidence color scale is fixed to [0,1].",
        "pseudo_examples": pseudo_examples,
        "trajectory": "PCA projection of estimated sparse train-camera centers and selected virtual cameras; each panel uses its own coordinate scale.",
        "comparison_sha256": sha256_file(output / "comparison.csv"),
        "per_frame_metrics_sha256": sha256_file(output / "per_frame_metrics.csv"),
        "figure_sha256": {path.name: sha256_file(path) for path in sorted(figures.glob("*.png"))},
        "video_sha256": {path.name: sha256_file(path) for path in videos},
    })
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--assets-root", type=Path)
    parser.add_argument("--supplemental", type=Path, help="Receipt for the separately trained supplemental scene")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scene", action="append", help="Include one completed scene; repeat to include several")
    parser.add_argument("--videos-output", type=Path)
    parser.add_argument("--ffmpeg", type=Path, help="Optional explicit FFmpeg executable")
    parser.add_argument("--video-fps", type=int, default=5)
    args = parser.parse_args()
    if args.video_fps < 1:
        parser.error("--video-fps must be positive")
    result = summarize(args.run_root, args.output, assets_root=args.assets_root, supplemental=args.supplemental,
                       selected=args.scene, videos_output=args.videos_output, ffmpeg=args.ffmpeg, fps=args.video_fps)
    print(f"Wrote Part 3 evidence to {result}")


if __name__ == "__main__":
    main()

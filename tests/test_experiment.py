"""CPU integration checks for Part 1 planning, honest coverage, and provenance."""
import copy
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest

from p4 import experiment
from p4.common import sha256_file


PROJECT_CONFIG = Path(__file__).resolve().parents[1] / "configs/part1.json"


@pytest.fixture
def config(tmp_path):
    config = json.loads(PROJECT_CONFIG.read_text(encoding="utf-8"))
    config["dataset_root"] = str(tmp_path / "prepared")
    config["output_root"] = str(tmp_path / "runs")
    for key in ("vggt_weights", "vggt_omega_weights"):
        path = tmp_path / key
        path.write_bytes(b"original checkpoint")
        config["initialization"][key] = str(path)
    return config


def test_exact_mandatory_nine_conditions_and_uniform_budget(config, tmp_path):
    conditions = experiment.conditions(config)
    assert len(conditions) == 9
    assert len(set(conditions)) == 9
    assert {scene for scene, _ in conditions} == {"Waymo-405841", "Dl3dv-2", "Re10k-1"}
    assert {method for _, method in conditions} == {"colmap", "vggt", "vggt-omega"}
    for scene, method in conditions:
        run = tmp_path / scene / method
        commands = experiment.stage_commands(config, scene, method, run)
        assert commands[0][0] == "initialize"
        init = commands[0][1]
        assert init[init.index("--method") + 1] == method
        assert init[init.index("--dataset") + 1] == str(Path(config["dataset_root"]) / scene)
        train = commands[1][1]
        assert "--stop-after" not in train
        assert train[train.index("--config") + 1] == str(run / "frozen_config.json")
        render_stages = [name for name, _ in commands if name.startswith("render_")]
        evaluate_stages = [name for name, _ in commands if name.startswith("evaluate_")]
        assert render_stages == [f"render_{step}" for step in sorted(set(config["training"]["save_steps"] + [7000]))]
        assert evaluate_stages == [stage.replace("render_", "evaluate_") for stage in render_stages]



def test_reproduction_config_routes_runtime_colmap_options(tmp_path):
    config = json.loads(PROJECT_CONFIG.with_name("part1_reproduce.json").read_text(encoding="utf-8"))
    default = json.loads(PROJECT_CONFIG.read_text(encoding="utf-8"))
    assert config["training"] == default["training"]
    assert config["output_root"] != default["output_root"]
    variants = {
        "Waymo-405841": ("--colmap-single-camera", "8", "200"),
        "Dl3dv-2": ("--no-colmap-single-camera", "4", "50"),
        "Re10k-1": ("--colmap-single-camera", "8", "50"),
    }
    for scene, (camera, threads, block) in variants.items():
        def plan(method, configuration):
            return experiment.stage_commands(configuration, scene, method, tmp_path)[0][1]

        assert plan("colmap", config)[-7:] == [
            "--matcher", "exhaustive", camera, "--colmap-num-threads", threads,
            "--colmap-exhaustive-block-size", block,
        ]
        assert not any(token.startswith(("--colmap-", "--no-colmap-")) for token in plan("colmap", default))
        assert "--vggt-offload-features" in plan("vggt", config)
        assert "--vggt-offload-features" not in plan("vggt-omega", config)


def test_planned_cli_commands_parse_without_cuda(config, tmp_path, monkeypatch):
    from p4 import evaluate, initialize, train
    run = tmp_path / "run"
    run.mkdir()
    (run / "frozen_config.json").write_text(json.dumps(config))
    invoked = []
    monkeypatch.setattr(initialize, "run_initializer", lambda **kw: invoked.append(("init", kw)) or {})
    monkeypatch.setattr(train, "train", lambda *args: invoked.append(("train", args)))
    monkeypatch.setattr(train, "render", lambda *args: invoked.append(("render", args)))
    monkeypatch.setattr(evaluate, "evaluate_renders", lambda *args, **kw: {"complete": True, "psnr": 20., "ssim": .8, "lpips": .2})
    modules = {"p4.initialize": initialize, "p4.train": train, "p4.evaluate": evaluate}
    for stage, command in experiment.stage_commands(config, "Waymo-405841", "vggt-omega", run, pilot_steps=10):
        assert command[0] == sys.executable
        monkeypatch.setattr(sys, "argv", [command[2], *command[3:]])
        modules[command[2]].main()
    assert invoked[0][1]["method"] == "vggt-omega"
    assert invoked[1][1][-1] == 10
    assert len([call for call in invoked if call[0] == "render"]) == 1


def test_missing_data_yields_nine_pending_without_scores(config, tmp_path):
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    assert experiment.run_matrix(path) == 9
    rows = experiment.summarize(path, tmp_path / "summary")
    assert len(rows) == 9
    for row in rows:
        assert row["status"] == "pending"
        assert row["psnr"] is None and row["ssim"] is None and row["lpips"] is None
    assert "0/9 completed" in (tmp_path / "summary/coverage.md").read_text()


def test_summary_does_not_publish_unverified_leftover_scores(config, tmp_path):
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    run = Path(config["output_root"]) / "Waymo-405841/colmap"
    (run / "metrics").mkdir(parents=True)
    (run / "status.json").write_text(json.dumps({"status": "failed", "reason": "retry pending"}))
    (run / "metrics/step_007000.json").write_text(json.dumps({"psnr": 99., "ssim": 1., "lpips": 0.}))
    rows = experiment.summarize(path, tmp_path / "summary")
    failed = next(row for row in rows if row["scene"] == "Waymo-405841" and row["method"] == "colmap")
    assert failed["status"] == "failed"
    assert failed["psnr"] is None and failed["ssim"] is None and failed["lpips"] is None


def _minimal_signature_dataset(tmp_path):
    dataset = tmp_path / "dataset"
    (dataset / "images").mkdir(parents=True)
    image = dataset / "images/frame.png"
    image.write_bytes(b"frozen RGB bytes")
    (dataset / "manifest.json").write_text(json.dumps({"schema_version": 1, "frames": [
        {"image": "images/frame.png", "split": "train", "sha256": sha256_file(image)}]}))
    return dataset


def test_signature_changes_with_manifest_or_budget(config, tmp_path, monkeypatch):
    dataset = _minimal_signature_dataset(tmp_path)
    monkeypatch.setattr(experiment, "source_version", lambda: {"package_source_sha256": "fixed"})
    before = experiment.signature(config, dataset)
    config["training"]["steps"] += 1
    assert experiment.signature(config, dataset) != before
    config["training"]["steps"] -= 1
    manifest = json.loads((dataset / "manifest.json").read_text())
    manifest["split_source"] = {"kind": "changed protocol"}
    (dataset / "manifest.json").write_text(json.dumps(manifest))
    assert experiment.signature(config, dataset) != before


def test_signature_rejects_tampered_rgb(config, tmp_path, monkeypatch):
    dataset = _minimal_signature_dataset(tmp_path)
    monkeypatch.setattr(experiment, "source_version", lambda: {"package_source_sha256": "fixed"})
    before = experiment.signature(config, dataset)
    (dataset / "images/frame.png").write_bytes(b"changed RGB bytes")
    try:
        after = experiment.signature(config, dataset)
    except ValueError:
        return  # Manifest SHA mismatch is preferable to silently treating it as a new run.
    assert after != before


def test_signature_changes_when_weights_replaced_at_same_path(config, tmp_path, monkeypatch):
    dataset = _minimal_signature_dataset(tmp_path)
    monkeypatch.setattr(experiment, "source_version", lambda: {"package_source_sha256": "fixed"})
    before = experiment.signature(config, dataset)
    Path(config["initialization"]["vggt_weights"]).write_bytes(b"replacement checkpoint")
    assert experiment.signature(config, dataset) != before


def _mock_pipeline(config, tmp_path, monkeypatch, missing_lpips=False):
    """Use real files/receipts with fake costly initializer/CUDA subprocesses."""
    config["scenes"] = ["Waymo-405841"]
    config["methods"] = ["colmap"]
    config["training"]["steps"] = 2
    config["training"]["save_steps"] = [2]
    dataset = Path(config["dataset_root"]) / "Waymo-405841"
    (dataset / "images").mkdir(parents=True)
    frames = []
    for i in range(4):
        image = dataset / f"images/{i}.png"
        Image.new("RGB", (20, 20), (i * 40, 0, 0)).save(image)
        frames.append({"id": str(i), "image": f"images/{i}.png", "split": "train" if i < 3 else "test",
                       "width": 20, "height": 20, "sha256": sha256_file(image)})
    (dataset / "manifest.json").write_text(json.dumps({"schema_version": 1, "scene": "Waymo-405841", "frames": frames}))
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    calls, state = [], {"lpips_missing": missing_lpips}
    monkeypatch.setattr(experiment, "source_version", lambda: {"package_source_sha256": "fixed"})

    def fake_subprocess(command, **kwargs):
        def argument(name):
            return Path(command[command.index(name) + 1])
        module = command[2]
        output = argument("--output")
        if module == "p4.initialize":
            stage = "initialize"
            output.mkdir(parents=True, exist_ok=True)
            poses = np.tile(np.eye(4), (3, 1, 1))
            poses[:, :3, 3] = [[0, 0, 0], [1, 0, 0], [0, 1, 0]]
            np.savez(output / "scene.npz", names=np.array([f"images/{i}.png" for i in range(3)]),
                     c2w=poses, K=np.tile(np.eye(3), (3, 1, 1)), points=np.ones((4, 3)),
                     colors=np.full((4, 3), 100, dtype=np.uint8))
            (output / "scene.json").write_text(json.dumps({"status": "completed", "manifest_sha256": sha256_file(dataset / "manifest.json")}))
        elif module == "p4.train" and command[3] == "train":
            stage = "train"
            (output / "checkpoints").mkdir(parents=True, exist_ok=True)
            (output / "checkpoints/step_000002.pt").write_bytes(b"fake checkpoint for receipt accounting only")
            (output / "provenance.json").write_text(json.dumps({"initial_gaussians": 4}))
            (output / "config.json").write_text(json.dumps(config["training"]))
            (output / "train.jsonl").write_text(json.dumps({"step": 2, "num_gaussians": 4, "training_seconds": 1., "peak_cuda_gib": 1.}) + "\n")
        elif module == "p4.train":
            stage = "render_2"
            (output / "images").mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (20, 20)).save(output / "images/3.png")
            (output / "render.json").write_text(json.dumps({"step": 2, "raw_gaussian_renders": True, "ate_rmse": .01}))
        else:
            stage = "evaluate_2"
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps({"psnr": 20., "ssim": .8, "lpips": None if state["lpips_missing"] else .2,
                                          "complete": not state["lpips_missing"]}))
        calls.append(stage)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(experiment.subprocess, "run", fake_subprocess)
    return path, Path(config["output_root"]) / "Waymo-405841/colmap", calls, state


def test_incomplete_lpips_is_recomputed_on_retry(config, tmp_path, monkeypatch):
    path, run, calls, state = _mock_pipeline(config, tmp_path, monkeypatch, missing_lpips=True)
    assert experiment.run_matrix(path) == 1
    assert json.loads((run / "status.json").read_text())["status"] == "failed"
    assert not (run / "logs/evaluate_2.done.json").exists()
    state["lpips_missing"] = False
    calls.clear()
    assert experiment.run_matrix(path) == 0
    assert calls == ["evaluate_2"]
    assert json.loads((run / "status.json").read_text())["status"] == "completed"


def test_upstream_rerun_invalidates_downstream_receipts(config, tmp_path, monkeypatch):
    path, run, calls, state = _mock_pipeline(config, tmp_path, monkeypatch)
    assert experiment.run_matrix(path) == 0
    calls.clear()
    assert experiment.run_matrix(path) == 0
    assert calls == []  # Exact frozen run and all outputs can be reused.
    (run / "init/scene.npz").unlink()
    assert experiment.run_matrix(path) == 0
    assert calls == ["initialize", "train", "render_2", "evaluate_2"]


@pytest.mark.parametrize("metric", [None, float("nan"), float("inf")])
def test_stage_validation_rejects_missing_or_nonfinite_metrics(config, tmp_path, metric):
    run = tmp_path / "run"
    (run / "metrics").mkdir(parents=True)
    (run / "metrics/step_007000.json").write_text(json.dumps({"complete": True, "psnr": metric, "ssim": .8, "lpips": .2}))
    with pytest.raises(ValueError, match="incomplete"):
        experiment.stage_outputs("evaluate_7000", run, config, tmp_path, None)

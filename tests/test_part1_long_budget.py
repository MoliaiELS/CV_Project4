"""Validate compact long-budget workflow without starting CUDA training."""

from pathlib import Path
from zipfile import ZipFile

import pytest

from p4.common import sha256_file, write_json
from p4.train import TrainConfig
from scripts import part1_long_budget as long_budget


def write_checkpoint(checkpoint):
    with ZipFile(checkpoint, "w") as archive:
        archive.writestr("checkpoint/data.pkl", checkpoint.stem)


def test_thirty_thousand_step_schedule():
    assert long_budget.budget_config(30000).save_steps == (7000, 15000, 30000)
    assert long_budget.budget_config(10000).save_steps == (7000, 10000)


@pytest.mark.parametrize("completed", (0, 1, 2))
def test_records_fixed_endpoints_before_pruning_checkpoints(tmp_path, monkeypatch, completed):
    config = TrainConfig(steps=2000, save_steps=(1000, 1500, 2000))
    case = {
        "scene": "Dl3dv-2", "method": "colmap", "dataset": "/dataset", "initializer": "/initializer",
        "sample_images": ["images/frame.png"], "split_hash": "test-split", "test_frames": 3,
    }
    calls = []

    def fake_module(module, command, **flags):
        assert module == "p4.train"
        calls.append(command)
        output = flags["output"]
        if command == "train":
            expected = checkpoints[completed - 1] if completed else None
            assert flags.get("resume") == expected
            assert [path for path in checkpoints if path.exists()] == checkpoints[:completed]
            (output / "checkpoints").mkdir(parents=True, exist_ok=True)
            (output / "config.json").write_bytes(flags["config"].read_bytes())
            write_json(output / "provenance.json", {"pose_optimization": False})
            (output / "train.jsonl").write_text("".join(
                f'{{"step": {step}, "loss": 0.2, "num_gaussians": 10, '
                f'"training_seconds": {step}, "peak_cuda_gib": 1}}\n'
                for step in range(50, 2001, 50)
            ), encoding="utf-8")
            for checkpoint in checkpoints[completed:]:
                write_checkpoint(checkpoint)
            return
        checkpoint = flags["checkpoint"]
        step = int(checkpoint.stem.split("_")[-1])
        write_json(output / "render.json", {
            "step": step, "checkpoint_sha256": sha256_file(checkpoint), "raw_gaussian_renders": True,
            "test_pose_refinement": False, "ate_rmse": 0.2,
        })
        (output / "images").mkdir()
        (output / "images" / "frame.png").write_bytes(b"sample")

    def fake_evaluate(dataset, renders):
        assert dataset == Path(case["dataset"])
        assert (renders / "images" / "frame.png").is_file()
        return {"test_frames": 3, "split_hash": "test-split", "complete": True,
                "psnr": 20.0, "ssim": 0.5, "lpips": 0.3}

    monkeypatch.setattr(long_budget, "run_module", fake_module)
    monkeypatch.setattr(long_budget, "evaluate_renders", fake_evaluate)
    output, scratch = tmp_path / "publish", tmp_path / "scratch"
    checkpoint_dir = scratch / case["scene"] / case["method"] / "train" / "checkpoints"
    checkpoints = [checkpoint_dir / f"step_{step:06d}.pt" for step in config.save_steps]
    if completed:
        checkpoint_dir.mkdir(parents=True)
        for path in checkpoints[:completed]:
            write_checkpoint(path)
        checkpoints[completed].write_bytes(b"interrupted write")
        for path in checkpoints[completed + 1:]:
            write_checkpoint(path)
    rows = long_budget.run_case(case, config, output, scratch, keep_final=False)
    assert calls == ["train", "render", "render", "render"]
    assert [row["step"] for row in rows] == [1000, 1500, 2000]
    assert not list(scratch.rglob("*.pt"))
    assert len(list(output.rglob("metrics.json"))) == 3
    assert len(list(output.rglob("frame.png"))) == 3
    assert long_budget.run_case(case, config, output, scratch, keep_final=False) == rows
    assert calls == ["train", "render", "render", "render"]

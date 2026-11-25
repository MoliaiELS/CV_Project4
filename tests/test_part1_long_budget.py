"""Validate compact long-budget workflow without starting CUDA training."""

from pathlib import Path

from p4.common import sha256_file, write_json
from p4.train import TrainConfig
from scripts import part1_long_budget as long_budget


def test_records_fixed_endpoints_before_pruning_checkpoints(tmp_path, monkeypatch):
    config = TrainConfig(steps=2000, save_steps=(1000, 2000))
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
            output.mkdir(parents=True)
            (output / "checkpoints").mkdir()
            (output / "config.json").write_bytes(flags["config"].read_bytes())
            write_json(output / "provenance.json", {"pose_optimization": False})
            (output / "train.jsonl").write_text("".join(
                f'{{"step": {step}, "loss": 0.2, "num_gaussians": 10, '
                f'"training_seconds": {step}, "peak_cuda_gib": 1}}\n'
                for step in range(50, 2001, 50)
            ), encoding="utf-8")
            for step in config.save_steps:
                (output / "checkpoints" / f"step_{step:06d}.pt").write_bytes(str(step).encode())
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
    rows = long_budget.run_case(case, config, output, scratch, keep_final=False)
    assert calls == ["train", "render", "render"]
    assert [row["step"] for row in rows] == [1000, 2000]
    assert not list(scratch.rglob("*.pt"))
    assert len(list(output.rglob("metrics.json"))) == 2
    assert len(list(output.rglob("frame.png"))) == 2
    assert long_budget.run_case(case, config, output, scratch, keep_final=False) == rows
    assert calls == ["train", "render", "render"]

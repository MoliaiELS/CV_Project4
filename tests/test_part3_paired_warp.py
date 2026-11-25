import csv
from dataclasses import asdict
from types import SimpleNamespace

import numpy as np
import pytest

from p4.common import sha256_file, write_json
from scripts import part3_paired_warp as paired
from scripts.part3_fusion import SyntheticObservation
from scripts.part3_views import Camera, save_rgb


def test_naive_and_gated_receive_same_target_with_unfiltered_restoration_in_rejected_pixels():
    source = SyntheticObservation("probe", np.eye(4), np.eye(3), np.zeros((1, 2, 3), np.float32),
                                  np.full((1, 2, 3), 0.25, np.float32), np.array([[0.5, 0]], np.float32))
    left = np.full((1, 2, 3), 0.5, np.float32)
    right = np.ones((1, 2, 3), np.float32)
    target = paired._matched_rgb(source, left, right)
    np.testing.assert_array_equal(target, [[[0.25] * 3, [0.75] * 3]])
    np.testing.assert_array_equal(source.enhanced_rgb, np.full((1, 2, 3), 0.25, np.float32))


def test_fit_matches_r1_inputs_and_reuses_r2_geometry_across_separate_parents(tmp_path, monkeypatch):
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    manifest = {"schema_version": 1, "scene": "probe", "split_hash": "sparse",
                "frames": [{"id": name, "image": name, "split": "train"} for name in ("a.png", "b.png")]
                + [{"id": "hidden.png", "image": "hidden.png", "split": "test"}]}
    write_json(dataset / "manifest.json", manifest)
    initializer, checkpoint = tmp_path / "scene.npz", tmp_path / "m0.pt"
    initializer.write_bytes(b"original-coordinate refined train camera scene")
    checkpoint.write_bytes(b"m0 train-only Gaussian parameters")
    m0_path = tmp_path / "m0.json"
    m0_sha = sha256_file(checkpoint)
    write_json(m0_path, {"dataset": str(dataset), "initializer": str(initializer), "checkpoint": str(checkpoint),
                         "checkpoint_sha256": m0_sha, "initializer_sha256": sha256_file(initializer)})
    initial = {"training_checkpoint_sha256": m0_sha, "step": 30_000,
               "manifest_sha256": sha256_file(dataset / "manifest.json"), "scene_sha256": sha256_file(initializer)}
    m0 = SimpleNamespace(provenance=initial, normalization={"origin": [1, 2, 3], "radius": 5.0},
                         checkpoint_sha256=m0_sha, checkpoint_path=checkpoint, fusion_steps=0,
                         splats={"means": [[0, 0, 0]]})
    mono = np.zeros((2, 2, 3), np.float32)
    conf = np.zeros((2, 2), np.float32)
    calls = []

    def round_plan(model, cameras, folder, round_index, views):
        assert model is m0
        camera = Camera(f"view{round_index}", np.eye(4), np.eye(3), 2, 2)
        return [SimpleNamespace(camera=camera, anchors=cameras)], {
            "chosen": [camera.name], "candidates": [], "planning_seconds": 0.0}

    def generate(model, cameras, train, folder, arm, round_index, *_):
        assert arm == "warp_gated" and train == dataset
        plan = paired._read_json(folder / "plan.json")
        assert plan["parent_checkpoint_sha256"] == model.checkpoint_sha256
        name, = plan["chosen"]
        calls.append((round_index, model.checkpoint_sha256, name))
        save_rgb(folder / name / "restored_left.png", mono)
        save_rgb(folder / name / "restored_right.png", mono + 1)
        receipt = {"observations": [], "planning_seconds": 0.0, "generation_seconds": 0.0}
        write_json(folder / "observations.json", receipt)
        return [SyntheticObservation(name, np.eye(4), np.eye(3), mono, mono, conf)], receipt

    def tune(model, data, context, directory, mode, observations, cfg):
        assert (data, context) == (dataset, initializer)
        path = directory / "checkpoint.pt"
        write_json(path, {"round_step": cfg.steps, "peak_cuda_gib": 0.0, "training_seconds": 1.0,
                          "real_sampler": {"cursor": model.fusion_steps + cfg.steps}, "splats": {"means": [[0, 0, 0]]},
                          "run": {"parent_fusion_steps": model.fusion_steps, "mode": mode, "config": asdict(cfg),
                                  "initial": model.provenance, "n_real_views": 2,
                                  "parent_checkpoint_sha256": model.checkpoint_sha256,
                                  "observations_sha256": paired.fusion._observation_digest(observations, manifest)}})
        return SimpleNamespace(provenance=initial, normalization=m0.normalization, checkpoint_path=path,
                               checkpoint_sha256=sha256_file(path), fusion_steps=model.fusion_steps + cfg.steps)

    monkeypatch.setattr(paired, "training_cameras", lambda *args: [SimpleNamespace(name=name) for name in ("a.png", "b.png")])
    monkeypatch.setattr(paired.progressive, "_round_plan", round_plan)
    monkeypatch.setattr(paired.progressive, "_setup_round", generate)
    monkeypatch.setattr(paired.fusion, "load_local_checkpoint", lambda *args: m0)
    monkeypatch.setattr(paired.fusion, "fine_tune", tune)
    monkeypatch.setattr(paired.fusion, "_read_fusion_payload", paired._read_json)
    monkeypatch.setattr(paired, "_evaluate", lambda *args: pytest.fail("Fit opened the held-out evaluator"))
    args = SimpleNamespace(m0=m0_path, output=tmp_path / "run", steps=5000, seed=42, synthetic_weight=0.4,
                           views=1, max_edge=768, warp_depth=None, precision="float16")
    root = paired.fit(args)
    completed, protocol = paired._check_fit(root, m0_path, dataset, initializer, checkpoint, m0_sha)
    assert protocol["m0_normalization"] == m0.normalization
    assert len(calls) == 3
    assert calls[0] == (1, m0_sha, "view1")
    assert calls[1][2] == calls[2][2] == "view2"
    assert calls[1][1] != calls[2][1]
    assert all(completed["stages"][arm]["2"]["cumulative_updates"] == 40_000 for arm in paired.ARMS)
    assert not (dataset / "hidden.png").exists()

    def reload(path):
        saved = paired._read_json(path)
        return SimpleNamespace(provenance=saved["run"]["initial"], normalization=m0.normalization,
                               checkpoint_sha256=sha256_file(path), splats=saved["splats"],
                               fusion_steps=saved["run"]["parent_fusion_steps"] + saved["round_step"])

    def score(model, *_):
        return {"checkpoint_sha256": model.checkpoint_sha256, "test_frames": 1, "split_hash": "sparse",
                "psnr": 20., "ssim": 0.5, "lpips": 0.25}

    monkeypatch.setattr(paired.fusion, "load_fusion_checkpoint", reload)
    monkeypatch.setattr(paired, "_evaluate", score)
    paired.evaluate(args)
    paired.evaluate(args)
    with (args.output / "comparison.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 7
    assert rows[0]["cumulative_updates"] == "30000"
    assert {row["cumulative_updates"] for row in rows if row["round"] == "2"} == {"40000"}

    (root / "warp_naive/round1/checkpoint.pt").write_bytes(b"changed")
    with pytest.raises(ValueError, match="Fit lineage changed"):
        paired._check_fit(root, m0_path, dataset, initializer, checkpoint, m0_sha)

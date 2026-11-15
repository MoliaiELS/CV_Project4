"""Prepare and verify a separate, checkpoint-compatible Part 2 workspace."""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys

REPO = Path(__file__).resolve().parents[1]
FROZEN_REV = "bd2e3ecae209df5de315fcddb258accf15a57592"
FROZEN_SHA = "c8b6722a8b18deaecd6766279d8e69617e762ec9cb9add52300ba171424bf51c"
DEFAULT_ASSETS = Path("/data/disk2/home/zguo400/CV_Project4")
MANIFEST = REPO / "results/part2_supplement_v1/hpc_artifacts.json"


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def package_digest(root):
    h = hashlib.sha256()
    for path in sorted((Path(root) / "p4").glob("*.py")):
        h.update(path.name.encode())
        h.update(path.read_bytes())
    return h.hexdigest()


def direct_env():
    return {k: v for k, v in os.environ.items() if not k.lower().endswith("_proxy")}


def git(*args, cwd=REPO):
    return subprocess.check_output(
        ["git", "-c", "http.proxy=", "-c", "https.proxy=", *map(str, args)],
        cwd=cwd, env=direct_env(), text=True,
    ).strip()


def inventory(assets):
    assert package_digest(assets) == FROZEN_SHA
    patterns = {
        "part2_checkpoint": "runs/part2_formal/*/full_sparse_7000_v1/*/train/checkpoints/*.pt",
        "part1_checkpoint": "runs/part1_import_901c5ae/checkpoints/*/*/*.pt",
        "part2_initialization": "runs/part2_formal/*/full_sparse_7000_v1/*/init/scene.*",
        "frozen_config": "runs/part2_formal/*/full_sparse_7000_v1/frozen_config.json",
        "full_manifest": "data/prepared/*/manifest.json",
        "full_reference": "data/prepared/*/reference.npz",
        "sparse_manifest": "data/part2_sparse/*/manifest.json",
        "sparse_reference": "data/part2_sparse/*/reference.npz",
    }
    records = []
    for role, pattern in patterns.items():
        files = sorted(assets.glob(pattern))
        if role == "part2_checkpoint":
            assert len(files) == 30
        if role == "part1_checkpoint":
            assert len(files) == 9
        for path in files:
            records.append({"role": role, "relative_path": str(path.relative_to(assets)),
                            "bytes": path.stat().st_size, "sha256": digest(path)})
        print("Inventoried", role, len(files), flush=True)
    for role, rel in [
        ("vggt_weights", "weights/vggt/model.pt"),
        ("lpips_weights", "weights/torch_cache/hub/checkpoints/alexnet-owt-7be5be79.pth"),
        ("original_part2_bundle", "results/part2_formal_7000/summary_bundle.zip"),
        ("original_supplement_bundle", "results/part2_supplement_v1/part2_enhanced_bundle.zip"),
        ("part1_handoff_bundle", "results/hpc_setup/part1_handoff_901c5ae/part1-combined-20261115-211003.zip"),
    ]:
        path = assets / rel
        records.append({"role": role, "relative_path": rel, "bytes": path.stat().st_size,
                        "sha256": digest(path)})
    record = {
        "schema_version": 1, "assets_root": str(assets), "frozen_revision": FROZEN_REV,
        "part2_package_sha256": FROZEN_SHA,
        "part1_training_revision": "bde9244f692cc497355a307826ce1ca25f39cd99",
        "feature_initializer_revision": "901c5aedfe4627e7af93814e8f37e847237adb7b",
        "environment_root": str(assets.parent / "envs/project4"),
        "torch_cache": str(assets / "weights/torch_cache"),
        "cuda_home": os.environ.get("CUDA_HOME", "/usr/local/cuda"),
        "python": sys.version.split()[0],
        "packages": {p: importlib.metadata.version(p) for p in
                     ["torch", "torchvision", "gsplat", "numpy", "Pillow", "scipy", "lpips", "matplotlib"]},
        "checkpoints_in_git": False, "checkpoint_count": 39, "files": records,
    }
    MANIFEST.write_text(json.dumps(record, indent=2) + "\n")
    print("Manifest:", MANIFEST, flush=True)


def load_manifest():
    return json.loads(MANIFEST.read_text())


def verify(assets, full_hash=False):
    manifest = load_manifest()
    failures = []
    for record in manifest["files"]:
        path = assets / record["relative_path"]
        if not path.is_file() or not os.access(path, os.R_OK):
            failures.append(record["relative_path"] + ": missing/unreadable")
        elif path.stat().st_size != record["bytes"]:
            failures.append(record["relative_path"] + ": size differs")
        elif full_hash and digest(path) != record["sha256"]:
            failures.append(record["relative_path"] + ": SHA256 differs")
    if failures:
        raise RuntimeError("\n".join(failures))
    print(json.dumps({"checked_files": len(manifest["files"]), "sha256_checked": full_hash,
                      "checkpoint_count": manifest["checkpoint_count"], "assets_root": str(assets)}))


def validate_destination(destination, assets, repo=REPO):
    destination = Path(destination).resolve()
    for protected in [Path(assets).resolve(), Path(repo).resolve()]:
        if destination == protected or destination.is_relative_to(protected):
            raise ValueError("Destination must be outside the shared assets and publishing checkout")
    if destination.exists():
        raise FileExistsError("Use a new destination; existing workspaces are never overwritten")
    return destination


def prepare(destination, assets):
    destination = validate_destination(destination, assets)
    verify(assets)
    manifest = load_manifest()
    # Clone local Git objects only. No checkout or source mutation in shared assets.
    git("clone", "--no-hardlinks", "--no-checkout", REPO, destination)
    git("checkout", "--detach", FROZEN_REV, cwd=destination)
    for name in git("ls-files", "-z").split("\0"):
        if not name:
            continue
        relative = Path(name)
        if relative.parts[0] not in {"scripts", "docs", "report", "results", "configs"} and name != "README.md":
            continue
        source = REPO / relative
        if source.is_file():
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
    assert package_digest(destination) == FROZEN_SHA
    links = [
        "data/prepared", "data/part2_sparse", "weights", "third_party/vggt",
        "runs/part2_formal", "runs/part1_import_901c5ae",
        "results/hpc_setup/part1_handoff_901c5ae/archive",
        "results/hpc_setup/part1_handoff_901c5ae/published",
        "results/part2_formal_7000/summary_bundle.zip",
    ]
    for name in links:
        source, target = assets / name, destination / name
        if not source.exists():
            raise FileNotFoundError(source)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.symlink_to(source, target_is_directory=source.is_dir())
    # Copy only small preview/cache files; generated diagnostics stay writable.
    for name in ["coverage", "source", "fast_initialization"]:
        source = assets / "runs/part2_supplement_v1" / name
        target = destination / "runs/part2_supplement_v1" / name
        shutil.copytree(source, target, ignore=shutil.ignore_patterns("__pycache__"))
    runtime = destination / ".part2-runtime"
    for name in ["tmp", "hf", "torch/hub/checkpoints", "extensions", "matplotlib"]:
        (runtime / name).mkdir(parents=True, exist_ok=True)
    cache = Path(manifest["torch_cache"]) / "hub/checkpoints"
    for path in cache.glob("*.pth"):
        (runtime / "torch/hub/checkpoints" / path.name).symlink_to(path)
    cuda = manifest["cuda_home"]
    env_lines = [
        "unset http_proxy https_proxy all_proxy ftp_proxy wss_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY FTP_PROXY WSS_PROXY",
        "export NO_PROXY='*' no_proxy='*' HF_HUB_OFFLINE=1",
        "export GIT_CONFIG_GLOBAL=/dev/null GIT_CONFIG_NOSYSTEM=1",
        "export PATH=" + shlex.quote(manifest["environment_root"] + "/bin") + ":" + shlex.quote(cuda + "/bin") + ':"$PATH"',
        "export CUDA_HOME=" + shlex.quote(cuda),
        "export LD_LIBRARY_PATH=" + shlex.quote(cuda + "/lib64"),
        "export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4",
    ]
    for key, name in [("TMPDIR", "tmp"), ("HF_HOME", "hf"), ("TORCH_HOME", "torch"),
                      ("TORCH_EXTENSIONS_DIR", "extensions"), ("MPLCONFIGDIR", "matplotlib")]:
        env_lines.append("export " + key + "=" + shlex.quote(str(runtime / name)))
    (destination / "part2_direct.env").write_text("\n".join(env_lines) + "\n")
    (destination / "handoff_workspace.json").write_text(json.dumps({
        "release_commit": git("rev-parse", "HEAD"), "release_checkout": str(REPO),
        "frozen_revision": FROZEN_REV, "package_sha256": FROZEN_SHA,
        "assets_root": str(assets), "shared_input_links": links,
        "note": "Run commands only in this destination. Keep inputs unchanged and use new outputs for new experiments."
    }, indent=2) + "\n")
    with (destination / ".gitignore").open("a") as stream:
        stream.write("\n.part2-runtime/\npart2_direct.env\nhandoff_workspace.json\n")
    print("Prepared:", destination)
    print("Activate: source " + shlex.quote(str(destination / "part2_direct.env")))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["inventory", "verify", "prepare"])
    parser.add_argument("--assets-root", type=Path, default=DEFAULT_ASSETS)
    parser.add_argument("--destination", type=Path)
    parser.add_argument("--sha256", action="store_true", help="Hash all large assets (slower)")
    args = parser.parse_args()
    assets = args.assets_root.resolve()
    if args.action == "inventory":
        inventory(assets)
    elif args.action == "verify":
        verify(assets, args.sha256)
    else:
        if args.destination is None:
            parser.error("prepare requires --destination")
        prepare(args.destination, assets)


if __name__ == "__main__":
    main()

"""Restore a rendered view with NVIDIA Difix-ref and a training-view reference."""

import argparse
import os
from pathlib import Path
from urllib.request import Request, urlopen

import numpy as np
from PIL import Image

from p4.common import sha256_file, write_json


MODEL_ID = "nvidia/difix_ref"
MODEL_REVISION = "f88399d769a3648f04c5b74f41661bfa25361607"
CODE_REPO = "nv-tlabs/Difix3D"
CODE_REVISION = "c76edc595586e16732c91ddee82f3a6d83a8a9cc"
PROMPT = "remove degradation"
TIMESTEP = 199
PRECISIONS = ("float32", "float16", "bfloat16")

# Exact byte lengths and official SHA-256 digests from the pinned release.
MODEL_FILES = {
    "model_index.json": (598, "06c73e4d1c483e1c81fd0b2f8a89f021f5b31f244c217ff7d62d28f6d37cc1e1"),
    "scheduler/scheduler_config.json": (700, "78e1c4d74df2c94c7d886f0d3f9ccff9c88851dda9c6ae4ccab3356a18efa855"),
    "text_encoder/config.json": (607, "dcd891d1479e4e083780455e8b7dddedc230515abe098e7df2ef34f31be99b29"),
    "text_encoder/model.safetensors": (1361596304, "67e013543d4fac905c882e2993d86a2d454ee69dc9e8f37c0c23d33a48959d15"),
    "tokenizer/merges.txt": (524619, "9fd691f7c8039210e0fced15865466c65820d09b63988b0174bfe25de299051a"),
    "tokenizer/special_tokens_map.json": (574, "c2d0fb8b86ad86b1f46134d4a5f93fd1e688c932a78efc8d149087c33a53ad06"),
    "tokenizer/tokenizer_config.json": (885, "b91e0a1eba063043b4ee76bec870f2fa0c12a3ff404155b30e64c77d25c0758f"),
    "tokenizer/vocab.json": (1059962, "e089ad92ba36837a0d31433e555c8f45fe601ab5c221d4f607ded32d9f7a4349"),
    "unet/config.json": (1856, "7d12877cd698d80a4f028c75f5b6e81dd73c984054610a4e5e501ffaecfe1fe9"),
    "unet/diffusion_pytorch_model.safetensors": (3463726504, "cf723d40f29c6915b6ac2aac3c1dab4fe685afe35a41e725ad63a34124c0ec46"),
    "unet/unet_2d_condition.py": (69247, "d2acb23ac4d0bd164311b6622006c7184cf5e3469086cb255cfa8b32e6a184b5"),
    "vae/config.json": (702, "a4c243c684069c913f79da23d23c9cd8ac98f4cb957b899b2603e2adbfbebc19"),
    "vae/diffusion_pytorch_model.safetensors": (338717612, "d3aa93824c839302d1103d72b0ea933df65206945b0e93140328562fffa6cf65"),
    "vae/autoencoder_kl.py": (24456, "a0c16e2fe489d0386b04274b25e6cec212f37264283f8ce1c042270d27250edf"),
    "LICENSE.txt": (15848, "e2f578631a7d4b5aff03be4489b5defbd6536c8878024720c8261a3b0eea6c1a"),
    "NOTICE": (128, "9fb1c3887e21c02407a5aa5f995f3a522aab46c05aee7f8a1aa0f3cdc13f5962"),
}
PIPELINE_FILE = "pipeline_difix.py"
PIPELINE_SIZE = 56400
PIPELINE_SHA256 = "2f73e2708b3f9ce560800163554f869e5e43e3a42049f67da3609f7736cbab3a"


def model_directory(model_dir=None, cache_dir=None):
    """Find the staged snapshot or a project-specific cache outside the repo."""
    if model_dir is not None and cache_dir is not None:
        raise ValueError("Supply either model_dir or cache_dir.")
    if model_dir is not None:
        return Path(model_dir).expanduser().resolve()
    if cache_dir is None and os.environ.get("P4_DIFIX_MODEL_DIR"):
        return Path(os.environ["P4_DIFIX_MODEL_DIR"]).expanduser().resolve()
    root = Path(cache_dir or Path.home() / ".cache" / "cv_project4").expanduser()
    return (root / f"difix_ref-{MODEL_REVISION[:12]}").resolve()


def _asset_valid(path, expected_size, expected_sha256):
    return (path.is_file()
            and (expected_size is None or path.stat().st_size == expected_size)
            and sha256_file(path) == expected_sha256)


def verify_assets(model_dir):
    """Verify all checkpoints and custom Python before running model code."""
    required = {**MODEL_FILES, PIPELINE_FILE: (PIPELINE_SIZE, PIPELINE_SHA256)}
    for filename, (size, digest) in required.items():
        path = Path(model_dir) / filename
        if not _asset_valid(path, size, digest):
            raise ValueError(f"Missing or invalid Difix-ref asset: {path}. Run --prepare-assets.")


def _download_verified(url, destination, size, digest):
    if _asset_valid(destination, size, digest):
        return
    if destination.exists():
        raise ValueError(f"Invalid existing asset: {destination}. Use a fresh model directory.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(f"{destination.name}.part")
    offset = partial.stat().st_size if partial.exists() else 0
    if offset >= size:
        if _asset_valid(partial, size, digest):
            partial.replace(destination)
            return
        offset = 0
    headers = {"Range": f"bytes={offset}-"} if offset else {}
    print(f"Downloading {destination.name} ({size} bytes)", flush=True)
    with urlopen(Request(url, headers=headers), timeout=120) as response:
        resumed = response.status == 206 and offset > 0
        if resumed and not response.headers.get("Content-Range", "").startswith(f"bytes {offset}-"):
            raise ValueError(f"Unexpected resumed range for {destination}.")
        with partial.open("ab" if resumed else "wb") as stream:
            while block := response.read(1024 * 1024):
                stream.write(block)
    if not _asset_valid(partial, size, digest):
        raise ValueError(f"Downloaded asset failed verification: {partial}")
    partial.replace(destination)


def prepare_assets(model_dir, endpoint=None):
    """Download only the required files, directly into the verified snapshot."""
    endpoint = endpoint or os.environ.get("HF_ENDPOINT", "https://huggingface.co")
    root = Path(model_dir)
    base = f"{endpoint.rstrip('/')}/{MODEL_ID}/resolve/{MODEL_REVISION}"
    for filename, (size, digest) in MODEL_FILES.items():
        _download_verified(f"{base}/{filename}", root / filename, size, digest)
    pipeline_url = f"https://raw.githubusercontent.com/{CODE_REPO}/{CODE_REVISION}/src/{PIPELINE_FILE}"
    _download_verified(pipeline_url, root / PIPELINE_FILE, PIPELINE_SIZE, PIPELINE_SHA256)


def _read_rgb(image):
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    with Image.open(image) as source:
        return source.convert("RGB")


def _pad_for_vae(image):
    # Difix's VAE downsamples by eight. Edge padding preserves the calibrated
    # image extent and is discarded before the output is evaluated or reused.
    width, height = image.size
    pixels = np.asarray(image)
    return Image.fromarray(np.pad(pixels, ((0, -height % 8), (0, -width % 8), (0, 0)), mode="edge"))


class DifixRestorer:
    """Offline model loading and reproducible single-view restoration."""

    def __init__(self, *, device="cuda", model_dir=None, cache_dir=None,
                 dtype="float32", max_edge=1024, seed=0):
        if dtype not in PRECISIONS:
            raise ValueError(f"Invalid Difix precision: {dtype}")
        if max_edge is not None and max_edge < 8:
            raise ValueError("max_edge must be at least 8 pixels.")
        self.model_dir = model_directory(model_dir, cache_dir)
        verify_assets(self.model_dir)

        import torch
        from diffusers import DiffusionPipeline

        self.torch = torch
        self.device = torch.device(device)
        self.dtype = dtype
        self.max_edge = max_edge
        self.seed = seed
        self.pipeline = DiffusionPipeline.from_pretrained(
            str(self.model_dir),
            custom_pipeline=str(self.model_dir / PIPELINE_FILE),
            local_files_only=True,
            use_safetensors=True,
            torch_dtype=getattr(torch, dtype),
        ).to(self.device)
        self.pipeline.set_progress_bar_config(disable=True)

    def provenance(self):
        return {
            "method": "Difix-ref",
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "code_repository": f"https://github.com/{CODE_REPO}",
            "code_revision": CODE_REVISION,
            "sha256": {**{name: digest for name, (_, digest) in MODEL_FILES.items()},
                       PIPELINE_FILE: PIPELINE_SHA256},
            "prompt": PROMPT,
            "timesteps": [TIMESTEP],
            "inference_steps": 1,
            "guidance_scale": 0.0,
            "device": str(self.device),
            "precision": self.dtype,
            "max_edge": self.max_edge,
            "seed": self.seed,
            "image_transform": "Aspect-preserving input resize, reference resized to input, edge-pad to 8, crop and restore original dimensions",
        }

    def restore(self, image, reference, *, seed=None):
        """Restore one RGB image from PIL or disk, returning its original size."""
        image = _read_rgb(image)
        reference = _read_rgb(reference)
        original_size = image.size
        if self.max_edge is not None and max(original_size) > self.max_edge:
            scale = self.max_edge / max(original_size)
            image = image.resize(tuple(max(1, round(edge * scale)) for edge in original_size), Image.Resampling.LANCZOS)
        width, height = image.size
        if reference.size != image.size:
            reference = reference.resize(image.size, Image.Resampling.LANCZOS)

        generator = self.torch.Generator(device=self.device).manual_seed(self.seed if seed is None else seed)
        restored = self.pipeline(
            PROMPT,
            image=_pad_for_vae(image),
            ref_image=_pad_for_vae(reference),
            num_inference_steps=1,
            timesteps=[TIMESTEP],
            guidance_scale=0.0,
            generator=generator,
        ).images[0]
        if restored.size != (width + (-width % 8), height + (-height % 8)):
            raise RuntimeError(f"Difix returned unexpected output size {restored.size}.")
        restored = restored.crop((0, 0, width, height)).convert("RGB")
        if restored.size != original_size:
            restored = restored.resize(original_size, Image.Resampling.LANCZOS)
        return restored


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    directory = parser.add_mutually_exclusive_group()
    directory.add_argument("--model-dir", type=Path)
    directory.add_argument("--cache-dir", type=Path)
    parser.add_argument("--endpoint", help="Optional Hugging Face mirror used by --prepare-assets")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--prepare-assets", action="store_true")
    mode.add_argument("--verify-assets", action="store_true")
    mode.add_argument("--image", type=Path)
    parser.add_argument("--reference", type=Path, help="RGB image from the training split")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=PRECISIONS, default="float32")
    parser.add_argument("--max-edge", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    root = model_directory(args.model_dir, args.cache_dir)
    if args.prepare_assets:
        prepare_assets(root, args.endpoint)
        print(f"Verified Difix-ref assets in {root}")
    elif args.verify_assets:
        verify_assets(root)
        print(f"Verified Difix-ref assets in {root}")
    else:
        if not all((args.image, args.reference, args.output)):
            parser.error("supply --image, --reference and --output for restoration")
        restorer = DifixRestorer(model_dir=root, device=args.device, dtype=args.dtype,
                                max_edge=args.max_edge, seed=args.seed)
        result = restorer.restore(args.image, args.reference)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        result.save(args.output)
        write_json(args.output.with_suffix(args.output.suffix + ".json"), {
            **restorer.provenance(),
            "input_sha256": sha256_file(args.image),
            "reference_sha256": sha256_file(args.reference),
            "output_sha256": sha256_file(args.output),
        })


if __name__ == "__main__":
    main()

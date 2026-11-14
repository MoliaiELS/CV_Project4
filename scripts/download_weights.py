"""Download explicit official artifacts; HF_TOKEN may be required for Omega.

Does not print credentials, request access, or accept license agreements.
"""
import argparse
from pathlib import Path
import shutil


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=["vggt", "vggt-omega", "all"], default="all")
    args = parser.parse_args()
    from huggingface_hub import hf_hub_download
    assets = {
        "vggt": ("facebook/VGGT-1B", "model.pt", "weights/vggt/model.pt"),
        "vggt-omega": ("facebook/VGGT-Omega", "vggt_omega_1b_416_reproduce.pt", "weights/vggt-omega/vggt_omega_1b_416_reproduce.pt")
    }
    for method, (repo, filename, destination) in assets.items():
        if args.method not in (method, "all"):
            continue
        path = Path(destination)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.is_file():
            print(f"Exists: {path}; not replacing")
            continue
        try:
            cached = hf_hub_download(repo_id=repo, filename=filename)
            shutil.copy2(cached, path)
            print(f"Downloaded {method}: {path}")
        except Exception as error:
            raise SystemExit(f"{method} download failed ({type(error).__name__}). Check the official model page and account access; set HF_TOKEN if required.") from None


if __name__ == "__main__":
    main()

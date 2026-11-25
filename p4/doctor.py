"""Record environment and prerequisites without loading model weights."""
import argparse
import importlib.metadata
import json
from pathlib import Path
import shutil
import subprocess

from p4.common import source_version, write_json


def doctor():
    result = {**source_version(), "packages": {}, "executables": {}, "cuda_available": False}
    for package in ["numpy", "Pillow", "scipy", "scikit-image", "torch", "torchvision", "gsplat", "lpips"]:
        try:
            result["packages"][package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            result["packages"][package] = None
    for executable in ["git", "colmap", "nvcc", "nvidia-smi"]:
        result["executables"][executable] = shutil.which(executable)
    try:
        import torch
        result["cuda_available"] = torch.cuda.is_available()
        result["torch_cuda"] = torch.version.cuda
        result["gpus"] = [{"name": torch.cuda.get_device_name(i),
                           "memory_gib": torch.cuda.get_device_properties(i).total_memory / 1024**3}
                          for i in range(torch.cuda.device_count())]
    except ImportError:
        pass
    result["ready_for_training"] = bool(result["cuda_available"] and result["packages"]["gsplat"] == "1.5.3")
    result["ready_for_colmap"] = bool(result["executables"]["colmap"])
    result["weights"] = {str(p): p.is_file() for p in [Path("weights/vggt/model.pt"), Path("weights/vggt-omega/vggt_omega_1b_416_reproduce.pt")]}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="results/doctor.json")
    args = parser.parse_args()
    result = doctor()
    write_json(args.output, result)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
# Run inside a Python 3.10/3.11 environment on Linux with CUDA toolkit 12.4.
python -c 'import sys; assert (3,10) <= sys.version_info[:2] <= (3,11), "Use Python 3.10 or 3.11"'
command -v nvidia-smi >/dev/null || { echo 'NVIDIA driver unavailable'; exit 1; }
command -v nvcc >/dev/null || { echo 'Install CUDA toolkit 12.4 (nvcc), not only the driver'; exit 1; }
nvidia-smi
nvcc --version
python -m pip install --upgrade 'pip>=24' 'setuptools>=68' wheel
python -m pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r requirements-server.txt
python -m pip install --no-build-isolation 'git+https://github.com/nerfstudio-project/gsplat.git@937e29912570c372bed6747a5c9bf85fed877bae'
python -m pip install --no-deps -e .
mkdir -p third_party weights/vggt weights/vggt-omega
clone_pinned() {
    local url="$1" destination="$2" revision="$3"
    if [[ ! -d "$destination/.git" ]]; then
        git clone --filter=blob:none --no-checkout "$url" "$destination"
        git -C "$destination" checkout --detach "$revision"
    fi
    [[ "$(git -C "$destination" rev-parse HEAD)" == "$revision" ]] || { echo "Wrong source revision: $destination"; exit 1; }
}
clone_pinned https://github.com/facebookresearch/vggt.git third_party/vggt a288dd0f14786c93483e45524328726ab7b1b4ce
clone_pinned https://github.com/facebookresearch/vggt-omega.git third_party/vggt-omega 48b23c8ce72c9a8fcf987f0d7f43d8e4870759f6
# COLMAP is installed separately because building options vary by server.
mkdir -p output/environment
python -m p4.doctor --output output/environment/doctor.json
python -m pip freeze > output/environment/packages.txt
echo 'Next: install COLMAP, download weights, run python scripts/gpu_smoke.py, then prepare course data.'

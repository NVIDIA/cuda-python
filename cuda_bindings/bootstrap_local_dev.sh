#! /bin/bash
set -euo pipefail

if [[ $# -ne 3 ]] || { [[ "$3" != "build_wheel" ]] && [[ "$3" != "use_wheel" ]]; }; then
    echo "Usage: $(basename "$0") CondaEnvNamePrefix 12.x {build_wheel|use_wheel}"
    exit 1
fi

CEN="${1}$(echo "${2}" | tr . _)"
CUV="$2"

source "$HOME/miniforge3/etc/profile.d/conda.sh"

set -x
conda create --yes -n "$CEN" python=3.12 \
    cuda-cudart-dev cuda-cudart cuda-nvrtc-dev cuda-nvrtc \
    cuda-profiler-api cuda-nvcc libnvjitlink libnvjitlink-dev \
    cuda-version="$CUV"
set +xu
conda activate "$CEN"
set -xu
export CUDA_HOME="$CONDA_PREFIX/targets/x86_64-linux"

python -m venv "${CEN}Venv"
set +x
source "${CEN}Venv/bin/activate"
set -x
pip install --upgrade pip
pip install -r requirements.txt

if [[ "$3" == "build_wheel" ]]; then
    pip install build
    rm -rf ./dist
    CUDA_PYTHON_PARALLEL_LEVEL=64 python -m build -v
fi

if ls ./dist/cuda_bindings-*.whl 1>/dev/null 2>&1; then
    pip install -v --force-reinstall ./dist/cuda_bindings-*.whl
else
    echo "Error: No wheel file found in ./dist/"
    exit 1
fi

export LD_LIBRARY_PATH="${CUDA_HOME}/nvvm/lib64"
pytest -v tests/

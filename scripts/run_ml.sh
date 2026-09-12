#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/ml_common.sh"
if (( $# < 2 )); then
    echo "Usage: bash scripts/run_ml.sh {tf|rl} COMMAND [ARGUMENTS...]" >&2
    exit 2
fi
select_ml_profile "$1"
shift
if [[ ! -x "$UV_PROJECT_ENVIRONMENT/bin/python" ]]; then
    echo "Environment missing. Run: bash scripts/setup_ml.sh all" >&2
    exit 1
fi
# Use the environment's NVIDIA wheels; do not rely on a machine-wide CUDA toolkit.
if [[ "$ML_PROJECT" == "$ML_ROOT" ]]; then
    nvidia_dir="$UV_PROJECT_ENVIRONMENT/lib/python3.12/site-packages/nvidia"
    for lib_dir in "$nvidia_dir"/*/lib; do
        [[ -d "$lib_dir" ]] && export LD_LIBRARY_PATH="$lib_dir${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    done
    if [[ -d "$nvidia_dir/cuda_nvcc/bin" ]]; then
        export PATH="$nvidia_dir/cuda_nvcc/bin:$PATH"
    fi
    if [[ -d "$nvidia_dir/cuda_nvcc/nvvm/libdevice" && "${XLA_FLAGS:-}" != *--xla_gpu_cuda_data_dir=* ]]; then
        export XLA_FLAGS="${XLA_FLAGS:+$XLA_FLAGS }--xla_gpu_cuda_data_dir=$nvidia_dir/cuda_nvcc"
    fi
fi
exec uv run --project "$ML_PROJECT" --locked "${ML_EXTRAS[@]}" "$@"

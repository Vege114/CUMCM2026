#!/usr/bin/env bash
# Source this file from the setup and run entry points.
set -euo pipefail
ML_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ML_ROOT"
export PATH="$HOME/.local/bin:$PATH"
export UV_CACHE_DIR="$ML_ROOT/.cache/uv"
export KERAS_HOME="$ML_ROOT/.cache/keras"
export MPLCONFIGDIR="$ML_ROOT/.cache/matplotlib"
export XDG_CACHE_HOME="$ML_ROOT/.cache"
export TORCH_HOME="$ML_ROOT/.cache/torch"
export TRITON_CACHE_DIR="$ML_ROOT/.cache/triton"
export CUDA_CACHE_PATH="$ML_ROOT/.cache/cuda"
mkdir -p "$CUDA_CACHE_PATH" "$KERAS_HOME" "$MPLCONFIGDIR"
export TF_FORCE_GPU_ALLOW_GROWTH="${TF_FORCE_GPU_ALLOW_GROWTH:-true}"
export TF_CPP_MIN_LOG_LEVEL="${TF_CPP_MIN_LOG_LEVEL:-2}"

select_ml_profile() {
    case "$1" in
        tf)
            ML_PROJECT="$ML_ROOT"
            export UV_PROJECT_ENVIRONMENT="$ML_ROOT/.venv-wsl"
            ML_EXTRAS=(--extra cuda)
            ;;
        rl)
            ML_PROJECT="$ML_ROOT/environments/rl"
            export UV_PROJECT_ENVIRONMENT="$ML_ROOT/.venv-rl-wsl"
            ML_EXTRAS=()
            ;;
        *) echo "Profile must be tf or rl" >&2; return 2 ;;
    esac
}

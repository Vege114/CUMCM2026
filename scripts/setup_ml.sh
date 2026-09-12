#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/ml_common.sh"
profile="${1:-all}"
case "$profile" in
    all) profiles=(tf rl) ;;
    tf|rl) profiles=("$profile") ;;
    *) echo "Usage: bash scripts/setup_ml.sh [all|tf|rl]" >&2; exit 2 ;;
esac
command -v uv >/dev/null || { echo "Install uv in Linux/WSL2 first: https://docs.astral.sh/uv/" >&2; exit 1; }
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
for profile in "${profiles[@]}"; do
    select_ml_profile "$profile"
    uv sync --project "$ML_PROJECT" --locked "${ML_EXTRAS[@]}" --python 3.12
    if [[ "$profile" == tf ]]; then
        bash "$ML_ROOT/scripts/run_ml.sh" tf python scripts/check_environment.py
    else
        bash "$ML_ROOT/scripts/run_ml.sh" rl python scripts/check_rl_environment.py
    fi
done

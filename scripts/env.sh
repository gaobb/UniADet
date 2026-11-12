_EXPERIMENTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${_EXPERIMENTS_DIR}/.." && pwd)"
cd "$PROJECT_ROOT"

export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/third_party:${PROJECT_ROOT}/third_party/perception_models"
export UNIADET_DATA="${UNIADET_DATA:-${PROJECT_ROOT}/data}"
export UNIADET_CKPT="${UNIADET_CKPT:-${PROJECT_ROOT}/checkpoints}"
export UNIADET_RESULTS="${UNIADET_RESULTS:-${PROJECT_ROOT}/results}"
export UNIADET_DEVICE="${UNIADET_DEVICE:-0}"

if [ -f "${_EXPERIMENTS_DIR}/env.local.sh" ]; then
  source "${_EXPERIMENTS_DIR}/env.local.sh"
fi

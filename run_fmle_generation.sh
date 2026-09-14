#!/usr/bin/env bash

set -Eeuo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SESSION_START_EPOCH="$(date +%s)"

FMLE_STORAGE_ROOT="${FMLE_STORAGE_ROOT:-/workspace}"
if [[ ! -d "${FMLE_STORAGE_ROOT}" || ! -w "${FMLE_STORAGE_ROOT}" ]]; then
  FMLE_STORAGE_ROOT="${PROJECT_ROOT}"
fi

RUNS_ROOT="${RUNS_ROOT:-${FMLE_STORAGE_ROOT}/repoexec-runs}"
TASK_LIMIT="${TASK_LIMIT:-30}"
NUM_RETURN_SEQUENCES="${NUM_RETURN_SEQUENCES:-5}"
OLLAMA_PARALLEL_REQUESTS="${OLLAMA_PARALLEL_REQUESTS:-5}"
SERIAL_OLLAMA_REQUESTS="${SERIAL_OLLAMA_REQUESTS:-1}"
PARALLEL_TASKS="${PARALLEL_TASKS:-1}"
OLLAMA_NUM_CTX="${OLLAMA_NUM_CTX:-4096}"
RUN_PREFIX="${RUN_PREFIX:-fmle-generation30}"
PULL_MODELS="${PULL_MODELS:-1}"
INSTALL_OLLAMA="${INSTALL_OLLAMA:-1}"
REQUIRE_GPU="${REQUIRE_GPU:-1}"
RESTART_OLLAMA="${RESTART_OLLAMA:-1}"

export HF_HOME="${HF_HOME:-${FMLE_STORAGE_ROOT}/hf-cache}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-0}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-0}"
export OLLAMA_MODELS="${OLLAMA_MODELS:-${FMLE_STORAGE_ROOT}/ollama-models}"
OLLAMA_RUNTIME_DIR="${OLLAMA_RUNTIME_DIR:-${FMLE_STORAGE_ROOT}/ollama-runtime}"
export PATH="${OLLAMA_RUNTIME_DIR}/bin:${PATH}"
export LD_LIBRARY_PATH="${OLLAMA_RUNTIME_DIR}/lib/ollama:${LD_LIBRARY_PATH:-}"
export OLLAMA_HOST="${OLLAMA_HOST:-127.0.0.1:11434}"
OLLAMA_BASE_URL="${OLLAMA_BASE_URL:-http://${OLLAMA_HOST}}"
export OLLAMA_FLASH_ATTENTION="${OLLAMA_FLASH_ATTENTION:-1}"
export OLLAMA_KEEP_ALIVE="${OLLAMA_KEEP_ALIVE:-30m}"
DEFAULT_OLLAMA_NUM_PARALLEL="$((OLLAMA_PARALLEL_REQUESTS * PARALLEL_TASKS))"
export OLLAMA_NUM_PARALLEL="${OLLAMA_NUM_PARALLEL:-${DEFAULT_OLLAMA_NUM_PARALLEL}}"
export OLLAMA_CONTEXT_LENGTH="${OLLAMA_CONTEXT_LENGTH:-${OLLAMA_NUM_CTX}}"

LOCAL_NO_PROXY="127.0.0.1,localhost"
export NO_PROXY="${NO_PROXY:+${NO_PROXY},}${LOCAL_NO_PROXY}"
export no_proxy="${no_proxy:+${no_proxy},}${LOCAL_NO_PROXY}"

MODELS=(
  "qwen2.5-coder:1.5b-base"
  "qwen2.5-coder:3b-base"
  "qwen2.5-coder:7b-base"
  "deepseek-coder:1.3b-base-q4_K_M"
  "deepseek-coder:6.7b-base-q4_K_M"
  "deepseek-coder-v2:16b-lite-base-q4_K_M"
  "codegemma:2b-code-q4_K_M"
  "codegemma:7b-code-q4_K_M"
)

PARALLEL_MODELS=(
  "qwen2.5-coder:1.5b-base"
  "qwen2.5-coder:3b-base"
  "qwen2.5-coder:7b-base"
  "deepseek-coder:1.3b-base-q4_K_M"
  "deepseek-coder:6.7b-base-q4_K_M"
  "codegemma:2b-code-q4_K_M"
  "codegemma:7b-code-q4_K_M"
)

python_supported() {
  "$1" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 1)' >/dev/null 2>&1
}

PYTHON_BIN="${PYTHON_BIN:-}"
if [[ -n "${PYTHON_BIN}" ]] && ! python_supported "${PYTHON_BIN}"; then
  echo "PYTHON_BIN must point to Python 3.10 or newer: ${PYTHON_BIN}" >&2
  exit 1
fi

if [[ -z "${PYTHON_BIN}" ]]; then
  for candidate in python3.12 python3.11 python3.10 /opt/conda/bin/python python3; do
    candidate_path="$(command -v "${candidate}" 2>/dev/null || true)"
    if [[ -n "${candidate_path}" ]] && python_supported "${candidate_path}"; then
      PYTHON_BIN="${candidate_path}"
      break
    fi
  done
fi

if [[ -z "${PYTHON_BIN}" ]]; then
  echo "Python 3.10 or newer is required. Load a Python/Conda module or set PYTHON_BIN explicitly." >&2
  python3 --version 2>/dev/null || true
  exit 1
fi

PYTHON_VERSION="$("${PYTHON_BIN}" --version 2>&1)"
echo "Using Python: ${PYTHON_BIN} (${PYTHON_VERSION})"

if [[ "${REQUIRE_GPU}" == "1" ]]; then
  GPU_LIST="$(nvidia-smi -L 2>/dev/null || true)"
  if [[ -z "${GPU_LIST}" ]]; then
    echo "No allocated NVIDIA GPU is visible. Run this inside the FMLe GPU/Jupyter instance, not on login01." >&2
    exit 1
  fi
fi

VENV_DIR="${VENV_DIR:-${FMLE_STORAGE_ROOT}/repoexec-venv-py310}"
mkdir -p "${HF_DATASETS_CACHE}" "${OLLAMA_MODELS}" "${RUNS_ROOT}"

if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
  "${PYTHON_BIN}" -m venv "${VENV_DIR}"
elif ! python_supported "${VENV_DIR}/bin/python"; then
  echo "Existing venv uses Python older than 3.10: ${VENV_DIR}" >&2
  echo "Set VENV_DIR to a new path and run again." >&2
  exit 1
fi
"${VENV_DIR}/bin/python" -m pip install --upgrade pip
"${VENV_DIR}/bin/python" -m pip install -r "${PROJECT_ROOT}/requirements-baseline.txt"

if ! command -v ollama >/dev/null 2>&1; then
  if [[ "${INSTALL_OLLAMA}" != "1" ]]; then
    echo "ollama is required (or run with INSTALL_OLLAMA=1)" >&2
    exit 1
  fi
  echo "Installing Ollama without sudo into ${OLLAMA_RUNTIME_DIR}"
  "${VENV_DIR}/bin/python" -m repoexec_baseline.install_ollama_user \
    --install-dir "${OLLAMA_RUNTIME_DIR}"
fi

if ! command -v ollama >/dev/null 2>&1; then
  echo "Ollama installation completed but the binary is not available in PATH" >&2
  exit 1
fi

OLLAMA_PID=""
cleanup() {
  if [[ -n "${OLLAMA_PID}" ]] && kill -0 "${OLLAMA_PID}" >/dev/null 2>&1; then
    kill "${OLLAMA_PID}" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT

if [[ "${RESTART_OLLAMA}" == "1" ]] && ollama list >/dev/null 2>&1; then
  echo "Restarting Ollama so server concurrency settings take effect"
  pkill -x ollama >/dev/null 2>&1 || true
  for _ in $(seq 1 30); do
    if ! ollama list >/dev/null 2>&1; then
      break
    fi
    sleep 1
  done
  if ollama list >/dev/null 2>&1; then
    echo "Could not stop the existing Ollama server; parallel settings were not applied" >&2
    exit 1
  fi
fi

if ! ollama list >/dev/null 2>&1; then
  ollama serve > "${RUNS_ROOT}/${RUN_PREFIX}-ollama.log" 2>&1 &
  OLLAMA_PID="$!"
  for _ in $(seq 1 60); do
    if ollama list >/dev/null 2>&1; then
      break
    fi
    sleep 1
  done
fi

if ! ollama list >/dev/null 2>&1; then
  echo "Ollama did not become ready. Check ${RUNS_ROOT}/${RUN_PREFIX}-ollama.log" >&2
  exit 1
fi

nvidia-smi
echo "Generation concurrency: tasks=${PARALLEL_TASKS}, candidates=${OLLAMA_PARALLEL_REQUESTS}, server=${OLLAMA_NUM_PARALLEL}, num_ctx=${OLLAMA_NUM_CTX}"

if [[ "${PULL_MODELS}" == "1" ]]; then
  for model in "${MODELS[@]}"; do
    if ollama show "${model}" >/dev/null 2>&1; then
      echo "Model already available: ${model}"
    else
      echo "Pulling model: ${model}"
      ollama pull "${model}"
    fi
  done
fi

cd "${PROJECT_ROOT}"
echo "Parallel phase: models up to 7B"
"${VENV_DIR}/bin/python" -m repoexec_baseline.run_generation_matrix \
  --models "${PARALLEL_MODELS[@]}" \
  --representations raw ast reduced_ast \
  --subset full_context \
  --task-limit "${TASK_LIMIT}" \
  --runs-root "${RUNS_ROOT}" \
  --run-prefix "${RUN_PREFIX}" \
  --num-return-sequences "${NUM_RETURN_SEQUENCES}" \
  --max-new-tokens 256 \
  --do-sample \
  --temperature 0.2 \
  --top-p 0.95 \
  --seed 42 \
  --ollama-base-url "${OLLAMA_BASE_URL}" \
  --ollama-keep-alive "${OLLAMA_KEEP_ALIVE}" \
  --ollama-num-ctx "${OLLAMA_NUM_CTX}" \
  --ollama-parallel-requests "${OLLAMA_PARALLEL_REQUESTS}" \
  --parallel-tasks "${PARALLEL_TASKS}" \
  --no-archive \
  --session-start-epoch "${SESSION_START_EPOCH}"

echo "Restarting Ollama for serial deepseek-coder-v2:16b generation"
pkill -x ollama >/dev/null 2>&1 || true
for _ in $(seq 1 30); do
  if ! ollama list >/dev/null 2>&1; then
    break
  fi
  sleep 1
done
if ollama list >/dev/null 2>&1; then
  echo "Could not stop Ollama before the serial phase" >&2
  exit 1
fi

export OLLAMA_NUM_PARALLEL="${SERIAL_OLLAMA_REQUESTS}"
ollama serve >> "${RUNS_ROOT}/${RUN_PREFIX}-ollama.log" 2>&1 &
OLLAMA_PID="$!"
for _ in $(seq 1 60); do
  if ollama list >/dev/null 2>&1; then
    break
  fi
  sleep 1
done
if ! ollama list >/dev/null 2>&1; then
  echo "Ollama did not become ready for the serial phase" >&2
  exit 1
fi

"${VENV_DIR}/bin/python" -m repoexec_baseline.run_generation_matrix \
  --models "${MODELS[@]}" \
  --representations raw ast reduced_ast \
  --subset full_context \
  --task-limit "${TASK_LIMIT}" \
  --runs-root "${RUNS_ROOT}" \
  --run-prefix "${RUN_PREFIX}" \
  --num-return-sequences "${NUM_RETURN_SEQUENCES}" \
  --max-new-tokens 256 \
  --do-sample \
  --temperature 0.2 \
  --top-p 0.95 \
  --seed 42 \
  --ollama-base-url "${OLLAMA_BASE_URL}" \
  --ollama-keep-alive "${OLLAMA_KEEP_ALIVE}" \
  --ollama-num-ctx "${OLLAMA_NUM_CTX}" \
  --ollama-parallel-requests "${SERIAL_OLLAMA_REQUESTS}" \
  --parallel-tasks 1 \
  --session-start-epoch "${SESSION_START_EPOCH}"

echo "Generation is complete. Stop the FMLe instance after downloading the bundle."

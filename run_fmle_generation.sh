#!/usr/bin/env bash

set -Eeuo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SESSION_START_EPOCH="$(date +%s)"

FMLE_STORAGE_ROOT="${FMLE_STORAGE_ROOT:-/workspace}"
if [[ ! -d "${FMLE_STORAGE_ROOT}" || ! -w "${FMLE_STORAGE_ROOT}" ]]; then
  FMLE_STORAGE_ROOT="${PROJECT_ROOT}"
fi

VENV_DIR="${VENV_DIR:-${FMLE_STORAGE_ROOT}/repoexec-venv}"
RUNS_ROOT="${RUNS_ROOT:-${FMLE_STORAGE_ROOT}/repoexec-runs}"
TASK_LIMIT="${TASK_LIMIT:-30}"
NUM_RETURN_SEQUENCES="${NUM_RETURN_SEQUENCES:-5}"
RUN_PREFIX="${RUN_PREFIX:-fmle-generation30}"
PULL_MODELS="${PULL_MODELS:-1}"
INSTALL_OLLAMA="${INSTALL_OLLAMA:-1}"

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

if ! command -v python3 >/dev/null 2>&1; then
  echo "python3 is required" >&2
  exit 1
fi
mkdir -p "${HF_DATASETS_CACHE}" "${OLLAMA_MODELS}" "${RUNS_ROOT}"

if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
  python3 -m venv "${VENV_DIR}"
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
  --session-start-epoch "${SESSION_START_EPOCH}"

echo "Generation is complete. Stop the FMLe instance after downloading the bundle."

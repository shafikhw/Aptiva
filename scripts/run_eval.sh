#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BASELINE="${BASELINE:-full-agent}"
SYSTEM="${SYSTEM:-all}"
RESULTS_PATH="${RESULTS_PATH:-"$ROOT_DIR/eval/results.json"}"

echo "Running eval (baseline=${BASELINE}, system=${SYSTEM}) -> ${RESULTS_PATH}"
PYTHONHASHSEED="${PYTHONHASHSEED:-0}" python3 "${ROOT_DIR}/eval/run_eval.py" \
  --baseline "${BASELINE}" \
  --system "${SYSTEM}" \
  --results-path "${RESULTS_PATH}"

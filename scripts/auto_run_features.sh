#!/bin/bash
# ══════════════════════════════════════════════════════════════
# Auto-restart wrapper for router feature generation.
#
# Automatically retries on failure (e.g. HPC scheduler kill,
# OOM, timeout).  The underlying script supports resume, so
# each restart picks up where it left off.
#
# Usage:
#   nohup bash scripts/auto_run_features.sh &> auto_features.log &
#   MAX_RETRIES=100 RETRY_DELAY=60 bash scripts/auto_run_features.sh
#
# All env vars (BATCH_SIZE, K, etc.) are forwarded to the inner script.
# ══════════════════════════════════════════════════════════════
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_DIR"

MAX_RETRIES="${MAX_RETRIES:-50}"
RETRY_DELAY="${RETRY_DELAY:-30}"
INNER_SCRIPT="${INNER_SCRIPT:-scripts/run_router_features_humaneval.sh}"
OUTPUT="${OUTPUT:-data/router_features/humaneval_noisy_heuristic.jsonl}"

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
CYAN='\033[0;36m'; BOLD='\033[1m'; NC='\033[0m'

for attempt in $(seq 1 "$MAX_RETRIES"); do
    echo -e "\n${BOLD}═══ Attempt ${attempt}/${MAX_RETRIES} — $(date '+%Y-%m-%d %H:%M:%S') ═══${NC}"

    if [[ -f "$OUTPUT" ]]; then
        records=$(wc -l < "$OUTPUT")
        echo -e "${CYAN}[INFO]${NC}  Output has ${records} records so far"
    fi

    bash "$INNER_SCRIPT" "$@"
    exit_code=$?

    if [[ $exit_code -eq 0 ]]; then
        echo -e "\n${GREEN}[OK]${NC}    Completed successfully on attempt ${attempt}"
        exit 0
    fi

    echo -e "${YELLOW}[WARN]${NC}  Exit code ${exit_code}. Retrying in ${RETRY_DELAY}s..."
    sleep "$RETRY_DELAY"
done

echo -e "\n${RED}[ERR]${NC}   All ${MAX_RETRIES} attempts exhausted"
exit 1

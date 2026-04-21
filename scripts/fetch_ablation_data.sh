#!/bin/bash
# Fetch ablation feature files from shared lab storage.
#
# Feature JSONL files are NOT tracked in git (too large ~140MB total).
# They live on two lab machines and can be pulled with this script.
#
# Usage (from project root):
#   bash scripts/fetch_ablation_data.sh
#   bash scripts/fetch_ablation_data.sh --source hyperturing1
#
# Prerequisites: SSH access to skampere2 or hyperturing1

set -euo pipefail

SOURCE="${1:-skampere2}"
LOCAL_DIR="data/router_features/final_ablations"
mkdir -p "$LOCAL_DIR"

FILES=(
    humaneval_lam_0.05.jsonl
    humaneval_lam_0.2.jsonl
    humaneval_lam_0.5.jsonl
    humaneval_lam_1.0.jsonl
    humaneval_no_consistency.jsonl
    textworld_lam_0.05.jsonl
    textworld_lam_0.2.jsonl
    textworld_lam_0.5.jsonl
    textworld_lam_1.0.jsonl
    textworld_no_consistency.jsonl
)

case "$SOURCE" in
    skampere2)
        REMOTE_USER="srivatsavad"
        REMOTE_DIR="/lfs/skampere2/0/srivatsavad/risk-routing-verifier-objective/data/router_features"
        ;;
    hyperturing1)
        REMOTE_USER="srivatsavad"
        REMOTE_DIR="/dfs/scratch0/srivatsavad/risk-routing-verifier-objective/data/router_features"
        ;;
    *)
        echo "Unknown source: $SOURCE. Use skampere2 or hyperturing1."
        exit 1
        ;;
esac

echo "Fetching ablation feature files from ${SOURCE}:${REMOTE_DIR}/"
echo ""

for f in "${FILES[@]}"; do
    DEST="${LOCAL_DIR}/${f}"
    if [[ -f "$DEST" ]]; then
        LINES=$(wc -l < "$DEST")
        echo "  [skip] $f already exists ($LINES lines)"
        continue
    fi
    echo "  [pull] $f ..."
    scp "${REMOTE_USER}@${SOURCE}.stanford.edu:${REMOTE_DIR}/${f}" "$DEST"
    LINES=$(wc -l < "$DEST")
    echo "         -> $LINES lines"
done

echo ""
echo "Done. Files in $LOCAL_DIR:"
ls -lh "$LOCAL_DIR"/*.jsonl 2>/dev/null | awk '{print "  " $5, $9}'

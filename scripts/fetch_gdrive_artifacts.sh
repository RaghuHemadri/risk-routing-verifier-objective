#!/bin/bash
# Download large artifacts from Google Drive.
#
# These files are NOT tracked in git (~23 GB total).  They were uploaded
# with rclone from the project root after the ablation sweep on 2026-04-21.
#
# Prerequisites:
#   brew install rclone
#   rclone config   # name the remote "gdrive", authorize Google Drive OAuth
#
# Usage (from project root):
#   bash scripts/fetch_gdrive_artifacts.sh              # download everything
#   bash scripts/fetch_gdrive_artifacts.sh --only router_features
#   bash scripts/fetch_gdrive_artifacts.sh --only policy
#   bash scripts/fetch_gdrive_artifacts.sh --only qwen_adapters
#   bash scripts/fetch_gdrive_artifacts.sh --only trajectories
#
# Re-running is safe: rclone skips files that already exist and match size/checksum.

set -euo pipefail

REMOTE="gdrive"
FOLDER_ID="1AzTpx_xOy-46MtfgkmACIB6TvgIyvFDn"
RCLONE_FLAGS="--transfers 8 --progress --drive-root-folder-id ${FOLDER_ID}"

# Parse --only flag
ONLY="${2:-all}"
if [[ "${1:-}" == "--only" ]]; then
    ONLY="${2:?--only requires an argument: router_features|policy|qwen_adapters|trajectories}"
fi

check_rclone() {
    if ! command -v rclone &>/dev/null; then
        echo "rclone not found. Install with: brew install rclone"
        echo "Then authorize: rclone config"
        exit 1
    fi
    if ! rclone listremotes 2>/dev/null | grep -q "^${REMOTE}:"; then
        echo "No rclone remote named '${REMOTE}' found."
        echo "Run: rclone config  (name the remote 'gdrive', choose Google Drive)"
        exit 1
    fi
}

download() {
    local src="$1"   # path inside Drive folder
    local dst="$2"   # local destination directory
    echo ""
    echo "--- Downloading ${src} -> ${dst}/ ---"
    mkdir -p "${dst}"
    rclone copy "${REMOTE}:${src}" "${dst}" ${RCLONE_FLAGS}
    echo "    Done: $(du -sh "${dst}" | cut -f1)"
}

check_rclone

echo "=== Fetching R2V artifacts from Google Drive ==="
echo "    Remote: ${REMOTE}: (folder ${FOLDER_ID})"
echo "    Target: $(pwd)"
echo ""

case "${ONLY}" in
    router_features)
        download "data/router_features" "data/router_features"
        ;;
    policy)
        download "outputs/policy" "outputs/policy"
        ;;
    qwen_adapters)
        download "qwen7_humaneval" "qwen7_humaneval"
        download "qwen7_textworld" "qwen7_textworld"
        ;;
    trajectories)
        download "updated_data" "updated_data"
        ;;
    all)
        download "data/router_features"  "data/router_features"
        download "outputs/policy"        "outputs/policy"
        download "qwen7_humaneval"       "qwen7_humaneval"
        download "qwen7_textworld"       "qwen7_textworld"
        download "updated_data"          "updated_data"
        ;;
    *)
        echo "Unknown --only value: ${ONLY}"
        echo "Valid options: router_features, policy, qwen_adapters, trajectories, all"
        exit 1
        ;;
esac

echo ""
echo "=== Done ==="

#!/usr/bin/env bash
# Convert HDF5 dataset to LeRobot v3.0 format.
#
# Steps:
#   1. HDF5 → v2.0  (hdf5_to_lerobot pipeline, driven by a config JSON)
#   2. v2.0 → v2.1  (scripts/lerobot_v033_v21/convert_dataset_v20_to_v21.py)
#   3. v2.1 → v3.0  (lerobot.datasets.v30.convert_dataset_v21_to_v30)
#
# Usage:
#   ./scripts/convert_to_v30.sh --config configs/default.json --repo-id kowyo/my-dataset
#   ./scripts/convert_to_v30.sh --config configs/default.json --repo-id kowyo/my-dataset --push-to-hub
#   ./scripts/convert_to_v30.sh --config configs/default.json --repo-id kowyo/my-dataset --skip-cleaning

set -euo pipefail

# ── Defaults ──────────────────────────────────────────────────────────────────
CONFIG="configs/default.json"
REPO_ID=""
PUSH_TO_HUB=false
SKIP_CLEANING=false
TMP_ROOT=""

# ── Cleanup trap ───────────────────────────────────────────────────────────────
cleanup() { [[ -n "$TMP_ROOT" && -d "$TMP_ROOT" ]] && rm -rf "$TMP_ROOT"; }
trap cleanup EXIT INT TERM

# ── Helpers ────────────────────────────────────────────────────────────────────
step() { echo ""; echo "▶ $*"; echo "────────────────────────────────────────────────────────────────────────────────"; }

usage() {
    echo "Usage: $0 --repo-id <user/dataset> [--config <path>] [--push-to-hub] [--skip-cleaning]"
    exit 1
}

# ── Argument parsing ───────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --config)        CONFIG="$2";       shift 2 ;;
        --repo-id)       REPO_ID="$2";      shift 2 ;;
        --push-to-hub)   PUSH_TO_HUB=true;  shift   ;;
        --skip-cleaning) SKIP_CLEANING=true; shift   ;;
        *) echo "Unknown argument: $1"; usage ;;
    esac
done

[[ -z "$REPO_ID" ]] && { echo "[ERROR] --repo-id is required"; usage; }
[[ -f "$CONFIG" ]]  || { echo "[ERROR] Config file not found: $CONFIG"; exit 1; }

# ── Read output_root from config ───────────────────────────────────────────────
OUTPUT_ROOT=$(python3 -c "import json; print(json.load(open('$CONFIG'))['output_root'])")
NUM_WORKERS=$(python3 -c "import json; print(json.load(open('$CONFIG')).get('num_workers', 1))")
[[ -n "$OUTPUT_ROOT" ]] || { echo "[ERROR] Could not read output_root from $CONFIG"; exit 1; }

echo "════════════════════════════════════════════════════════════════════════════════"
echo "  Config:      $CONFIG"
echo "  Output root: $OUTPUT_ROOT"
echo "  Workers:     $NUM_WORKERS"
echo "  Repo ID:     $REPO_ID"
echo "  Push to hub: $PUSH_TO_HUB"
echo "  Skip clean:  $SKIP_CLEANING"
echo "════════════════════════════════════════════════════════════════════════════════"

# ── Step 1: HDF5 → v2.0 ───────────────────────────────────────────────────────
step "Step 1/3 — HDF5 → v2.0"
if [[ "$SKIP_CLEANING" == "true" ]]; then
    uv run python -m hdf5_to_lerobot --config "$CONFIG" --skip-cleaning
else
    uv run python -m hdf5_to_lerobot --config "$CONFIG"
fi

# ── Step 2: v2.0 → v2.1 ───────────────────────────────────────────────────────
step "Step 2/3 — v2.0 → v2.1"
uv run python scripts/lerobot_v033_v21/convert_dataset_v20_to_v21.py --local-root "$OUTPUT_ROOT" --num-workers "$NUM_WORKERS"

# ── Step 3: v2.1 → v3.0 ───────────────────────────────────────────────────────
step "Step 3/3 — v2.1 → v3.0"

# The v3.0 script expects the dataset at <root>/<repo_id>, so we create a
# temporary directory with a symlink matching that structure.
TMP_ROOT=$(mktemp -d)
mkdir -p "$TMP_ROOT/$(dirname "$REPO_ID")"
ln -s "$(pwd)/$OUTPUT_ROOT" "$TMP_ROOT/$REPO_ID"

if [[ "$PUSH_TO_HUB" == "true" ]]; then
    step "Ensuring Hugging Face dataset repo exists"
    hf repo create "$REPO_ID" --repo-type dataset --exist-ok
fi

uv run python -m lerobot.datasets.v30.convert_dataset_v21_to_v30 \
    --repo-id="$REPO_ID" \
    --root="$TMP_ROOT" \
    --push-to-hub=false

step "Replacing intermediate data with v3.0 output"
[[ -d "$TMP_ROOT/$REPO_ID" ]] || { echo "[ERROR] v3.0 output not found at $TMP_ROOT/$REPO_ID"; exit 1; }
rm -rf "$OUTPUT_ROOT"
mv "$TMP_ROOT/$REPO_ID" "$OUTPUT_ROOT"
rm -rf "$TMP_ROOT"; TMP_ROOT=""  # cleared so cleanup trap is a no-op

# ── Push & tag ─────────────────────────────────────────────────────────────────
if [[ "$PUSH_TO_HUB" == "true" ]]; then
    step "Pushing v3.0 dataset to hub"
    hf upload-large-folder "$REPO_ID" "$OUTPUT_ROOT" --repo-type dataset

    step "Augmenting dataset with quantile stats"
    uv run python -m lerobot.datasets.v30.augment_dataset_quantile_stats \
        --repo-id="$REPO_ID" \
        --root="$OUTPUT_ROOT"

    step "Tagging dataset as v1.0"
    hf repo tag create "$REPO_ID" v1.0 --repo-type dataset
fi

echo ""
echo "════════════════════════════════════════════════════════════════════════════════"
echo "  Done! v3.0 dataset written to: $OUTPUT_ROOT"
if [[ "$PUSH_TO_HUB" == "true" ]]; then
    echo "  Pushed to:  https://huggingface.co/datasets/$REPO_ID"
fi
echo "════════════════════════════════════════════════════════════════════════════════"

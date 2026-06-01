#!/bin/bash
# ============================================================
# Download RunPod training artifacts → external SSD
# ============================================================
# Pulls checkpoints/, results/, logs/ from /workspace/Paper5/
# on the pod into /Volumes/SSD Ex/GENESIS_runpod/<variant>/
#
# Runs on Mac ONLY. Does not touch the Mac-local Paper5/ repo,
# so the Mac storage stays clean.
#
# Usage:
#   ./download_from_pod.sh <pod-ip> <variant>
#
# Examples:
#   ./download_from_pod.sh 1.2.3.4 small
#   ./download_from_pod.sh 1.2.3.4 base
#   ./download_from_pod.sh 1.2.3.4 large
#   ./download_from_pod.sh 1.2.3.4 all    # pulls everything on the pod
# ============================================================

set -e

POD_IP="${1:-}"
VARIANT="${2:-}"

if [ -z "$POD_IP" ] || [ -z "$VARIANT" ]; then
    echo "Usage: $0 <pod-ip> <small|base|large|all>"
    exit 1
fi

SSD_ROOT="/Volumes/SSD Ex/GENESIS_runpod"

if [ ! -d "/Volumes/SSD Ex" ]; then
    echo "ERROR: /Volumes/SSD Ex not mounted. Plug in SSD and retry."
    exit 1
fi

# Per-variant dest keeps Small / Base / Large cleanly separated so you can
# terminate one pod, plug the SSD, start fresh for the next variant, and
# nothing overwrites. Shared results/ dir merges all eval_{size}.json files
# from different runs without conflict.
DEST="$SSD_ROOT/$VARIANT"
mkdir -p "$DEST/checkpoints" "$DEST/logs"
mkdir -p "$SSD_ROOT/results"

echo "=============================================="
echo "Downloading pod:$POD_IP → $DEST"
echo "=============================================="

case "$VARIANT" in
    small|base|large)
        # Only this variant's checkpoint dirs
        echo ">>> checkpoints/${VARIANT}_stage1/, ${VARIANT}_stage2/, diffusion_${VARIANT}/"
        rsync -avz --progress \
            "root@$POD_IP:/workspace/Paper5/checkpoints/${VARIANT}_stage1/" \
            "$DEST/checkpoints/${VARIANT}_stage1/"
        rsync -avz --progress \
            "root@$POD_IP:/workspace/Paper5/checkpoints/${VARIANT}_stage2/" \
            "$DEST/checkpoints/${VARIANT}_stage2/"
        rsync -avz --progress \
            "root@$POD_IP:/workspace/Paper5/checkpoints/diffusion_${VARIANT}/" \
            "$DEST/checkpoints/diffusion_${VARIANT}/"
        ;;
    all)
        # Everything in checkpoints/
        echo ">>> all checkpoints/"
        rsync -avz --progress \
            "root@$POD_IP:/workspace/Paper5/checkpoints/" \
            "$DEST/checkpoints/"
        ;;
    *)
        echo "ERROR: variant must be small, base, large, or all"
        exit 1
        ;;
esac

echo ""
echo ">>> results/ (merged — scaling_summary.json aggregates across variants)"
rsync -avz --progress \
    "root@$POD_IP:/workspace/Paper5/results/" \
    "$SSD_ROOT/results/"

echo ""
echo ">>> logs/"
rsync -avz --progress \
    "root@$POD_IP:/workspace/Paper5/logs/" \
    "$DEST/logs/"

echo ""
echo "=============================================="
echo "Download complete."
echo "  Checkpoints: $DEST/checkpoints/"
echo "  Results:     $SSD_ROOT/results/"
echo "  Logs:        $DEST/logs/"
echo ""
echo "Disk usage on SSD:"
du -sh "$SSD_ROOT"/*/ 2>/dev/null || true
echo ""
echo "Safe to terminate the pod now. The Mac repo is unchanged."
echo "=============================================="



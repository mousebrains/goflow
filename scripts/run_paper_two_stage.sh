#!/bin/bash
# Two-stage GOFLOW training reproducing the paper recipe.
#
# Stage 0: pure L1 loss for 100 epochs (c_spec=0)
# Stage 1: L1 + spectral for 50 epochs (c_spec=0.2 — paper Methods "Loss
#   function design" section, scanned 0.05-0.9 and chose 0.2 as compromise)
#
# Inputs (must exist before running):
#   data/paper/llcGoes_gradT_trunc.nc   (~125 GB, paper LLC training file)
#   data/paper/goes_nesma.nc            (6.2 GB, paper GOES validation)
#
# Outputs:
#   data/run_paper/                     run-specific directory
#   data/run_paper/lgt_unet16_1_3_0.0cs.pth   stage-0 checkpoint
#   data/run_paper/lgt_unet16_1_3_0.5cs.pth   stage-1 checkpoint
#   data/run_paper/run_metrics.json    repro metrics (per-epoch + selected)
#   data/run_paper/run.log             tee'd stdout/stderr
#
# Wrapped in caffeinate -i so the Mac stays awake (lower jetsam priority).
set -euo pipefail

cd "$(dirname "$0")/.."
ROOT=$(pwd)

LLC=$ROOT/data/paper/llcGoes_gradT_trunc.nc
GOES=$ROOT/data/paper/goes_nesma.nc
OUT=$ROOT/data/run_paper

if [ ! -f "$LLC" ]; then
    echo "ERROR: missing $LLC" >&2
    echo "       Resolve the Drive quota (see docs/region_prep_recipe.md)" >&2
    exit 1
fi
if [ ! -f "$GOES" ]; then
    echo "ERROR: missing $GOES" >&2
    exit 1
fi

mkdir -p "$OUT"
LOG=$OUT/run.log
echo "Logging to $LOG"

caffeinate -i python3 train_twostage.py \
    --llc_file "$LLC" \
    --goes_file "$GOES" \
    --output_dir "$OUT/" \
    --model unet --nbase 16 \
    --epochs_stage1 100 \
    --epochs_stage2 50 \
    --c_spec_stage2 0.2 \
    --metrics_file "$OUT/run_metrics.json" \
    --skip_eval_nc \
    2>&1 | tee "$LOG"

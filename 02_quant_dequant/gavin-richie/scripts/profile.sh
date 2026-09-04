#!/usr/bin/env bash
# Collect Nsight Compute / Nsight Systems reports for the quantize and
# dequantize kernels. Output .ncu-rep / .nsys-rep files are portable: they
# can be opened on a Windows host with the Nsight Compute / Nsight Systems
# GUI (install a version >= the collecting CLI).
#
# Usage: profile.sh /path/to/lowp TENSOR.bin CONFIG.txt OUTDIR
set -euo pipefail

LOWP=${1:?path to lowp binary}
TENSOR=${2:?tensor file}
CONFIG=${3:?config file}
OUT=${4:?output dir}
BIN=$(basename "${TENSOR%.*}")

mkdir -p "$OUT"

# --- Nsight Compute: per-kernel metrics (memory throughput, occupancy...)
# NOTE: ncu needs GPU performance-counter permission (ERR_NVGPUCTRPERM on
# WSL2 / unconfigured hosts). On such hosts this step is skipped.
if ! ncu --set full \
       -k 'regex:quant|dequant' \
       --target-processes all \
       -o "$OUT/${BIN}_kernels" \
       "$LOWP" run --input "$TENSOR" --config "$CONFIG" 2>&1 |
     tee "$OUT/${BIN}_ncu.log" | grep -q ERR_NVGPUCTRPERM; then
  ncu --import "$OUT/${BIN}_kernels.ncu-rep" --page details | head -120
else
  echo "ncu skipped: no GPU perf-counter permission (see $OUT/${BIN}_ncu.log)"
fi

# --- Nsight Systems: timeline of the whole pipeline
nsys profile -o "$OUT/${BIN}_pipeline" \
    "$LOWP" run --input "$TENSOR" --config "$CONFIG"
nsys stats "$OUT/${BIN}_pipeline.nsys-rep" | head -60

echo "reports written to $OUT (copy to Windows and open with Nsight GUIs)"

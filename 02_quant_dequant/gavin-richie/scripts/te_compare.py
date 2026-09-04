#!/usr/bin/env python3
"""Transformer Engine (per-tensor FP8) quantize/dequantize benchmark.

Reads the same .bin tensor format as the C++ pipeline
(int64 rows, int64 cols, char[4] dtype, row-major data), quantizes it
through Transformer Engine's FP8 cast kernels (Float8Tensor.quantize_,
dequantize) with a per-tensor CurrentScaling-style scale, and reports the
same metrics as `lowp run --log` (max_abs_error / mae / mse /
compression_ratio / kernel ms / bandwidth) so the software-simulated
MXFP8/NVFP4 path can be compared against the production TE path.

Note on scope: TE 1.14 (this container) only ships per-tensor FP8 recipes
(DelayedScaling / CurrentScaling). The MXFP8/NVFP4 block-scaling recipes
need TE >= 2.0 and Blackwell (sm_100) hardware; on this Ada GPU the
comparison is therefore per-tensor-scale E4M3 vs the software 32/16-elem
block-scale simulation.

Usage:
    te_compare.py TENSOR.bin [--format e4m3|e5m2] [--dist LABEL]
                             [--out LOGFILE] [--warmup N] [--iters N]
"""
import argparse
import struct
import sys

import numpy as np
import torch

import transformer_engine  # noqa: F401  (must import before the C extension)
import transformer_engine_torch as tex
from transformer_engine.pytorch.tensor.float8_tensor import Float8Tensor

FP8_MAX = {"e4m3": 448.0, "e5m2": 57344.0}
FP8_DTYPE = {
    "e4m3": tex.DType.kFloat8E4M3,
    "e5m2": tex.DType.kFloat8E5M2,
}


def read_tensor(path):
    with open(path, "rb") as f:
        rows, cols = struct.unpack("<qq", f.read(16))
        dtype = f.read(4).decode()
        if dtype != "fp32":
            raise ValueError(f"te_compare expects fp32 input, got {dtype}")
        data = np.fromfile(f, dtype="<f4", count=rows * cols)
    if data.size != rows * cols:
        raise ValueError(f"truncated tensor file: {path}")
    return data.reshape(rows, cols), dtype


def quantize_amax_scale(x):
    """Per-tensor amax + scale, CurrentScaling semantics (guard amax=0)."""
    amax = x.abs().amax().reshape(1)
    scale = 448.0 / amax if ARGS.format == "e4m3" else FP8_MAX[ARGS.format] / amax
    scale = torch.nan_to_num(scale, nan=0.0, posinf=0.0, neginf=0.0)
    return amax, scale


def time_kernel(fn, warmup, iters):
    start = torch.cuda.Event(enable_timing=True)
    stop = torch.cuda.Event(enable_timing=True)
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start.record()
    for _ in range(iters):
        fn()
    stop.record()
    torch.cuda.synchronize()
    return start.elapsed_time(stop) / iters


def main():
    tensor, dtype = read_tensor(ARGS.tensor)
    rows, cols = tensor.shape
    n = rows * cols

    x = torch.from_numpy(tensor).to("cuda", dtype=torch.float32)
    fp8_dtype = FP8_DTYPE[ARGS.format]

    # --- correctness (fp64 metrics on GPU) ---
    amax, scale = quantize_amax_scale(x)
    data = torch.empty(rows, cols, dtype=torch.uint8, device="cuda")
    scale_inv = torch.empty(1, device="cuda")
    q = Float8Tensor(data=data, fp8_dtype=fp8_dtype,
                     fp8_scale_inv=scale_inv, dtype=torch.float32)
    q.quantize_(x, scale=scale, amax=amax)
    dq = q.dequantize(dtype=torch.float16)

    diff = dq.double() - x.double()
    max_abs_error = diff.abs().max().item()
    mae = diff.abs().mean().item()
    mse = (diff * diff).mean().item()

    # --- throughput ---
    in_bytes = n * 4.0                       # fp32 input, same accounting as main.cu
    q_bytes = n * 1.0 + 8.0                  # fp8 payload + fp32 scale/amax
    compression = in_bytes / q_bytes

    def quant_full():
        a, s = quantize_amax_scale(x)
        q.quantize_(x, scale=s, amax=a)

    quant_ms = time_kernel(quant_full, ARGS.warmup, ARGS.iters)
    quant_cast_ms = time_kernel(
        lambda: q.quantize_(x, scale=scale, amax=amax), ARGS.warmup, ARGS.iters)
    dequant_ms = time_kernel(
        lambda: q.dequantize(dtype=torch.float16), ARGS.warmup, ARGS.iters)

    lines = [
        "# Transformer Engine per-tensor FP8 (CurrentScaling-style scale,",
        f"# TE cast kernels; format = fp8-{ARGS.format}; TE 1.14 lacks the",
        "# MXFP8/NVFP4 block-scaling recipes, those need TE>=2.0 + sm_100)",
        f'tensor = "{ARGS.tensor}"',
        f'distribution = "{ARGS.dist}"',
        f'format = "te-fp8-{ARGS.format}"',
        "scale_mode = \"tensor\"",
        f"input_dtype = \"fp32\"",
        "output_type = \"fp16\"",
        f"rows = {rows}",
        f"cols = {cols}",
        f"num_elements = {n}",
        f"max_abs_error = {max_abs_error:.6f}",
        f"mae = {mae:.6f}",
        f"mse = {mse:.6f}",
        f"compression_ratio = {compression:.4f}",
        f"quantize_kernel_ms = {quant_ms:.4f}",
        f"quantize_cast_only_ms = {quant_cast_ms:.4f}",
        f"dequantize_kernel_ms = {dequant_ms:.4f}",
        f"quantize_bandwidth_gbps = {(in_bytes + q_bytes) / (quant_ms * 1e-3) / 1e9:.4f}",
        f"dequantize_bandwidth_gbps = {(in_bytes + q_bytes) / (dequant_ms * 1e-3) / 1e9:.4f}",
    ]
    report = "\n".join(lines) + "\n"
    if ARGS.out:
        with open(ARGS.out, "w") as f:
            f.write(report)
    print(report)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("tensor", help="input .bin tensor (fp32)")
    parser.add_argument("--format", choices=sorted(FP8_MAX), default="e4m3")
    parser.add_argument("--dist", default="", help="distribution label for the log")
    parser.add_argument("--out", default="", help="write report to this file")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    ARGS = parser.parse_args()
    sys.exit(main())

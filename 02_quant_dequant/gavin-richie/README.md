# MXFP8 / NVFP4 低精度模拟与反量化（CUDA 软件实现）

不依赖任何 FP8/FP4 硬件指令（兼容 Maxwell 及以上普通 CUDA GPU）的低精度格式
软件模拟：量化、4-bit 打包、缩放、解包、反量化与误差评估。

## 构建

需要 CUDA Toolkit >= 11.5（建议 12.x）、CMake >= 3.18。

```bash
cmake -S . -B build -DCMAKE_CUDA_ARCHITECTURES=89   # 4060/4090 (Ada)
cmake --build build -j
cd build && ctest          # 单元测试 + GPU roundtrip 测试
```

架构对照：3090=86，4060/4090=89，H20=90，5090=120。
默认 `CMAKE_CUDA_ARCHITECTURES=89`。

## 运行

```bash
# 生成测试张量（random / normal / outlier 分布，fp32 或 fp16）
python3 scripts/gen_tensor.py artifacts/t.bin 4096 4096 fp32 normal

# 端到端：量化 + 反量化 + 误差/性能日志
./build/lowp run --input artifacts/t.bin --config q.txt \
    --out-weight w.bin --out-dq out.bin --log log.txt --dist normal

# 或分步：先量化保存权重，再从权重文件反量化
./build/lowp quant   --input artifacts/t.bin --config q.txt --out-weight w.bin
./build/lowp dequant --weight w.bin --config q.txt --out-dq out.bin
```

量化参数文件（`q.txt`，python 风格键值对）：

```python
format = "mxfp8"      # mxfp8 (E4M3) / mxfp8-e5m2 / nvfp4
block_size = 32       # mxfp8 默认 32, nvfp4 默认 16
scale_mode = "block"  # tensor / block
output_type = "fp16"  # fp16 / bf16 / fp32
rounding = "nearest"  # nearest / stochastic
target_gpu = "RTX 4060"
```

## 文件格式

- 张量文件：`int64 rows; int64 cols; char dtype[4] ("fp32"/"fp16"); 行主序数据`
- 低精度权重文件：`WeightFileHeader`（magic "QWHT"，见
  `include/quant_types.h`）+ packed data + scale 数组 + fp32 scale。
  NVFP4 严格 2 元素/字节（元素 2k 在低半字节）。
- 误差/性能日志：max abs error、MAE、MSE、压缩率、量化/反量化 kernel
  时间（cudaEvent）、有效内存带宽（GB/s）。

## Profiling（ncu / nsys）

```bash
scripts/profile.sh ./build/lowp artifacts/t.bin q.txt artifacts/prof
```

生成的 `.nsys-rep` / `.ncu-rep` 是跨平台文件，可直接在 Windows 宿主机上用
Nsight Systems / Nsight Compute GUI 打开（GUI 版本需 >= 采集端 CLI 版本）。
注意 WSL2 环境下 ncu 无法访问 GPU 性能计数器（ERR_NVGPUCTRPERM），
脚本会自动跳过并记录日志；nsys 时间线不受影响。

## 目录

- `include/fp_formats.h` — E4M3/E5M2/E2M1/E8M0 软件编解码（host/device）
- `src/quant_kernels.cu` — 量化/反量化/误差统计 CUDA kernels
- `src/file_io.cpp` — 张量/参数/权重文件 IO
- `src/main.cu` — CLI
- `tests/` — 编解码单元测试、文件 IO 测试、GPU roundtrip 测试
- `artifacts/` — 样例日志与 nsys 报告

详细设计与结果分析见 `REPORT.md`。

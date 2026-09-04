# 总结报告：MXFP8 / NVFP4 低精度软件模拟与反量化

作者：gavin-richie ｜ 开发 GPU：RTX 4060 (Ada, sm_89) ｜ CUDA 12.6
验证 GPU：RTX 4090 D (Ada, sm_89) ｜ CUDA 12.8（NGC 25.01 容器，原生 Linux）

## 1. 低精度格式与缩放策略

### 1.1 数值格式（全部为软件模拟，无硬件指令）

| 格式 | 位宽 | 指数/尾数 | 特殊值 | 最大有限值 |
|---|---|---|---|---|
| E4M3 | 8 | 4/3 | 仅 NaN（0x7F/0xFF），无 inf（OCP MX 规范，溢出饱和到 ±448） | ±448 |
| E5M2 | 8 | 5/2 | IEEE 风格 inf/NaN | ±57344 |
| E2M1 | 4 | 2/1 | 无 | ±6，可表示幅值 {0, .5, 1, 1.5, 2, 3, 4, 6} |
| E8M0 | 8 | 8/0 | 仅作 block scale，值 = 2^(code−127) | — |

编码统一采用 round-to-nearest-even（`rintf`/显式中点偶数规则），
`include/fp_formats.h` 为 host/device 双端实现，GPU 与 CPU 参考实现逐位一致
（由 `test_roundtrip` 验证）。舍入到 nearest 的实现要点：
- FP8 正常数域：尾数整数化 `q = |x|·2^mant−e`，`rintf` 舍入后若进位到下一
  binade 则指数 +1；
- 次正规数域：ulp = 2^(emin−mant)，整数编码自然覆盖"舍入后进位为最小正规数"；
- E2M1 查表舍入，中点（0.25/0.75/1.25/1.75/2.5/3.5/5）取偶数编码。

随机舍入（stochastic）：在目标网格 ulp 宽度内加均匀噪声
`y += (u−0.5)·ulp(y)`（`u` 为基于元素下标的确定性哈希），再走 nearest 编码。

### 1.2 缩放策略

- **per-tensor（fp32 标量）**：`S = amax_tensor / vmax`（vmax：E4M3=448，
  NVFP4=6）。量化 `q = enc(x/S)`，反量化 `x̂ = dec(q)·S`。
- **MXFP8 block（32 元素共享 E8M0）**：每块
  `S_b = 2^ceil(log2(amax_b/448))`，即 2 的幂且保证 `amax_b/S_b ≤ 448`，
  不溢出。存储 1 字节 E8M0 编码。
- **NVFP4 两级缩放（block=16）**：
  - 全局 fp32：`GS = max(1, amax_tensor/2688)`（2688 = 448×6）；
  - 块级 E4M3：`S_b = E4M3(amax_b/GS/6)`（先除全局再编码，保证块内幅值
    落入 E2M1 范围，且 E4M3 编码本身的误差 ≤ 2^-3 相对）；
  - 量化 `q = E2M1(x/GS/S_b)`，反量化 `x̂ = E2M1^{-1}(q)·S_b·GS`。

E5M2 与 E4M3 走同一 kernel，仅 vmax 与解码常量不同。

## 2. 打包布局与内存访问

- **MXFP8**：每元素 1 字节，天然对齐。
- **NVFP4**：每字节严格存 2 个元素（元素 2k 低半字节、2k+1 高半字节）。
  - 量化侧 packed store：每线程处理一个 16 元素 block，在寄存器中组装
    8 字节后以单次 8 字节（`unsigned long long`）store 写回；尾块不足 16
    元素时退化为字节写，高半字节补零。
  - 反量化侧 packed load：每线程加载 1 字节，解出 2 个 nibble、写 2 个输出，
    相邻线程访问连续字节，合并访存。
- 权重文件布局：`WeightFileHeader`（magic "QWHT"、format/scale_mode/
  block_size、rows/cols、packed 字节数、scale 数量、E8M0/E4M3 区分、是否带
  NVFP4 全局 scale）+ packed data + block scale 数组 + fp32 scale（tensor
  scale 恒有，NVFP4 block 模式额外带 global scale）。

## 3. Kernel 设计

- 量化：每线程一个 block——先块内求 amax（值域缩放），再逐元素编码写回；
  block scale 就地写出。
- 反量化：每线程 1（MXFP8）/ 2（NVFP4）元素，packed load → 解码 → 乘 scale
  → 按 output_type（fp16/bf16/fp32）写出。
- 误差统计：单 kernel 归约 max abs / MAE / MSE（double 累加原子）。
- 计时：cudaEvent 只包住量化/反量化 kernel 本身；有效带宽
  =（读+写字节数）/时间。

## 4. 实测结果（RTX 4060 Laptop, 8GB, 4096×4096 fp32 输入, fp16 输出）

误差（nearest 舍入，block 缩放）：

| 分布 | 格式 | max abs | MAE | MSE | 压缩率 |
|---|---|---|---|---|---|
| normal | MXFP8-E4M3 | 0.2494 | 0.0180 | 0.00070 | 3.88× |
| normal | NVFP4 | 0.7427 | 0.0714 | 0.00904 | 7.11× |
| random(U[-10,10]) | MXFP8-E4M3 | 0.5000 | 0.1167 | 0.02620 | 3.88× |
| random | NVFP4 | 1.6250 | 0.4379 | 0.33828 | 7.11× |
| outlier | MXFP8-E4M3 | 16.00 | 0.0106 | 0.09497 | 3.88× |
| outlier | NVFP4 | 76.56 | 0.0128 | 0.13973 | 7.11× |

- 压缩率含 scale 开销：MXFP8 = 32/9 ≈ 3.88（1B scale/32B 数据），
  NVFP4 = 16/2.25 ≈ 7.11（1B scale/8B packed 数据）。
- outlier 分布中极少数大值（±500）由缩放吸收，主体小值（σ=0.01）保持高
  精度（MAE 0.013），但 max abs error 由离群点所在块决定——这正是 block
  scaling 相对 per-tensor 的优势；per-tensor + E4M3 时小值会被整体压低
  （对照日志 `log_ts.txt`：normal 分布 tensor 模式 MAE 0.024 > block 0.018）。

性能（同上配置，cudaEvent 计时）：

| 格式 | 量化 ms | 反量化 ms | 量化 GB/s | 反量化 GB/s |
|---|---|---|---|---|
| MXFP8-E4M3 | 1.44 | 0.52 | 58 | 162 |
| NVFP4 | 0.42 | 0.31 | 182 | 246 |

反量化 kernel 为纯流式访存，166–246 GB/s 已达该卡有效带宽的较大部分
（4060 Laptop 理论 ≈ 272 GB/s）。量化 kernel 因每 block 先做串行 amax
归约（32/16 次全局读），带宽低于反量化，是后续优化点。

### 4.1 RTX 4090 D 复测（同 4096×4096 fp32 输入、fp16 输出、seed=42）

误差与 4060 **逐位一致**——6 组 max abs / MAE / MSE 全部相同，量化数学
与架构无关得到实证。性能对照（cudaEvent 计时，同口径 `(输入 4B +
压缩后字节)/时间`）：

| 格式 | 指标 | 4060 Laptop (≈272 GB/s) | 4090 D (≈1008 GB/s) | 提升 |
|---|---|---|---|---|
| MXFP8 | 量化 GB/s | 58 | 178 | 3.0× |
| MXFP8 | 反量化 GB/s | 162 | 767 | 4.7× |
| NVFP4 | 量化 GB/s | 182 | 750 | 4.1× |
| NVFP4 | 反量化 GB/s | 246 | 1089 | 4.4× |

- 4090 D 反量化带宽超过显存规格峰值（1089–1167 GB/s > 1008 GB/s）：
  64 MB 张量整体驻留该卡 72 MB L2，读命中 L2 所致（outlier 分布最高
  1167 GB/s）。
- 量化 kernel 的串行 amax 归约在两代卡上都是瓶颈（仅 178–750 GB/s），
  与 4060 上的结论一致，仍为首要优化点。
- 完整日志见 `artifacts/4090d/log_*.txt`，nsys 时间线见
  `artifacts/4090d/profile_{mxfp8,nvfp4}/`（nsys 捕获的
  quant/dequant/metrics kernel 耗时与 cudaEvent 计时一致）。

### 4.2 Transformer Engine 对比（per-tensor FP8，RTX 4090 D）

容器预装 TransformerEngine 1.14，仅有 DelayedScaling/CurrentScaling
per-tensor recipe；MXFP8/NVFP4 block-scaling recipe 需要 TE ≥ 2.0 且
Blackwell（sm_100）硬件，本卡无法运行。用 `scripts/te_compare.py` 走
TE 的 `Float8Tensor.quantize_` / `dequantize` cast kernel（E4M3、fp16
输出、与软件路径同误差/带宽口径）：

| 分布 | TE max abs | TE MAE | 软件 MXFP8 MAE | TE 压缩率 | TE 量化/反量化 ms | 软件量化/反量化 ms |
|---|---|---|---|---|---|---|
| normal | 0.1911 | 0.0180 | 0.0180 | 4.00 | 0.270 / 0.032 | 0.475 / 0.110 |
| random | 0.3605 | 0.1105 | 0.1167 | 4.00 | 0.270 / 0.032 | 0.473 / 0.111 |
| outlier | 17.95 | 0.0114 | 0.0106 | 4.00 | 0.270 / 0.032 | 0.475 / 0.112 |

- 误差：normal 下 TE per-tensor 与软件 block-scale 几乎一致；outlier 下
  软件 block scaling 的 max abs（16.0 vs 17.95）与 MAE 均略优——块级
  缩放把离群点的影响隔离在所在 32 元素块内；软件压缩率 3.88 略低于
  TE 的 4.00（含 block scale 开销）。
- 吞吐：TE dequant kernel 约为软件实现的 3.4×（2592 vs 767 GB/s，同口径
  公式），quantize（含 amax/scale 归约）约 1.8×。差距主要来自 TE cast
  kernel 的向量化程度，正是第 8 节优化项的预期空间。
- MXFP8/NVFP4 recipe 的硬件对照待 Blackwell + TE ≥ 2.0 补做，
  `te_compare.py` 已为此预留 `--format` 扩展点。

## 5. 软件模拟与硬件路径边界

- **全部数值编解码（E4M3/E5M2/E2M1/E8M0）、打包、缩放、反量化均为纯软件
  模拟**：只用整数/浮点位运算与 `ldexpf/frexpf/rintf`，无 `__nv_fp8` 内建、
  无 FP8/FP4 Tensor Core、无 Hopper/Blackwell/Ampere+ 专属指令，可在任意
  sm_60+ GPU（含 4060）运行。
- CUDA Math API（`__nv_cvt_float_to_fp8` 等）与 Transformer Engine 硬件
  加速路径属于**加分项**。TE 对比已在 RTX 4090 D 完成 per-tensor FP8
  路径（见 4.2 节，`scripts/te_compare.py`，不改 C++ 代码）；CMake 的
  `ENABLE_TE_COMPARISON` 开关注释已指向该脚本。MXFP8/NVFP4 block-scaling
  recipe 的硬件对照需 TE ≥ 2.0 + Blackwell，届时换
  `-DCMAKE_CUDA_ARCHITECTURES=120` 重编即可，kernel 代码无需修改。
- 测试覆盖：编解码单元测试（含 0/饱和/次正规/中点偶数/NaN/全码空间往返）、
  文件 IO 往返、GPU 与 CPU 参考实现逐位一致的 roundtrip 测试（含奇数尾块、
  tensor/block × nearest/stochastic 组合）。

## 6. ncu / nsys 使用

- 采集脚本 `scripts/profile.sh`（nsys 时间线已验证，
  `artifacts/sample_pipeline.nsys-rep`）。
- **跨平台结论**：`.nsys-rep` 与 `.ncu-rep` 均为自包含二进制报告，可在
  Windows 宿主机安装的 Nsight Systems / Nsight Compute GUI 中直接
  File→Open 打开分析，不依赖生成平台；只需保证 Windows 端软件版本 ≥
  采集端版本（建议均装最新版），文件通过共享目录/网络复制即可。
- ncu 计数器在两套环境都不可用，但根因不同、脚本处理一致：WSL2 是驱动
  层限制；4090 D 容器是宿主机驱动 `RmProfilingAdminOnly=1` 且容器未授予
  `CAP_SYS_ADMIN`（uid=0 也无效）。需在宿主机执行
  `modprobe nvidia NVreg_RestrictProfilingToAdminUsers=0`，或以
  `--cap-add SYS_ADMIN`（配合 `--security-opt seccomp=unconfined`）启动
  容器后方可采集。`profile.sh` 将 `ERR_NVGPUCTRPERM` 识别为"跳过"而非
  失败，并保留日志。
- nsys 时间线已在 4090 D（原生 Linux 容器，Nsight Systems 2024.6.2）
  采集：`artifacts/4090d/profile_{mxfp8,nvfp4}/*.nsys-rep`。
- `profile.sh` 现产出五类文件（`.ncu-rep` / `_ncu_summary.txt` /
  `_ncu.log` / `.nsys-rep` / `_nsys_stats.txt`），对"ncu 未安装 /
  计数器无权限 / 其他失败"三种情况分别处理（跳过 / 跳过 / 报错退出）；
  kernel 过滤放宽为 `regex:quant|dequant|amax|metrics` 覆盖全部 6 个
  kernel。

## 7. 开发中发现的问题

1. FP8 编码器初版把 `emin` 混入了正常数域的尾数量化指数（`q` 的移位多了
   一个 `emin` 因子），且次正规域移位错 2×；全码空间往返测试
   （decode→encode 稳定性）暴露后修正。
2. `frexpf` 的第二个参数传 `nullptr` 会在运行期崩溃——E8M0 ceil 编码需要
   同时取尾数判断"恰为 2 的幂"，必须用真实指针。
3. WSL2 与 4090 D 容器下 ncu 计数器均不可用（根因不同，见第 6 节），
   这也是把"报告文件跨平台打开"作为分析路径的原因。
4. 定制版 TE 1.14 直接调 `tex.cast_to_fp8(..., scaling_mode)` 会在
   `CheckScaleTensor` 处整数除零崩溃（SIGFPE）；改走
   `Float8Tensor.quantize_(x, scale=, amax=)` 封装层正常。此外
   `ncu --import | head` 这类"长输出管道接 head"在 `set -o pipefail`
   脚本里会因 SIGPIPE（退出码 141）杀死整个脚本——先落盘再用 `head`
   截断显示才安全。

## 8. 未来工作

- 量化 kernel 的块内 amax 改为 warp shuffle 归约 + 每线程多 block（提高
  带宽利用率）；NVFP4 反量化改 `uint4`（16B=32 元素）向量化 packed load。
- TE per-tensor FP8 对比已完成（4.2 节）；待 Blackwell GPU + TE ≥ 2.0
  补 MXFP8/NVFP4 block-scaling recipe 的误差/吞吐对照，并评估
  `__nv_fp4` 内建与本模拟的一致性。
- 3090/4090/H20 多架构带宽标定，验证 kernel 的架构无关性。
- 国产平台（metax/moore/iluvatar）移植：kernel 仅用 CUDA C++ 基础特性，
  适配工作主要在 RUNTIME 宏层（参考仓库 master 分支 tester 抽象）。

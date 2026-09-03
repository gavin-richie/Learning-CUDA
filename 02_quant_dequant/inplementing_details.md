## 二. MXFP8 / NVFP4 低精度模拟与反量化（CUDA）

凭借你在InfiniTensor训练营的经历和人工智能与并行计算方面的出色成绩，你受雇于一家领先的A基础设施公司，负责为下一代大模型推理引擎开发低精度权重加载模块。公司正在评估MXFP8、NVFP4等格式在模型压缩和推理加速中的价值，但现实问题是：测试环境并不能提供Hopper、Blackwell 或更新架构的 GPU，而 目前手中的 GPU 只支持常规 CUDA kernel。

你需要实现一套不依赖新硬件指令的低精度格式软件模拟与反量化程序：在普通CUDA GPU上完成低精度编码、打包、缩放、解包、反量化与误差评估；如果有更高架构硬件，则可以额外对比 CUDA Math API、Transformer Engine 或硬件加速路径。

### 领域知识简介

FP8 常见格式包括 E4M3 和 E5M2。MXFP8 是一种 microscaling FP8 格式。NVFP4 使用 E2M1 的 4bit数值，并结合分层缩放恢复高精度值。具体介绍可参考本方向的相关课程和特定大咖课。更多详细信息需要各位学员自行调研。

### 任务内容

开发一个CUDA程序，实现低精度浮点格式的软件模拟与反量化，须支持MXFP8和NVFP4。普通FP8不作为基础验收要求，可作为调试参考、误差对照或扩展功能。程序需要完成以下功能：

1. 读取 FP32 或 FP16 输入矩阵；

2. 根据配置将输入矩阵量化为指定低精度格式，并按真实bit宽度打包存储；

3. 保存低精度数据和缩放因子；

4. 使用 CUDA kernel 将低精度数据反量化为 FP16、BF16 或 FP32；

5. 输出误差统计和性能日志。

### 输入文件定义

### 1.张量数据文件



```bash
[header] 
num_rows: int64 # 矩阵行数
num_cols: int64 # 矩阵列数
dtype: string4  # 输入数据类型： fp32 或 fp16 

[data] 
values: dtype[num_rows * num_cols] # 按行主序存储
```

### 2.量化参数文件



```python
 format = "mxfp8"    # mxfp8 / nvfp4
 block_size = 32    # mxfp8 默认为 32, nvfp4 默认为 16
 scale_mode = "block"    # tensor / block
 output_type = "fp16"    # fp16 / bf16 / fp32
 rounding = "nearest"    # nearest / stochastic
 target_gpu = "T4"    # 目标 GPU 型号，仅用于报告说明
```

### 输出文件定义

• 低精度权重文件：二进制文件，包含packed data、scale数组和必要的header信息。4bit格式必须做到每个字节存储两个元素，不能用 `uint8`存一个4bit元素。

• 反量化后的张量：二进制文件，按行主序存储，数据类型由 `output_type` 指定。

• 误差与性能日志：包含最大绝对误差、平均绝对误差（MAE）、均方误差（MSE）、压缩率、量化kernel时间、反量化kernel时间、有效内存带宽（GB/s）。

### 要求



**1. 功能正确性**：

◦ 正确实现 MXFP8 数据表示、block scaling 和反量化流程；

◦ 正确实现NVFP4的数值编码、元素局部缩放、全局缩放、打包、解包与反量化规则；

◦ 正确实现 per-tensor scaling 和 block-wise scaling；

◦ 对随机矩阵、正态分布矩阵和含异常值矩阵分别输出误差统计。

**2. 硬件要求**：

◦ 基础版本不得依赖 Hopper、Blackwell 或 Ampere 以上架构特性；

◦ 基础版本不得要求硬件原生支持 FP8 / FP4 Tensor Core；

◦ 如有支持FP8/FP4的GPU，可额外实现硬件或库路径作为加分项，但不能作为基础验收前提。

**3. 4bit打包与内存访问**：

◦ 4 bit 格式必须实现 packed load / packed store；

4. **平台适配**：默认需在英伟达平台上支持。每在一款国产平台上支持，额外加分。

### 需提交内容：

• 整个完成上述功能的程序（包含测试）

​	◦ 提交地址：Learning-CUDA 本季度项目分支（2026-summer-project)

​	◦ 提交要求：参照本文档开头项目阶段信息和简介-项目代码要求中的第四条

• 总结报告。报告需至少包括：

​	◦ 详细阐述低精度格式、缩放策略、打包布局和反量化公式◦ 详细阐述实现思路以及优化方法，欢迎包含自己的优化历程和记录，开发中发现的问题等（取决于质量可以加分）

​	◦ 最终的误差指标、压缩率、性能指标和分析

​	◦ 明确说明哪些实现为软件模拟，哪些实现依赖特定GPU架构或第三方库

​	◦ 未来可继续提升的地方（可选）

​	◦ 包含ncu和/或nsys使用和分析的加分
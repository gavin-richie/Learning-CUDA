#include <vector>
#include <cuda_fp16.h>
#include <float.h>
#include "../tester/utils.h"

template<typename T>
__global__ void rmsNormKernel(const T* __restrict__ input, 
  const T* __restrict__ weight, T* __restrict__ output,
  size_t hidden_dim, float eps){
    const size_t row = blockIdx.x;
    const T* row_input = input + row*hidden_dim;
    T* row_output = output+row*hidden_dim;
    // 动态 shared memory，用于 reduction
    extern __shared__ float sdata[];

    //1: 计算平方和
    float sum_bd=0.0f;
    for(size_t j=threadIdx.x; j<hidden_dim;j+=blockDim.x){
      float block_value = static_cast<float>(row_input[j]);
      sum_bd += block_value*block_value;
    }

    //每个线程将自己的部分和写入 shared memory
    sdata[threadIdx.x]=sum_bd;
    __syncthreads();

    for(unsigned int stride=blockDim.x/2; stride>0;stride>>=1){
      if(threadIdx.x<stride){
        sdata[threadIdx.x]+=sdata[threadIdx.x+stride];
      }
    __syncthreads();
    }
    
    //3: 计算 RMS 的倒数
    float mean_row = sdata[0]/static_cast<float>(hidden_dim);
    float rms_inv = rsqrtf(mean_row+eps);
    
    //4: 归一化 + 缩放
    for(size_t j=threadIdx.x; j<hidden_dim;j+=blockDim.x){
      float val = static_cast<float>(row_input[j]);
      float w = static_cast<float>(weight[j]);
      row_output[j] = static_cast<T>(val*rms_inv*w);
    }
}


/**
 * @brief Computes RMSNorm over the last dimension of a 2D tensor.
 *
 * The input is a row-major matrix with shape [rows, hidden_dim]. For each row
 * i and column j:
 *
 *   output[i, j] = input[i, j] * rsqrt(mean(input[i, :]^2) + eps) * weight[j]
 *
 * The output vector is preallocated with rows * hidden_dim elements.
 *
 * @tparam T Data type of input, weight, and output tensors.
 * @param[in] h_input Flattened input matrix of shape [rows, hidden_dim].
 * @param[in] h_weight Per-column scale vector of shape [hidden_dim].
 * @param[out] h_output Flattened output matrix of shape [rows, hidden_dim].
 * @param[in] rows Number of rows/tokens.
 * @param[in] hidden_dim Size of the normalized dimension.
 * @param[in] eps Numerical stability epsilon.
 */
template <typename T>
void rmsNorm(const std::vector<T>& h_input, const std::vector<T>& h_weight,
              std::vector<T>& h_output, size_t rows, size_t hidden_dim,
              float eps) {
  // TODO: Implement the rmsNorm function
  T *d_input, *d_output, *d_weight;
  size_t input_bytes = rows*hidden_dim*sizeof(T);
  size_t weight_bytes = hidden_dim*sizeof(T);
  
  RUNTIME_CHECK(cudaMalloc(&d_input, input_bytes));
  RUNTIME_CHECK(cudaMalloc(&d_weight, weight_bytes));
  RUNTIME_CHECK(cudaMalloc(&d_output, input_bytes));

  RUNTIME_CHECK(cudaMemcpy(d_input, h_input.data(), input_bytes, cudaMemcpyHostToDevice));
  RUNTIME_CHECK(cudaMemcpy(d_weight, h_weight.data(), weight_bytes, cudaMemcpyHostToDevice));

  //--------4.config kernel args
  int block_size = 1;
  while(block_size<hidden_dim&&block_size<256){
    block_size<<=1;
  }

  int grid_size = static_cast<int>(rows);
  size_t shared_mem = block_size*sizeof(float);

  rmsNormKernel<T><<<grid_size, block_size, shared_mem>>>(
    d_input, d_weight, d_output, hidden_dim, eps
  );

  RUNTIME_CHECK(cudaGetLastError());
  RUNTIME_CHECK(cudaDeviceSynchronize());

  RUNTIME_CHECK(cudaMemcpy(h_output.data(), d_output, input_bytes, cudaMemcpyDeviceToHost));
  
  //---------7. free mem
  RUNTIME_CHECK(cudaFree(d_input));
  RUNTIME_CHECK(cudaFree(d_weight));
  RUNTIME_CHECK(cudaFree(d_output));

}
// ============================================================
// 辅助函数
// ============================================================
__device__ __forceinline__ float warpReduceMax(float val) {
    for (int offset = 16; offset > 0; offset >>= 1)
        val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, offset));
    return val;
}

__device__ __forceinline__ float warpReduceSum(float val) {
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_xor_sync(0xffffffff, val, offset);
    return val;
}

__device__ __forceinline__ float blockReduceMax(float val, float* shared) {
    int lane = threadIdx.x % 32;
    int wid  = threadIdx.x / 32;
    val = warpReduceMax(val);
    if (lane == 0) shared[wid] = val;
    __syncthreads();
    val = (threadIdx.x < (blockDim.x + 31) / 32) ? shared[lane] : -FLT_MAX;
    if (wid == 0) val = warpReduceMax(val);
    return val;
}

__device__ __forceinline__ float blockReduceSum(float val, float* shared) {
    int lane = threadIdx.x % 32;
    int wid  = threadIdx.x / 32;
    val = warpReduceSum(val);
    if (lane == 0) shared[wid] = val;
    __syncthreads();
    val = (threadIdx.x < (blockDim.x + 31) / 32) ? shared[lane] : 0.0f;
    if (wid == 0) val = warpReduceSum(val);
    return val;
}

// ============================================================
// Flash Attention Kernel
// 每个 Block 处理一个 (batch, query_head, q_tile)
// ============================================================
template <typename T>
__global__ void flashAttentionKernel(
    const T* __restrict__ Q,    // [B, Tq, Hq, D]
    const T* __restrict__ K,    // [B, Tk, Hkv, D]
    const T* __restrict__ V,    // [B, Tk, Hkv, D]
    T* __restrict__ O,          // [B, Tq, Hq, D]
    int tgt_seq_len,
    int src_seq_len,
    int query_heads,
    int kv_heads,
    int head_dim,
    bool is_causal)
{
    // Tile 大小
    constexpr int Br = 32;   // Q tile 行数
    constexpr int Bc = 32;   // K/V tile 列数

    // 解码 block 索引
    int q_tile_idx = blockIdx.x;   // 第几个 Q tile
    int head_idx   = blockIdx.y;   // 第几个 query head
    int batch_idx  = blockIdx.z;   // 第几个 batch

    // GQA: 多个 query heads 共享一个 kv head
    int heads_per_group = query_heads / kv_heads;
    int kv_head_idx = head_idx / heads_per_group;

    // Q tile 的行范围
    int q_start = q_tile_idx * Br;
    if (q_start >= tgt_seq_len) return;
    int q_len = min(Br, tgt_seq_len - q_start);

    // 缩放因子
    float scale = rsqrtf(static_cast<float>(head_dim));

    // ---- 基地址计算 ----
    // Q[batch_idx, q_start:q_start+Br, head_idx, :]
    const T* q_base = Q + ((size_t)batch_idx * tgt_seq_len * query_heads 
                         + (size_t)head_idx) * head_dim;
    // K[batch_idx, :, kv_head_idx, :]
    const T* k_base = K + ((size_t)batch_idx * src_seq_len * kv_heads 
                         + (size_t)kv_head_idx) * head_dim;
    // V[batch_idx, :, kv_head_idx, :]
    const T* v_base = V + ((size_t)batch_idx * src_seq_len * kv_heads 
                         + (size_t)kv_head_idx) * head_dim;
    // O[batch_idx, q_start:q_start+Br, head_idx, :]
    T* o_base = O + ((size_t)batch_idx * tgt_seq_len * query_heads 
                   + (size_t)head_idx) * head_dim;

    // ---- Shared Memory ----
    extern __shared__ float smem[];
    // 布局: [Br * head_dim] Q_tile + [Br * Bc] S_tile + [Br] m + [Br] l
    float* s_Q = smem;                          // Br × head_dim
    float* s_S = smem + Br * head_dim;         // Br × Bc (attention scores)
    float* s_m = s_S + Br * Bc;                // Br (running max)
    float* s_l = s_m + Br;                     // Br (running sum)

    // ---- 初始化 ----
    // 每个线程负责一个或多个 q 位置
    int tid = threadIdx.x;
    int num_threads = blockDim.x;

    // 初始化 m = -inf, l = 0, O_acc = 0 (FA2: 存储 un-normalized 累加器)
    for (int i = tid; i < q_len; i += num_threads) {
        s_m[i] = -FLT_MAX;
        s_l[i] = 0.0f;
    }
    __syncthreads();

    // 加载 Q tile 到 shared memory
    for (int idx = tid; idx < q_len * head_dim; idx += num_threads) {
        int row = idx / head_dim;
        int col = idx % head_dim;
        s_Q[row * head_dim + col] = static_cast<float>(
            q_base[(size_t)(q_start + row) * query_heads * head_dim + col]);
    }
    __syncthreads();

    // O 累加器初始化为 0（un-normalized 形式，最终除以 l_total）
    for (int idx = tid; idx < q_len * head_dim; idx += num_threads) {
        int row = idx / head_dim;
        int col = idx % head_dim;
        o_base[(size_t)(q_start + row) * query_heads * head_dim + col] = static_cast<T>(0.0f);
    }
    __syncthreads();

    // ---- 确定 K/V 遍历范围（Causal Mask 优化）----
    int kv_end = src_seq_len;
    if (is_causal) {
        // 最远的 q 位置决定了需要遍历的 K/V 范围
        int max_q_pos = q_start + q_len - 1;
        kv_end = min(src_seq_len, max_q_pos + 1);
    }
    int num_kv_tiles = (kv_end + Bc - 1) / Bc;

    // ---- 遍历 K/V tiles ----
    for (int kv_tile = 0; kv_tile < num_kv_tiles; kv_tile++) {
        int kv_start = kv_tile * Bc;
        int kv_len = min(Bc, kv_end - kv_start);

        // 计算 S = Q_tile × K_tile^T × scale
        // S[i][j] = sum_d(Q[i][d] * K[j][d]) * scale
        // 每个线程计算 S 矩阵的一个或多个元素
        for (int idx = tid; idx < q_len * kv_len; idx += num_threads) {
            int qi = idx / kv_len;   // q 位置 (tile 内)
            int kj = idx % kv_len;   // k 位置 (tile 内)

            int q_pos = q_start + qi;     // 全局 q 位置
            int k_pos = kv_start + kj;    // 全局 k 位置

            // Causal mask: q_pos < k_pos 时 mask 掉
            if (is_causal && k_pos > q_pos) {
                s_S[qi * Bc + kj] = -FLT_MAX;
                continue;
            }

            // 计算点积
            float dot = 0.0f;
            for (int d = 0; d < head_dim; d++) {
                float q_val = s_Q[qi * head_dim + d];
                float k_val = static_cast<float>(
                    k_base[(size_t)k_pos * kv_heads * head_dim + d]);
                dot += q_val * k_val;
            }
            s_S[qi * Bc + kj] = dot * scale;
        }
        __syncthreads();

        // ---- Online Softmax 更新 (FA2: un-normalized accumulator) ----
        for (int qi = tid; qi < q_len; qi += num_threads) {
            // 1. 找当前 tile 的 row max
            float row_max = -FLT_MAX;
            for (int kj = 0; kj < kv_len; kj++) {
                row_max = fmaxf(row_max, s_S[qi * Bc + kj]);
            }

            // 2. 更新全局 running max
            float m_old = s_m[qi];
            float m_new = fmaxf(m_old, row_max);
            s_m[qi] = m_new;

            // 3. 计算 exp(S - m_new) 并求和
            float row_sum = 0.0f;
            for (int kj = 0; kj < kv_len; kj++) {
                s_S[qi * Bc + kj] = expf(s_S[qi * Bc + kj] - m_new);
                row_sum += s_S[qi * Bc + kj];
            }

            // 4. 更新 running sum: l_new = l_old * exp(m_old - m_new) + row_sum
            float l_old = s_l[qi];
            float exp_mm = expf(m_old - m_new);  // 复用于 O 更新
            float l_new = l_old * exp_mm + row_sum;
            s_l[qi] = l_new;

            // 5. 更新 O (un-normalized): O = O * exp(m_old - m_new) + P × V
            //    避免每个 tile 都除以 l_new —— 最后一次性除以 l_total,
            //    精度显著更好 (特别是在 l_new 很大、o_val 很小的中间步骤)。
            for (int d = 0; d < head_dim; d++) {
                float o_old = static_cast<float>(
                    o_base[(size_t)(q_start + qi) * query_heads * head_dim + d]);
                float o_val = o_old * exp_mm;

                float pv_sum = 0.0f;
                for (int kj = 0; kj < kv_len; kj++) {
                    float p_val = s_S[qi * Bc + kj];  // exp(S - m_new)
                    float v_val = static_cast<float>(
                        v_base[(size_t)(kv_start + kj) * kv_heads * head_dim + d]);
                    pv_sum += p_val * v_val;
                }
                o_val += pv_sum;

                o_base[(size_t)(q_start + qi) * query_heads * head_dim + d]
                    = static_cast<T>(o_val);
            }
        }
        __syncthreads();
    }

    // ---- 最终归一化: O = O / l_total ----
    for (int idx = tid; idx < q_len * head_dim; idx += num_threads) {
        int row = idx / head_dim;
        int col = idx % head_dim;
        float o_val = static_cast<float>(
            o_base[(size_t)(q_start + row) * query_heads * head_dim + col]);
        float l_total = s_l[row];
        o_val /= l_total;
        o_base[(size_t)(q_start + row) * query_heads * head_dim + col]
            = static_cast<T>(o_val);
    }
}
/**
 * @brief Computes flash attention for given query, key, and value tensors.
 * 
 * @tparam T Data type (float) for input/output tensors
 * @param[in] h_q Query tensor of shape [batch_size, tgt_seq_len, query_heads, head_dim]
 * @param[in] h_k Key tensor of shape [batch_size, src_seq_len, kv_heads, head_dim]
 * @param[in] h_v Value tensor of shape [batch_size, src_seq_len, kv_heads, head_dim]
 * @param[out] h_o Output attention tensor of shape [batch_size, tgt_seq_len, query_heads, head_dim]
 * @param[in] batch_size Batch dimension size
 * @param[in] target_seq_len Target sequence length
 * @param[in] src_seq_len Source sequence length  
 * @param[in] query_heads Number of query attention heads
 * @param[in] kv_heads Number of key/value heads (supports grouped query attention)
 * @param[in] head_dim Dimension size of each attention head
 * @param[in] is_causal Whether to apply causal masking
 */
template <typename T>
void flashAttention(const std::vector<T>& h_q, const std::vector<T>& h_k,
                    const std::vector<T>& h_v, std::vector<T>& h_o,
                    int batch_size, int target_seq_len, int src_seq_len, 
                    int query_heads, int kv_heads, int head_dim, bool is_causal) {       
  // 预分配输出
    h_o.resize((size_t)batch_size * target_seq_len * query_heads * head_dim);

    // 分配 Device 内存
    T *d_q, *d_k, *d_v, *d_o;
    size_t q_bytes = (size_t)batch_size * target_seq_len * query_heads * head_dim * sizeof(T);
    size_t k_bytes = (size_t)batch_size * src_seq_len * kv_heads * head_dim * sizeof(T);
    size_t v_bytes = k_bytes;
    size_t o_bytes = q_bytes;

    RUNTIME_CHECK(cudaMalloc(&d_q, q_bytes));
    RUNTIME_CHECK(cudaMalloc(&d_k, k_bytes));
    RUNTIME_CHECK(cudaMalloc(&d_v, v_bytes));
    RUNTIME_CHECK(cudaMalloc(&d_o, o_bytes));

    // Host → Device
    RUNTIME_CHECK(cudaMemcpy(d_q, h_q.data(), q_bytes, cudaMemcpyHostToDevice));
    RUNTIME_CHECK(cudaMemcpy(d_k, h_k.data(), k_bytes, cudaMemcpyHostToDevice));
    RUNTIME_CHECK(cudaMemcpy(d_v, h_v.data(), v_bytes, cudaMemcpyHostToDevice));

    // Kernel 配置
    constexpr int Br = 32;
    int num_q_tiles = (target_seq_len + Br - 1) / Br;

    dim3 grid(num_q_tiles, query_heads, batch_size);
    dim3 block(256);

    // Shared memory: Q_tile + S_tile + m + l + reduce
    size_t smem_size = (Br * head_dim + Br * Br + Br + Br + 32) * sizeof(float);

    // 启动 Kernel
    flashAttentionKernel<T><<<grid, block, smem_size>>>(
        d_q, d_k, d_v, d_o,
        target_seq_len, src_seq_len,
        query_heads, kv_heads, head_dim, is_causal);

    RUNTIME_CHECK(cudaGetLastError());
    RUNTIME_CHECK(cudaDeviceSynchronize());

    // Device → Host
    RUNTIME_CHECK(cudaMemcpy(h_o.data(), d_o, o_bytes, cudaMemcpyDeviceToHost));

    // 释放
    RUNTIME_CHECK(cudaFree(d_q));
    RUNTIME_CHECK(cudaFree(d_k));
    RUNTIME_CHECK(cudaFree(d_v));
    RUNTIME_CHECK(cudaFree(d_o));
}

// *********************************************************************
// Explicit Template Instantiations (REQUIRED FOR LINKING WITH TESTER.O)
// DO NOT MODIFY THIS SECTION
// *********************************************************************
template void rmsNorm<float>(const std::vector<float>&, const std::vector<float>&,
  std::vector<float>&, size_t, size_t, float);
template void rmsNorm<half>(const std::vector<half>&, const std::vector<half>&,
  std::vector<half>&, size_t, size_t, float);
template void flashAttention<float>(const std::vector<float>&, const std::vector<float>&,
  const std::vector<float>&, std::vector<float>&,
  int, int, int, int, int, int, bool);
template void flashAttention<half>(const std::vector<half>&, const std::vector<half>&,
  const std::vector<half>&, std::vector<half>&,
  int, int, int, int, int, int, bool);

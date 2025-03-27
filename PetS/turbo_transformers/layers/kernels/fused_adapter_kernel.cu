#include "turbo_transformers/layers/kernels/fused_adapter_kernel.h"
#include "turbo_transformers/core/cuda_utils.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>

namespace turbo_transformers {
namespace layers {
namespace kernels {

// GELU激活函数
__device__ __forceinline__ float gelu(float x) {
  // GELU激活函数的近似实现
  // 0.5 * x * (1 + tanh(sqrt(2 / pi) * (x + 0.044715 * x^3)))
  const float sqrt_2_over_pi = 0.7978845608028654f;
  const float coeff = 0.044715f;
  float x3 = x * x * x;
  float inner = sqrt_2_over_pi * (x + coeff * x3);
  float tanh_inner = tanhf(inner);
  return 0.5f * x * (1.0f + tanh_inner);
}

// 融合的Adapter内核 - 处理单个线程块的一部分数据
template <int BLOCK_SIZE, int WARP_SIZE>
__global__ void FusedAdapterKernel(
    const float* input,           // 输入张量 [batch_size, seq_len, hidden_size]
    const float* down_scale_w,    // 下采样权重 [hidden_size, intermediate_size]
    const float* down_scale_b,    // 下采样偏置 [intermediate_size]
    const float* up_scale_w,      // 上采样权重 [intermediate_size, hidden_size]
    float* output,                // 输出张量 [batch_size, seq_len, hidden_size]
    int batch_size,               // 批大小
    int seq_len,                  // 序列长度
    int hidden_size,              // 隐藏层大小
    int intermediate_size) {      // 中间层大小
    
    // 计算当前线程处理的位置
    const int seq_idx = blockIdx.x;
    const int batch_idx = blockIdx.y;
    
    if (batch_idx >= batch_size || seq_idx >= seq_len) return;
    
    // 输入和输出的基址
    const int base_idx = batch_idx * seq_len * hidden_size + seq_idx * hidden_size;
    const float* input_ptr = input + base_idx;
    float* output_ptr = output + base_idx;
    
    // 使用共享内存存储中间结果，避免全局内存访问
    extern __shared__ float shared_mem[];
    float* intermediate = shared_mem; // [intermediate_size]
    
    // 步骤1: 下采样矩阵乘法 (input * down_scale_w -> intermediate)
    // 每个线程负责计算一部分中间结果
    for (int i = threadIdx.x; i < intermediate_size; i += blockDim.x) {
        float sum = 0.0f;
        for (int j = 0; j < hidden_size; j++) {
            sum += input_ptr[j] * down_scale_w[j * intermediate_size + i];
        }
        intermediate[i] = sum;
    }
    
    // 确保所有线程完成下采样计算
    __syncthreads();
    
    // 步骤2: 添加偏置和GELU激活
    for (int i = threadIdx.x; i < intermediate_size; i += blockDim.x) {
        intermediate[i] = gelu(intermediate[i] + down_scale_b[i]);
    }
    
    // 确保所有线程完成激活计算
    __syncthreads();
    
    // 步骤3: 上采样矩阵乘法 (intermediate * up_scale_w -> output)
    // 并添加残差连接 (output += input)
    for (int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        float sum = 0.0f;
        for (int j = 0; j < intermediate_size; j++) {
            sum += intermediate[j] * up_scale_w[j * hidden_size + i];
        }
        // 残差连接
        output_ptr[i] += sum;
    }
}

// 针对较大的隐藏层尺寸优化的版本
template <int BLOCK_SIZE, int WARP_SIZE>
__global__ void FusedAdapterKernelLarge(
    const float* input,           // 输入张量 [batch_size, seq_len, hidden_size]
    const float* down_scale_w,    // 下采样权重 [hidden_size, intermediate_size]
    const float* down_scale_b,    // 下采样偏置 [intermediate_size]
    const float* up_scale_w,      // 上采样权重 [intermediate_size, hidden_size]
    float* output,                // 输出张量 [batch_size, seq_len, hidden_size]
    int batch_size,               // 批大小
    int seq_len,                  // 序列长度
    int hidden_size,              // 隐藏层大小
    int intermediate_size) {      // 中间层大小
    
    // 计算当前线程处理的位置
    const int tid = threadIdx.x;
    const int wid = tid / WARP_SIZE;
    const int lane = tid % WARP_SIZE;
    const int seq_idx = blockIdx.x;
    const int batch_idx = blockIdx.y;
    
    if (batch_idx >= batch_size || seq_idx >= seq_len) return;
    
    // 输入和输出的基址
    const int base_idx = batch_idx * seq_len * hidden_size + seq_idx * hidden_size;
    const float* input_ptr = input + base_idx;
    float* output_ptr = output + base_idx;
    
    // 使用共享内存存储中间结果，避免全局内存访问
    extern __shared__ float shared_mem[];
    float* intermediate = shared_mem; // [intermediate_size]
    
    // 步骤1: 下采样矩阵乘法 (input * down_scale_w -> intermediate)
    // 使用分块计算和warp级别的归约来提高效率
    const int warps_per_block = BLOCK_SIZE / WARP_SIZE;
    const int items_per_warp = (intermediate_size + warps_per_block - 1) / warps_per_block;
    const int warp_start = wid * items_per_warp;
    const int warp_end = min(warp_start + items_per_warp, intermediate_size);
    
    for (int i = warp_start + lane; i < warp_end; i += WARP_SIZE) {
        float sum = 0.0f;
        for (int j = 0; j < hidden_size; j++) {
            sum += input_ptr[j] * down_scale_w[j * intermediate_size + i];
        }
        intermediate[i] = sum + down_scale_b[i];
    }
    
    // 确保所有线程完成下采样计算
    __syncthreads();
    
    // 步骤2: GELU激活
    for (int i = warp_start + lane; i < warp_end; i += WARP_SIZE) {
        intermediate[i] = gelu(intermediate[i]);
    }
    
    // 确保所有线程完成激活计算
    __syncthreads();
    
    // 步骤3: 上采样矩阵乘法 (intermediate * up_scale_w -> output)
    // 并添加残差连接 (output += input)
    const int items_per_warp_out = (hidden_size + warps_per_block - 1) / warps_per_block;
    const int warp_start_out = wid * items_per_warp_out;
    const int warp_end_out = min(warp_start_out + items_per_warp_out, hidden_size);
    
    for (int i = warp_start_out + lane; i < warp_end_out; i += WARP_SIZE) {
        float sum = 0.0f;
        for (int j = 0; j < intermediate_size; j++) {
            sum += intermediate[j] * up_scale_w[j * hidden_size + i];
        }
        // 残差连接
        output_ptr[i] += sum;
    }
}

// 启动融合的Adapter内核的包装函数
void LaunchFusedAdapterKernel(
    const float* input,
    const float* down_scale_w,
    const float* down_scale_b,
    const float* up_scale_w,
    float* output,
    int batch_size,
    int seq_len,
    int hidden_size,
    int intermediate_size,
    cudaStream_t stream) {
    
    // 确定内核配置
    const int block_size = 256;
    const int warp_size = 32;
    
    dim3 grid(seq_len, batch_size);
    dim3 block(block_size);
    
    // 计算共享内存大小
    size_t shared_mem_size = intermediate_size * sizeof(float);
    
    // 根据模型大小选择合适的内核
    if (hidden_size <= 1024 && intermediate_size <= 1024) {
        FusedAdapterKernel<block_size, warp_size><<<grid, block, shared_mem_size, stream>>>(
            input, down_scale_w, down_scale_b, up_scale_w, output,
            batch_size, seq_len, hidden_size, intermediate_size);
    } else {
        FusedAdapterKernelLarge<block_size, warp_size><<<grid, block, shared_mem_size, stream>>>(
            input, down_scale_w, down_scale_b, up_scale_w, output,
            batch_size, seq_len, hidden_size, intermediate_size);
    }
}

}  // namespace kernels
}  // namespace layers
}  // namespace turbo_transformers 
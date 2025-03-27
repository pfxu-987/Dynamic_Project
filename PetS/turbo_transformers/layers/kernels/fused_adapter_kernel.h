#pragma once

#include <cuda_runtime.h>

namespace turbo_transformers {
namespace layers {
namespace kernels {

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
    cudaStream_t stream);

}  // namespace kernels
}  // namespace layers
}  // namespace turbo_transformers 
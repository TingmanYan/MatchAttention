#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>

#include "match_former_fused_forward.hpp"

// Fused forward function that combines all operations
void mf_fused_forward_cuda(
    at::Tensor max_offset,
    at::Tensor q,
    at::Tensor k,
    at::Tensor v,
    at::Tensor output,
    at::Tensor attn_out,
    const int H,
    const int W,
    const std::vector<int64_t>& win_r,
    const int attn_num,
    const std::string& attn_type,
    const float scale)
{
    match_former_fused_forward(max_offset, q, k, v, output, attn_out, H, W, win_r, attn_num, attn_type, scale);
}
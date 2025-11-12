#include <torch/extension.h>
#include <vector>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <string>
#include <ATen/core/op_registration/op_registration.h>

// CUDA declarations

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
    const float scale);

void mf_fused_forward(
    at::Tensor max_offset,
    at::Tensor q,
    at::Tensor k,
    at::Tensor v,
    at::Tensor output,
    at::Tensor attn_out,
    const int64_t H,
    const int64_t W,
    const std::vector<int64_t>& win_r,
    const int64_t attn_num,
    const std::string& attn_type,
    const double scale)
{
    mf_fused_forward_cuda(max_offset, q, k, v, output, attn_out, H, W, win_r, attn_num, attn_type, static_cast<float>(scale));
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("fused_forward", &mf_fused_forward, "Fused forward pass (CUDA)");
}

TORCH_LIBRARY(match_attention, m)
{
    m.def("fused_forward(Tensor max_offset, Tensor q, Tensor k, Tensor v, Tensor(a!) output, Tensor(b!) attn_out, int H, int W, int[] win_r, int attn_num, str attn_type, float scale) -> ()", &mf_fused_forward);
}
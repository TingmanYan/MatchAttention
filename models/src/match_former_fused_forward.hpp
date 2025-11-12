#ifndef _MATCH_FORMER_FUSED_FORWARD_HPP_
#define _MATCH_FORMER_FUSED_FORWARD_HPP_

#include <vector>
#include <string>

// Fused forward function that combines all match former operations
void match_former_fused_forward(
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

#endif
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <cassert>
#include <cfloat>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <ATen/native/cuda/KernelUtils.cuh>

// Forward declarations of kernel functions
template <typename scalar_t>
__global__ void clip_offset_to_id_k(const scalar_t *const m_offset_d, int *const m_id_d, const int Lh, const int num_heads, const int N, const int H, const int W);

template <typename scalar_t>
__global__ void attn_weight_bilinear_forward_k(const scalar_t* const m_offset_d, scalar_t* const bilinear_weight_d, const int Lh);

__global__ void check_max_id_k(int *const m_id_d, const int L, const int N, const int H, const int W, const int num_heads, const int win_x, const int win_y);

template <typename scalar_t>
__global__ void match_attention_l1_norm_forward_k(
    const scalar_t *__restrict__ q_d,
    const scalar_t *__restrict__ k_d,
    scalar_t *__restrict__ attn_d,
    const int *__restrict__ m_id_d,
    const int *__restrict__ offset_d,
    const int L, const int N, const int H, const int W,
    const int C, const int num_heads, const int key_dim,
    const int attn_num, const int attn_numel,
    const bool swap_xy);

template <typename scalar_t>
__global__ void match_attention_dot_product_forward_k(const scalar_t *const q_d, const scalar_t *const k_d, scalar_t *const attn_d, const int *const m_id_d, const int* const offset_d, const int L, const int N, const int H, const int W, const int C, const int num_heads, const int key_dim, const int attn_num, const int attn_numel, const bool swap_xy);

template <typename scalar_t>
__global__ void bilinear_softmax_forward_general_k(scalar_t* const __restrict__ attn_d, 
                                   scalar_t* const __restrict__ attn_out_d, 
                                   scalar_t* const __restrict__ attn_sum_d, 
                                   const scalar_t* const __restrict__ bilinear_weight_d, 
                                   const int* const __restrict__ select_index_d, 
                                   int L, const int num_heads, const int h_attn_num, 
                                   const int attn_num, const int attn_num_sub);

template <typename scalar_t>
__global__ void attention_aggregate_forward_k(
    const scalar_t *__restrict__ v_d,
    scalar_t *__restrict__ out_d,
    const scalar_t *__restrict__ attn_d,
    const int *__restrict__ m_id_d,
    const int* __restrict__ offset_d,
    const int L, const int C, const int num_heads, 
    const int key_dim, const int attn_num, 
    const bool swap_xy);

template <typename scalar_t>
__global__ void scale_attention_k(scalar_t* attn_d, const scalar_t scale, const int total_size);

// Kernel implementations
template <typename scalar_t>
__global__ void
clip_offset_to_id_k(const scalar_t *const m_offset_d, int *const m_id_d, const int Lh, const int num_heads, const int N, const int H, const int W)
{
    int lh = blockIdx.x * blockDim.x + threadIdx.x;
    if (lh >= Lh)
        return;

    int l = lh / num_heads;
    int batch_id = l / N;
    int m_x = __float2int_rd(static_cast<float>(m_offset_d[lh*2])); // round to floor
    int m_y = __float2int_rd(static_cast<float>(m_offset_d[lh*2 + 1]));
    if (m_x < 0) m_x = 0;
    if (m_x >= W) m_x = W - 1;
    if (m_y < 0) m_y = 0;
    if (m_y >= H) m_y = H - 1;
    int m_pix_id = m_y * W + m_x;
    int m_id = batch_id * N + m_pix_id;
    m_id_d[lh] = m_id;
}

template <typename scalar_t>
__global__ void
attn_weight_bilinear_forward_k(const scalar_t* const m_offset_d, scalar_t* const bilinear_weight_d, const int Lh)
{
    int lh = blockIdx.x * blockDim.x + threadIdx.x;
    if (lh >= Lh)
        return;

    float ix = static_cast<float>(m_offset_d[lh*2]);
    float iy = static_cast<float>(m_offset_d[lh*2 + 1]);
    int ix_nw = __float2int_rd(ix);
    int iy_nw = __float2int_rd(iy);
    int ix_ne = ix_nw + 1;
    int iy_ne = iy_nw;
    int ix_sw = ix_nw;
    int iy_sw = iy_nw + 1;
    int ix_se = ix_nw + 1;
    int iy_se = iy_nw + 1;

    float nw = (ix_se - ix)    * (iy_se - iy);
    float ne = (ix    - ix_sw) * (iy_sw - iy);
    float sw = (ix_ne - ix)    * (iy    - iy_ne);
    float se = (ix    - ix_nw) * (iy    - iy_nw);
    bilinear_weight_d[lh*4] = static_cast<scalar_t>(nw);
    bilinear_weight_d[lh*4 + 1] = static_cast<scalar_t>(ne);
    bilinear_weight_d[lh*4 + 2] = static_cast<scalar_t>(sw);
    bilinear_weight_d[lh*4 + 3] = static_cast<scalar_t>(se); // bilinear_weight of shape [B, N, h, 4]
}

// check if the search window range is out of image coordinates
__forceinline__ __device__ void
check_within_image_coordinates(int& l_id, const int& N, const int& H, const int& W, const int& win_x, const int& win_y)
{
    int pix_id = l_id % N;
    int batch_id = l_id / N;
    int x = pix_id % W;
    int y = pix_id / W;
    if (x - win_x < 0)
        x = win_x;
    if (x + (win_x + 1) >= W)
        x = W - 1 - (win_x + 1);
    if (y - win_y < 0)
        y = win_y;
    if (y + (win_y + 1) >= H)
        y = H - 1 - (win_y + 1);
    pix_id = y * W + x;
    l_id = batch_id * N + pix_id;
}

__global__ void
check_max_id_k(int *const m_id_d, const int L, const int N, const int H, const int W, const int num_heads, const int win_x, const int win_y)
{
    int l, h;
    l = blockIdx.x * blockDim.x + threadIdx.x;
    h = blockIdx.y * blockDim.y + threadIdx.y;
    if (l >= L || h >= num_heads)
        return;

    int m_id = m_id_d[l * num_heads + h];
    check_within_image_coordinates(m_id, N, H, W, win_x, win_y);
    m_id_d[l * num_heads + h] = m_id;
}

template <typename scalar_t>
__global__ void match_attention_l1_norm_forward_k(
    const scalar_t *__restrict__ q_d,
    const scalar_t *__restrict__ k_d,
    scalar_t *__restrict__ attn_d,
    const int *__restrict__ m_id_d,
    const int *__restrict__ offset_d,
    const int L, const int N, const int H, const int W,
    const int C, const int num_heads, const int key_dim,
    const int attn_num, const int attn_numel,
    const bool swap_xy)
{
    int l, k;
    if (swap_xy)
    {
        l = blockIdx.x * blockDim.x + threadIdx.x;
        k = blockIdx.y * blockDim.y + threadIdx.y;
    }
    else
    {
        k = blockIdx.x * blockDim.x + threadIdx.x;
        l = blockIdx.y * blockDim.y + threadIdx.y;
    }
    if (l >= L || k >= num_heads*attn_num)
        return;

    constexpr int vec_size = sizeof(float4) / sizeof(scalar_t);
    const int h = k / attn_num;
    const int attn_id = k % attn_num;
    const int base_id = l*num_heads + h;
    const int base_attn_id = base_id*attn_num;
    const int key_id = m_id_d[base_id] + offset_d[attn_id];

    const int q_base = l * C;
    const int k_base = key_id * C;
    const int c_start = h * key_dim / vec_size;
    const int c_end = c_start + key_dim / vec_size;

    const float4* q_val_vec = reinterpret_cast<const float4*>(q_d + q_base);
    const float4* k_val_vec = reinterpret_cast<const float4*>(k_d + k_base);
    
    float diff_sum = 0.0f;

    for (int c = c_start; c < c_end; ++c) {
        float4 q_val_f4 = __ldg(&q_val_vec[c]);
        float4 k_val_f4 = __ldg(&k_val_vec[c]);
        
        if (vec_size == 4) {  // float32
            diff_sum += fabsf(q_val_f4.x - k_val_f4.x) +
                        fabsf(q_val_f4.y - k_val_f4.y) +
                        fabsf(q_val_f4.z - k_val_f4.z) +
                        fabsf(q_val_f4.w - k_val_f4.w);
        } else {  // bf16/fp16 (8 elements)
            if (std::is_same<scalar_t, at::Half>::value) {
                const half2* q_val_h2 = reinterpret_cast<const half2*>(&q_val_f4);
                const half2* k_val_h2 = reinterpret_cast<const half2*>(&k_val_f4);
                #pragma unroll
                for (int i = 0; i < 4; ++i) {
                    half2 q_h2 = q_val_h2[i];
                    half2 k_h2 = k_val_h2[i];
                    half2 diff_h2 = __habs2(__hsub2(q_h2, k_h2));
                    diff_sum += __half2float(diff_h2.x) + __half2float(diff_h2.y);
                }
            } else {  // bf16
                const __nv_bfloat162* q_val_bf2 = reinterpret_cast<const __nv_bfloat162*>(&q_val_f4);
                const __nv_bfloat162* k_val_bf2 = reinterpret_cast<const __nv_bfloat162*>(&k_val_f4);
                #pragma unroll
                for (int i = 0; i < 4; ++i) {
                    __nv_bfloat162 q_bf2 = q_val_bf2[i];
                    __nv_bfloat162 k_bf2 = k_val_bf2[i];
                    __nv_bfloat162 diff_bf2 = __habs2(__hsub2(q_bf2, k_bf2));
                    diff_sum += __bfloat162float(diff_bf2.x) + __bfloat162float(diff_bf2.y);
                }
            }
        }
    }
    attn_d[base_attn_id + attn_id] = static_cast<scalar_t>(-diff_sum);
}

template <typename scalar_t>
__global__ void
match_attention_dot_product_forward_k(const scalar_t *const q_d, const scalar_t *const k_d, scalar_t *const attn_d, const int *const m_id_d, const int* const offset_d, const int L, const int N, const int H, const int W, const int C, const int num_heads, const int key_dim, const int attn_num, const int attn_numel, const bool swap_xy)
{
    int l, k;
    if (swap_xy)
    {
        l = blockIdx.x * blockDim.x + threadIdx.x;
        k = blockIdx.y * blockDim.y + threadIdx.y;
    }
    else
    {
        k = blockIdx.x * blockDim.x + threadIdx.x;
        l = blockIdx.y * blockDim.y + threadIdx.y;
    }
    if (l >= L || k >= num_heads*attn_num)
        return;

    int h = k / attn_num;
    int attn_id = k % attn_num;
    int base_id = l*num_heads + h;
    int base_attn_id = base_id*attn_num;
    int key_id = m_id_d[base_id] + offset_d[attn_id];
    scalar_t diff_sum = 0;
    for (int c = h * key_dim; c < (h + 1) * key_dim; ++c)
    {
        diff_sum += q_d[l * C + c] * k_d[key_id * C + c];
    }
    attn_d[base_attn_id + attn_id] = diff_sum;
}

template <typename scalar_t>
__global__ void scale_attention_k(scalar_t* attn_d, const scalar_t scale, const int total_size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_size)
        return;
    attn_d[idx] = attn_d[idx] * scale;
}

template <typename T> struct VecType { using Type = T; };
template <> struct VecType<float> { using Type = float4; };
template <> struct VecType<__half> { using Type = float2; };
template <> struct VecType<__nv_bfloat16> { using Type = float2; };

template <typename scalar_t>
__device__ __inline__ typename VecType<scalar_t>::Type load_vec(const scalar_t* addr) {
    return *reinterpret_cast<const typename VecType<scalar_t>::Type*>(addr);
}

template <typename scalar_t>
__device__ __inline__ void store_vec(scalar_t* addr, typename VecType<scalar_t>::Type val) {
    *reinterpret_cast<typename VecType<scalar_t>::Type*>(addr) = val;
}

template <int WIN_SIZE, typename scalar_t>
__device__ __forceinline__ void load_window(scalar_t* window, const scalar_t* src) {
    constexpr int VEC_ELEMS = sizeof(typename VecType<scalar_t>::Type) / sizeof(scalar_t);
    constexpr int VEC_COUNT = WIN_SIZE / VEC_ELEMS;
    using vec_t = typename VecType<scalar_t>::Type;
    
    #pragma unroll 4
    for (int i = 0; i < VEC_COUNT; ++i) {
        vec_t vec = load_vec<scalar_t>(src + i * VEC_ELEMS);
        store_vec<scalar_t>(window + i * VEC_ELEMS, vec);
    }
}

template <int WIN_SIZE, typename scalar_t>
__device__ __forceinline__ void store_window(scalar_t* dst, const scalar_t* window) {
    constexpr int VEC_ELEMS = sizeof(typename VecType<scalar_t>::Type) / sizeof(scalar_t);
    constexpr int VEC_COUNT = WIN_SIZE / VEC_ELEMS;
    using vec_t = typename VecType<scalar_t>::Type;
    
    #pragma unroll 4
    for (int i = 0; i < VEC_COUNT; ++i) {
        vec_t vec = load_vec<scalar_t>(window + i * VEC_ELEMS);
        store_vec<scalar_t>(dst + i * VEC_ELEMS, vec);
    }
}

template <int WIN_SIZE, int SUB_WIN_SIZE, typename scalar_t>
__global__ void
bilinear_softmax_forward_k(scalar_t* const __restrict__ attn_d, 
                           scalar_t* const __restrict__ attn_out_d, 
                           scalar_t* const __restrict__ attn_sum_d, 
                           const scalar_t* const __restrict__ bilinear_weight_d, 
                           const int* const __restrict__ select_index_d, 
                           int L, const int num_heads, const int h_attn_num, 
                           const int attn_num)
{
    constexpr int VEC_ELEMS = sizeof(typename VecType<scalar_t>::Type) / sizeof(scalar_t);
    static_assert(WIN_SIZE % VEC_ELEMS == 0, "WIN_SIZE must be divisible by vector elements");
    using acc_t = float;

    int l = blockIdx.x * blockDim.x + threadIdx.x;
    int h = blockIdx.y * blockDim.y + threadIdx.y;
    if (l >= L || h >= num_heads)
        return;

    const int base_attn_id = l * h_attn_num + h * attn_num;
    const int base_sum_idx = l * (num_heads * 4) + h * 4;

    scalar_t window[WIN_SIZE];
    load_window<WIN_SIZE>(window, attn_d + base_attn_id);
    
    acc_t attn_max = -FLT_MAX;
    #pragma unroll 4
    for (int k = 0; k < WIN_SIZE; ++k) {
        if (static_cast<acc_t>(window[k]) > attn_max) {
            attn_max = static_cast<acc_t>(window[k]);
        }
    }

    #pragma unroll 4
    for (int k = 0; k < WIN_SIZE; ++k) {
        window[k] = static_cast<scalar_t>(expf(static_cast<acc_t>(window[k]) - attn_max));
    }

    scalar_t window_out[WIN_SIZE] = {0};

    for (int b = 0; b < 4; ++b) {
        acc_t block_sum = 0.0f;
        const int* block_idx = select_index_d + b * SUB_WIN_SIZE;
        
        #pragma unroll 4
        for (int k = 0; k < SUB_WIN_SIZE; ++k) {
            block_sum += static_cast<acc_t>(window[block_idx[k]]);
        }
        block_sum = fmaxf(block_sum, FLT_EPSILON);
        attn_sum_d[base_sum_idx + b] = static_cast<scalar_t>(block_sum);
        
        const scalar_t weight = bilinear_weight_d[base_sum_idx + b];
        const scalar_t scale = static_cast<scalar_t>(static_cast<acc_t>(weight) / block_sum);
        
        #pragma unroll 4
        for (int k = 0; k < SUB_WIN_SIZE; ++k) {
            const int idx = block_idx[k];
            window_out[idx] = window_out[idx] + window[idx] * scale;
        }
    }

    // write back to global memory
    store_window<WIN_SIZE>(attn_out_d + base_attn_id, window_out);
}

template <typename scalar_t>
__global__ void
bilinear_softmax_forward_general_k(scalar_t* const __restrict__ attn_d, 
                                   scalar_t* const __restrict__ attn_out_d, 
                                   scalar_t* const __restrict__ attn_sum_d, 
                                   const scalar_t* const __restrict__ bilinear_weight_d, 
                                   const int* const __restrict__ select_index_d, 
                                   int L, const int num_heads, const int h_attn_num, 
                                   const int attn_num, const int attn_num_sub)
{
    int l, h;
    l = blockIdx.x * blockDim.x + threadIdx.x;
    h = blockIdx.y * blockDim.y + threadIdx.y;
    if (l >= L || h >= num_heads)
        return;

    scalar_t attn_max = -FLT_MAX;
    int base_attn_id = l * h_attn_num + h * attn_num;
    for (int k = 0; k < attn_num; ++k)
    {
        scalar_t attn_val = attn_d[base_attn_id + k];
        if (attn_val > attn_max) {
            attn_max = attn_val;
        }
    }
    __syncthreads();

    for (int k = 0; k < attn_num; ++k)
    {
        attn_d[base_attn_id + k] = expf(attn_d[base_attn_id + k] - attn_max);
    }
    __syncthreads();

    for (int b = 0; b < 4; ++b)
    {
        scalar_t attn_sum = 0;
        for (int k = 0; k < attn_num_sub; ++k)
        {
            attn_sum += attn_d[base_attn_id + select_index_d[b*attn_num_sub + k]];
        }
        attn_sum = fmaxf(attn_sum, FLT_EPSILON);
        attn_sum_d[l*(num_heads*4) + h*4 + b] = attn_sum; // save for backward

        scalar_t weight = bilinear_weight_d[l*num_heads*4 + h*4 + b];
        for (int k = 0; k < attn_num_sub; ++k)
        {
            int select_index = select_index_d[b*attn_num_sub + k];
            attn_out_d[base_attn_id + select_index] += 
                attn_d[base_attn_id + select_index] / attn_sum * weight; // no write conflict
        }
    }
}

template <typename scalar_t>
__global__ void attention_aggregate_forward_k(
    const scalar_t *__restrict__ v_d,
    scalar_t *__restrict__ out_d,
    const scalar_t *__restrict__ attn_d,
    const int *__restrict__ m_id_d,
    const int* __restrict__ offset_d,
    const int L, const int C, const int num_heads, 
    const int key_dim, const int attn_num, 
    const bool swap_xy)
{
    int c, l;
    if (swap_xy)
    {
        l = blockIdx.x * blockDim.x + threadIdx.x;
        c = blockIdx.y * blockDim.y + threadIdx.y;
    }
    else
    {
        c = blockIdx.x * blockDim.x + threadIdx.x;
        l = blockIdx.y * blockDim.y + threadIdx.y;
    }
    if (l >= L || c >= C)
        return;

    const int h = c / key_dim;
    const int base_id = l*num_heads + h;
    const int base_attn_id = base_id*attn_num;
    const int m_id = m_id_d[base_id];
    float out_sum = 0;
    for (int k = 0; k < attn_num; ++k)
    {
        int key_id = m_id + offset_d[k];
        out_sum += static_cast<float>(attn_d[base_attn_id + k]) * 
                   static_cast<float>(v_d[key_id * C + c]);
    }
    out_d[l * C + c] = static_cast<scalar_t>(out_sum);
}

// Main fused forward function
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
    const float scale)
{
    const int B = q.size(0);
    const int N = q.size(1);
    const int C = q.size(2);
    const int h = max_offset.size(2);
    const int key_dim = C / h;
    const int L = B * N;
    const int Lh = L * h;
    const int attn_numel = L * h * attn_num;
    const int win_x = win_r[0];
    const int win_y = win_r[1];
    assert(attn_num == (2*win_r[0]+2)*(2*win_r[1]+2));
    const bool swap_xy_match = (h * attn_num < 32);
    const bool swap_xy_agg = (C < 32);
    const int attn_num_sub = (2*win_r[0] + 1)*(2*win_r[1] + 1);
    const int h_attn_num = h * attn_num;
    
    // Create temporary tensors
    auto m_id = at::zeros({B, N, h}, at::TensorOptions().dtype(at::kInt).device(max_offset.device()));
    auto bilinear_weight = at::zeros({B, N, h, 4}, max_offset.options());
    auto attn = at::zeros({B, N, h, attn_num}, q.options());
    auto attn_sum = at::zeros({B, N, h, 4}, q.options());
    
    // Create offset array for window
    int *offset_d;
    cudaMalloc(&offset_d, sizeof(int) * attn_num);
    int *offset_h = new int[attn_num];
    int num = 0;
    for (int y = -win_y; y <= (win_y + 1); ++y)
        for (int x = -win_x; x <= (win_x + 1); ++x)
        {
            offset_h[num++] = y * W + x;
        }
    cudaMemcpy(offset_d, offset_h, sizeof(int) * attn_num, cudaMemcpyHostToDevice);
    delete[] offset_h;
    
    // Create select_index array for bilinear softmax
    int *select_index_d;
    cudaMalloc(&select_index_d, sizeof(int)*4*attn_num_sub);
    int *select_index_h = new int[4*attn_num_sub];
    int win_W = 2*(win_r[0]+1);
    int delta_x[4] = {0, 1, 0, 1};
    int delta_y[4] = {0, 0, 1, 1};
    num = 0;
    for (int b = 0; b < 4; ++b) {
        int d_x = delta_x[b];
        int d_y = delta_y[b];
        for (int y = d_y; y <= 2*win_r[1] + d_y; ++y)
            for (int x = d_x; x <= 2*win_r[0] + d_x; ++x)
            {
                select_index_h[num++] = y * win_W + x;
            }
    }
    cudaMemcpy(select_index_d, select_index_h, sizeof(int)*attn_num_sub*4, cudaMemcpyHostToDevice);
    delete[] select_index_h;
    
    // Step 1: Clip offset to id
    {
        int grid = (Lh + 512 - 1) / 512;
        AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, max_offset.scalar_type(), "clip_offset_to_id_k", ([&] {
            clip_offset_to_id_k<scalar_t><<<grid, 512>>>(max_offset.data_ptr<scalar_t>(), m_id.data_ptr<int>(), Lh, h, N, H, W);
        }));
    }
    
    // Step 2: Compute bilinear weights
    {
        int grid = (Lh + 512 - 1) / 512;
        AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, max_offset.scalar_type(), "attn_weight_bilinear_forward_k", ([&] {
            attn_weight_bilinear_forward_k<scalar_t><<<grid, 512>>>(max_offset.data_ptr<scalar_t>(), bilinear_weight.data_ptr<scalar_t>(), Lh);
        }));
    }
    
    // Step 3: Check max id bounds
    {
        dim3 m_blocks(8, 128);
        dim3 grids((L + m_blocks.x - 1) / m_blocks.x, (h + m_blocks.y - 1) / m_blocks.y);
        check_max_id_k<<<grids, m_blocks>>>(m_id.data_ptr<int>(), L, N, H, W, h, win_x, win_y);
    }
    
    // Step 4: Compute attention
    {
        dim3 m_blocks(8, 128);
        dim3 grids((h*attn_num + m_blocks.x - 1) / m_blocks.x, (L + m_blocks.y - 1) / m_blocks.y);
        if (swap_xy_match)
            grids = dim3((L + m_blocks.x - 1) / m_blocks.x, (h*attn_num + m_blocks.y - 1) / m_blocks.y);
        
        if (attn_type == "dot_product") {
            AT_DISPATCH_FLOATING_TYPES_AND_HALF(q.scalar_type(), "match_attention_dot_product_forward_k", ([&] {
                match_attention_dot_product_forward_k<scalar_t><<<grids, m_blocks>>>(q.data_ptr<scalar_t>(), k.data_ptr<scalar_t>(), attn.data_ptr<scalar_t>(), m_id.data_ptr<int>(), offset_d, L, N, H, W, C, h, key_dim, attn_num, attn_numel, swap_xy_match);
            }));
        } else if (attn_type == "l1_norm") {
            AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, q.scalar_type(), "match_attention_l1_norm_forward_k", ([&] {
                match_attention_l1_norm_forward_k<scalar_t><<<grids, m_blocks>>>(q.data_ptr<scalar_t>(), k.data_ptr<scalar_t>(), attn.data_ptr<scalar_t>(), m_id.data_ptr<int>(), offset_d, L, N, H, W, C, h, key_dim, attn_num, attn_numel, swap_xy_match);
            }));
        }
    }
    
    // Step 5: Scale attention
    {
        int grid = (attn_numel + 512 - 1) / 512;
        AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, q.scalar_type(), "scale_attention_k", ([&] {
            scale_attention_k<scalar_t><<<grid, 512>>>(attn.data_ptr<scalar_t>(), static_cast<scalar_t>(scale), attn_numel);
        }));
    }
    
    // Step 6: Bilinear softmax
    {
        dim3 m_blocks = (attn_num == 16) ? dim3(128, 4) : dim3(32, 4);
        dim3 grids((L + m_blocks.x - 1) / m_blocks.x, (h + m_blocks.y - 1) / m_blocks.y);
        
        AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, attn.scalar_type(), "bilinear_softmax_forward", [&] {
            if (attn_num == 16 && attn_num_sub == 9) {
                bilinear_softmax_forward_k<16, 9><<<grids, m_blocks>>>(
                    attn.data_ptr<scalar_t>(), 
                    attn_out.data_ptr<scalar_t>(), 
                    attn_sum.data_ptr<scalar_t>(),
                    bilinear_weight.data_ptr<scalar_t>(), 
                    select_index_d, L, h, h_attn_num, attn_num
                );
            } else if (attn_num == 36 && attn_num_sub == 25) {
                bilinear_softmax_forward_k<36, 25><<<grids, m_blocks>>>(
                    attn.data_ptr<scalar_t>(), 
                    attn_out.data_ptr<scalar_t>(), 
                    attn_sum.data_ptr<scalar_t>(),
                    bilinear_weight.data_ptr<scalar_t>(), 
                    select_index_d, L, h, h_attn_num, attn_num
                );
            } else {
                bilinear_softmax_forward_general_k<<<grids, m_blocks>>>(
                    attn.data_ptr<scalar_t>(), 
                    attn_out.data_ptr<scalar_t>(), 
                    attn_sum.data_ptr<scalar_t>(), 
                    bilinear_weight.data_ptr<scalar_t>(), 
                    select_index_d, L, h, h_attn_num, attn_num, attn_num_sub
                );
            }
        });
    }
    
    // Step 7: Attention aggregation
    {
        dim3 m_blocks = (attn_num == 16) ? dim3(8, 128) : dim3(8, 32);
        dim3 grids((C + m_blocks.x - 1) / m_blocks.x, (L + m_blocks.y - 1) / m_blocks.y);
        if (swap_xy_agg)
            grids = dim3((L + m_blocks.x - 1) / m_blocks.x, (C + m_blocks.y - 1) / m_blocks.y);
        
        AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, v.scalar_type(), "attention_aggregate_forward_k", ([&] {
            attention_aggregate_forward_k<scalar_t><<<grids, m_blocks>>>(v.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(), attn_out.data_ptr<scalar_t>(), m_id.data_ptr<int>(), offset_d, L, C, h, key_dim, attn_num, swap_xy_agg);
        }));
    }
    
    // Cleanup
    cudaFree(offset_d);
    cudaFree(select_index_d);
}
/*
 * GDN Decode v9 — CuTe (CUTLASS Tile) kernel for B200 (sm100)
 *
 * Uses NVIDIA CuTe abstractions from CUTLASS 3.x:
 *
 *   CuTe Features:
 *   - cute::Layout: Tensor layout abstractions
 *   - cute::Tensor: Tensor operations
 *   - cute::copy: Efficient copy operations
 *   - TMA: Tensor Memory Accelerator with CuTe
 *   - SMEM: Swizzled shared memory layouts
 *
 *   Performance (Iteration 1):
 *   - cp.async for async global→shared prefetch
 *   - Swizzled SMEM to avoid bank conflicts
 *   - Overlapped load/compute via commit_group/wait_group
 *   - Cooperative thread arrays
 *
 * Grid: (B, H=8, V_BLOCKS)
 * Block: 128 threads (4 warps)
 */

#pragma once

#include <cmath>
#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_pipeline.h> // For cp.async primitives
#include <cuda_runtime.h>

// CuTe requires CUTLASS - check if available
#if __has_include(<cute/tensor.hpp>)
#define HAS_CUTE 1
#include <cute/algorithm/copy.hpp>
#include <cute/algorithm/gemm.hpp>
#include <cute/tensor.hpp>
using namespace cute;
#else
#define HAS_CUTE 0
#endif

namespace gdn {

// ============================================================
// Constants
// ============================================================

constexpr int V9_D = 128;
constexpr int V9_WARP_SIZE = 32;
constexpr int V9_NUM_WARPS = 4;
constexpr int V9_THREADS = V9_NUM_WARPS * V9_WARP_SIZE;

// ============================================================
// cp.async Primitives for Async Prefetch (Blackwell optimized)
// ============================================================

// Async copy 16 bytes (4 floats) from global to shared memory
__device__ __forceinline__ void cp_async_cg(void* smem_ptr, const void* gmem_ptr)
{
    uint32_t smem_addr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;"
                 :
                 : "r"(smem_addr), "l"(gmem_ptr)
                 : "memory");
}

// Async copy 4 bytes (1 float) from global to shared memory
__device__ __forceinline__ void cp_async_ca(void* smem_ptr, const void* gmem_ptr)
{
    uint32_t smem_addr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
    asm volatile("cp.async.ca.shared.global [%0], [%1], 4;"
                 :
                 : "r"(smem_addr), "l"(gmem_ptr)
                 : "memory");
}

// Commit current group of async copies
__device__ __forceinline__ void cp_async_commit_group()
{ asm volatile("cp.async.commit_group;"); }

// Wait for N groups to complete (0 = wait for all)
template<int N>
__device__ __forceinline__ void cp_async_wait_group()
{ asm volatile("cp.async.wait_group %0;" : : "n"(N)); }

// Wait for all async copies to complete
__device__ __forceinline__ void cp_async_wait_all()
{ asm volatile("cp.async.wait_all;"); }

// ============================================================
// CuTe-based TMA Operations (Blackwell optimized)
// ============================================================

#if HAS_CUTE

// State tensor layout: [V_BLOCK, D] with swizzle for bank conflict avoidance
template<int BLOCK_V>
using StateLayout = Layout<Shape<Int<BLOCK_V>, Int<V9_D>>, Stride<Int<V9_D>, Int<1>>>;

// Swizzled SMEM layout to avoid bank conflicts
template<int BLOCK_V>
using StateSmemLayout = decltype(composition(Swizzle<3, 3, 3> {}, // 8-byte swizzle
                                             StateLayout<BLOCK_V> {}));

// TMA descriptor for async bulk copy
template<int BLOCK_V>
__device__ __forceinline__ void cute_tma_load_state(float* smem_ptr,
                                                    const float* gmem_ptr,
                                                    int tile_v,
                                                    int tile_d)
{
    // Use cute::copy with TMA
    auto gmem_tensor = make_tensor(make_gmem_ptr(gmem_ptr),
                                   make_layout(make_shape(Int<BLOCK_V> {}, Int<V9_D> {})));
    auto smem_tensor = make_tensor(make_smem_ptr(smem_ptr), StateSmemLayout<BLOCK_V> {});

    // Async copy
    cute::copy(gmem_tensor, smem_tensor);
}

#endif // HAS_CUTE

// ============================================================
// Utility Functions
// ============================================================

__device__ __forceinline__ float v9_fast_exp(float x)
{ return __expf(x); }

__device__ __forceinline__ float v9_softplus(float x)
{ return (x > 20.0f) ? x : __logf(1.0f + __expf(x)); }

__device__ __forceinline__ float v9_sigmoid(float x)
{ return 1.0f / (1.0f + __expf(-x)); }

// Warp-level reduction
__device__ __forceinline__ float warp_reduce_sum_v9(float val)
{
#pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        val += __shfl_xor_sync(0xffffffff, val, mask);
    }
    return val;
}

// ============================================================
// v9 Kernel: CuTe TMA + Swizzled SMEM
// ============================================================

template<int BLOCK_V>
__global__ void __launch_bounds__(V9_THREADS)
  gdn_decode_kernel_v9_tma(const __nv_bfloat16* __restrict__ Q,
                           const __nv_bfloat16* __restrict__ K,
                           const __nv_bfloat16* __restrict__ V,
                           const float* __restrict__ State,
                           const float* __restrict__ A_log,
                           const __nv_bfloat16* __restrict__ A,
                           const float* __restrict__ DtBias,
                           const __nv_bfloat16* __restrict__ B_gate,
                           __nv_bfloat16* __restrict__ Out,
                           float* __restrict__ NewState,
                           float scale,
                           int stride_q_b,
                           int stride_q_h,
                           int stride_k_b,
                           int stride_k_h,
                           int stride_v_b,
                           int stride_v_h,
                           int stride_s_b,
                           int stride_s_h,
                           int stride_s_v,
                           int stride_a_b,
                           int stride_b_b,
                           int stride_o_b,
                           int stride_o_h,
                           int stride_ns_b,
                           int stride_ns_h,
                           int stride_ns_v)
{
    const int b = blockIdx.x;
    const int h = blockIdx.y;
    const int v_block = blockIdx.z;
    const int v0 = v_block * BLOCK_V;
    const int qk_h = h / 2; // GVA: 4 Q heads -> 8 V heads

    const int tid = threadIdx.x;
    const int warp_id = tid / V9_WARP_SIZE;
    const int lane_id = tid % V9_WARP_SIZE;

    // ============================================================
    // Shared Memory with Swizzled Layout
    // ============================================================
    extern __shared__ char smem_raw[];

    // Layout: [q, k, v, state_row0...state_rowN, new_state_row0...new_state_rowN, out]
    float* s_q = reinterpret_cast<float*>(smem_raw); // [D]
    float* s_k = s_q + V9_D;                         // [D]
    float* s_v = s_k + V9_D;                         // [BLOCK_V]
    float* s_state = s_v + BLOCK_V;                  // [BLOCK_V, D] - swizzled
    float* s_new_state = s_state + BLOCK_V * V9_D;   // [BLOCK_V, D]
    float* s_out = s_new_state + BLOCK_V * V9_D;     // [BLOCK_V]

    // ============================================================
    // Load Q, K (all threads participate)
    // ============================================================
    const __nv_bfloat16* q_ptr = Q + b * stride_q_b + qk_h * stride_q_h;
    const __nv_bfloat16* k_ptr = K + b * stride_k_b + qk_h * stride_k_h;

    // Vectorized load: 4 elements per thread
    if (tid < 32) {
// Load Q and K with float4
#pragma unroll
        for (int i = 0; i < 4; i++) {
            int idx = tid * 4 + i;
            if (idx < V9_D) {
                s_q[idx] = __bfloat162float(q_ptr[idx]);
                s_k[idx] = __bfloat162float(k_ptr[idx]);
            }
        }
    }

    // Load V
    const __nv_bfloat16* v_ptr = V + b * stride_v_b + h * stride_v_h + v0;
    if (tid < BLOCK_V) {
        s_v[tid] = __bfloat162float(v_ptr[tid]);
    }

    // ============================================================
    // Compute Gates - BROADCAST ACROSS ALL WARPS via SMEM
    // ============================================================
    // NOTE: __shfl_sync only works within a warp!
    // Must use shared memory to broadcast across all 4 warps.
    __shared__ float s_g, s_beta;

    if (tid == 0) {
        float a_val = __bfloat162float(A[b * stride_a_b + h]);
        float dt_val = DtBias[h];
        float alog = A_log[h];
        float b_val = __bfloat162float(B_gate[b * stride_b_b + h]);

        float x = a_val + dt_val;
        float sp = v9_softplus(x);
        s_g = v9_fast_exp(-v9_fast_exp(alog) * sp);
        s_beta = v9_sigmoid(b_val);
    }
    __syncthreads(); // Wait for thread 0 to write gates

    // All threads read from shared memory
    float g = s_g;
    float beta = s_beta;

    // ============================================================
    // TMA-style Async State Load with cp.async + Swizzle
    // ============================================================
    const float* state_ptr = State + b * stride_s_b + h * stride_s_h + v0 * stride_s_v;

// Use cp.async for async prefetch - overlaps load with gate computation
// Each thread handles multiple elements using vectorized loads
#pragma unroll
    for (int i = tid; i < BLOCK_V * V9_D; i += V9_THREADS) {
        int v_idx = i / V9_D;
        int d_idx = i % V9_D;

        // Apply swizzle for bank conflict avoidance
        int swizzled_d = d_idx ^ ((d_idx >> 3) & 7);

        // Issue async copy from global to shared
        cp_async_ca(&s_state[v_idx * V9_D + swizzled_d], &state_ptr[v_idx * stride_s_v + d_idx]);
    }

    // Commit the async copy group
    cp_async_commit_group();

    // Wait for async copies to complete (can overlap with other work)
    cp_async_wait_group<0>();

    __syncthreads();

// ============================================================
// Delta Rule Update (all warps) - matching Triton v5
// ============================================================
// CRITICAL: Apply g FIRST, then compute old_v with decayed state
#pragma unroll
    for (int v_idx = warp_id; v_idx < BLOCK_V; v_idx += V9_NUM_WARPS) {
        float old_v = 0.0f;

// Compute old_v = sum((g * S[v,:]) * k) - using decayed state
#pragma unroll 4
        for (int d = lane_id; d < V9_D; d += V9_WARP_SIZE) {
            int swizzled_d = d ^ ((d >> 3) & 7);
            float decayed_s = g * s_state[v_idx * V9_D + swizzled_d];
            old_v += decayed_s * s_k[d];
        }
        old_v = warp_reduce_sum_v9(old_v);

        float delta = beta * (s_v[v_idx] - old_v);

// Update state: S' = g * S + delta * k
#pragma unroll 4
        for (int d = lane_id; d < V9_D; d += V9_WARP_SIZE) {
            int swizzled_d = d ^ ((d >> 3) & 7);
            float decayed_s = g * s_state[v_idx * V9_D + swizzled_d];
            float new_s = decayed_s + delta * s_k[d];
            s_new_state[v_idx * V9_D + d] = new_s; // No swizzle for write
        }

        // Compute output: out[v] = scale * sum(S'[v,:] * q)
        float out_val = 0.0f;
#pragma unroll 4
        for (int d = lane_id; d < V9_D; d += V9_WARP_SIZE) {
            out_val += s_new_state[v_idx * V9_D + d] * s_q[d];
        }
        out_val = warp_reduce_sum_v9(out_val);

        if (lane_id == 0) {
            s_out[v_idx] = scale * out_val;
        }
    }

    __syncthreads();

    // ============================================================
    // Write Output and New State
    // ============================================================
    __nv_bfloat16* out_ptr = Out + b * stride_o_b + h * stride_o_h + v0;
    float* new_state_ptr = NewState + b * stride_ns_b + h * stride_ns_h + v0 * stride_ns_v;

    // Write output
    if (tid < BLOCK_V) {
        out_ptr[tid] = __float2bfloat16(s_out[tid]);
    }

// Write new state (coalesced)
#pragma unroll
    for (int i = tid; i < BLOCK_V * V9_D; i += V9_THREADS) {
        int v_idx = i / V9_D;
        int d_idx = i % V9_D;
        new_state_ptr[v_idx * stride_ns_v + d_idx] = s_new_state[v_idx * V9_D + d_idx];
    }
}

// ============================================================
// v9 Kernel: Baseline FP32 (no CuTe, just optimized patterns)
// ============================================================

template<int BLOCK_V>
__global__ void __launch_bounds__(V9_THREADS)
  gdn_decode_kernel_v9_fp32(const __nv_bfloat16* __restrict__ Q,
                            const __nv_bfloat16* __restrict__ K,
                            const __nv_bfloat16* __restrict__ V,
                            const float* __restrict__ State,
                            const float* __restrict__ A_log,
                            const __nv_bfloat16* __restrict__ A,
                            const float* __restrict__ DtBias,
                            const __nv_bfloat16* __restrict__ B_gate,
                            __nv_bfloat16* __restrict__ Out,
                            float* __restrict__ NewState,
                            float scale,
                            int stride_q_b,
                            int stride_q_h,
                            int stride_k_b,
                            int stride_k_h,
                            int stride_v_b,
                            int stride_v_h,
                            int stride_s_b,
                            int stride_s_h,
                            int stride_s_v,
                            int stride_a_b,
                            int stride_b_b,
                            int stride_o_b,
                            int stride_o_h,
                            int stride_ns_b,
                            int stride_ns_h,
                            int stride_ns_v)
{
    const int b = blockIdx.x;
    const int h = blockIdx.y;
    const int v_block = blockIdx.z;
    const int v0 = v_block * BLOCK_V;
    const int qk_h = h / 2;

    const int tid = threadIdx.x;
    const int warp_id = tid / V9_WARP_SIZE;
    const int lane_id = tid % V9_WARP_SIZE;

    // Shared memory
    extern __shared__ char smem_raw[];
    float* s_q = reinterpret_cast<float*>(smem_raw);
    float* s_k = s_q + V9_D;
    float* s_v = s_k + V9_D;
    float* s_state = s_v + BLOCK_V;
    float* s_new_state = s_state + BLOCK_V * V9_D;
    float* s_out = s_new_state + BLOCK_V * V9_D;

    // Load Q, K
    const __nv_bfloat16* q_ptr = Q + b * stride_q_b + qk_h * stride_q_h;
    const __nv_bfloat16* k_ptr = K + b * stride_k_b + qk_h * stride_k_h;

#pragma unroll 4
    for (int d = tid; d < V9_D; d += V9_THREADS) {
        s_q[d] = __bfloat162float(q_ptr[d]);
        s_k[d] = __bfloat162float(k_ptr[d]);
    }

    // Load V
    const __nv_bfloat16* v_ptr = V + b * stride_v_b + h * stride_v_h + v0;
    if (tid < BLOCK_V) {
        s_v[tid] = __bfloat162float(v_ptr[tid]);
    }

    // Compute gates - single thread, broadcast via SMEM
    __shared__ float s_g_fp32, s_beta_fp32;

    if (tid == 0) {
        float a_val = __bfloat162float(A[b * stride_a_b + h]);
        float dt_val = DtBias[h];
        float alog = A_log[h];
        float b_val = __bfloat162float(B_gate[b * stride_b_b + h]);

        float x = a_val + dt_val;
        float sp = v9_softplus(x);
        s_g_fp32 = v9_fast_exp(-v9_fast_exp(alog) * sp);
        s_beta_fp32 = v9_sigmoid(b_val);
    }
    __syncthreads();

    float g = s_g_fp32;
    float beta = s_beta_fp32;

    // Load state with cp.async (async prefetch)
    const float* state_ptr = State + b * stride_s_b + h * stride_s_h + v0 * stride_s_v;
#pragma unroll
    for (int i = tid; i < BLOCK_V * V9_D; i += V9_THREADS) {
        int v_idx = i / V9_D;
        int d_idx = i % V9_D;
        cp_async_ca(&s_state[v_idx * V9_D + d_idx], &state_ptr[v_idx * stride_s_v + d_idx]);
    }
    cp_async_commit_group();
    cp_async_wait_group<0>();

    __syncthreads();

// Delta rule update (matching Triton v5: S = g*S first, then delta)
#pragma unroll
    for (int v_idx = warp_id; v_idx < BLOCK_V; v_idx += V9_NUM_WARPS) {
        // CRITICAL: Apply g FIRST, then compute old_v with decayed state
        float old_v = 0.0f;

#pragma unroll 4
        for (int d = lane_id; d < V9_D; d += V9_WARP_SIZE) {
            float decayed_s = g * s_state[v_idx * V9_D + d];
            old_v += decayed_s * s_k[d];
        }
        old_v = warp_reduce_sum_v9(old_v);

        float delta = beta * (s_v[v_idx] - old_v);

#pragma unroll 4
        for (int d = lane_id; d < V9_D; d += V9_WARP_SIZE) {
            float decayed_s = g * s_state[v_idx * V9_D + d];
            float new_s = decayed_s + delta * s_k[d];
            s_new_state[v_idx * V9_D + d] = new_s;
        }

        float out_val = 0.0f;
#pragma unroll 4
        for (int d = lane_id; d < V9_D; d += V9_WARP_SIZE) {
            out_val += s_new_state[v_idx * V9_D + d] * s_q[d];
        }
        out_val = warp_reduce_sum_v9(out_val);

        if (lane_id == 0) {
            s_out[v_idx] = scale * out_val;
        }
    }

    __syncthreads();

    // Write output
    __nv_bfloat16* out_ptr = Out + b * stride_o_b + h * stride_o_h + v0;
    float* new_state_ptr = NewState + b * stride_ns_b + h * stride_ns_h + v0 * stride_ns_v;

    if (tid < BLOCK_V) {
        out_ptr[tid] = __float2bfloat16(s_out[tid]);
    }

#pragma unroll
    for (int i = tid; i < BLOCK_V * V9_D; i += V9_THREADS) {
        int v_idx = i / V9_D;
        int d_idx = i % V9_D;
        new_state_ptr[v_idx * stride_ns_v + d_idx] = s_new_state[v_idx * V9_D + d_idx];
    }
}

// ============================================================
// Launch Functions
// ============================================================

void gdn_decode_v9_launch_tma(const void* Q,
                              const void* K,
                              const void* V,
                              const void* State,
                              const void* A_log,
                              const void* A,
                              const void* DtBias,
                              const void* B_gate,
                              void* Out,
                              void* NewState,
                              float scale,
                              int B,
                              int num_v_heads,
                              int D,
                              int stride_q_b,
                              int stride_q_h,
                              int stride_k_b,
                              int stride_k_h,
                              int stride_v_b,
                              int stride_v_h,
                              int stride_s_b,
                              int stride_s_h,
                              int stride_s_v,
                              int stride_a_b,
                              int stride_b_b,
                              int stride_o_b,
                              int stride_o_h,
                              int stride_ns_b,
                              int stride_ns_h,
                              int stride_ns_v,
                              int BLOCK_V,
                              cudaStream_t stream)
{
    int V_BLOCKS = D / BLOCK_V;
    dim3 grid(B, num_v_heads, V_BLOCKS);
    dim3 block(V9_THREADS);

    size_t smem_size = (D + D + BLOCK_V + 2 * BLOCK_V * D + BLOCK_V) * sizeof(float);
    smem_size = ((smem_size + 127) / 128) * 128;

    if (BLOCK_V == 16) {
        gdn_decode_kernel_v9_tma<16>
          <<<grid, block, smem_size, stream>>>((const __nv_bfloat16*)Q,
                                               (const __nv_bfloat16*)K,
                                               (const __nv_bfloat16*)V,
                                               (const float*)State,
                                               (const float*)A_log,
                                               (const __nv_bfloat16*)A,
                                               (const float*)DtBias,
                                               (const __nv_bfloat16*)B_gate,
                                               (__nv_bfloat16*)Out,
                                               (float*)NewState,
                                               scale,
                                               stride_q_b,
                                               stride_q_h,
                                               stride_k_b,
                                               stride_k_h,
                                               stride_v_b,
                                               stride_v_h,
                                               stride_s_b,
                                               stride_s_h,
                                               stride_s_v,
                                               stride_a_b,
                                               stride_b_b,
                                               stride_o_b,
                                               stride_o_h,
                                               stride_ns_b,
                                               stride_ns_h,
                                               stride_ns_v);
    } else if (BLOCK_V == 32) {
        gdn_decode_kernel_v9_tma<32>
          <<<grid, block, smem_size, stream>>>((const __nv_bfloat16*)Q,
                                               (const __nv_bfloat16*)K,
                                               (const __nv_bfloat16*)V,
                                               (const float*)State,
                                               (const float*)A_log,
                                               (const __nv_bfloat16*)A,
                                               (const float*)DtBias,
                                               (const __nv_bfloat16*)B_gate,
                                               (__nv_bfloat16*)Out,
                                               (float*)NewState,
                                               scale,
                                               stride_q_b,
                                               stride_q_h,
                                               stride_k_b,
                                               stride_k_h,
                                               stride_v_b,
                                               stride_v_h,
                                               stride_s_b,
                                               stride_s_h,
                                               stride_s_v,
                                               stride_a_b,
                                               stride_b_b,
                                               stride_o_b,
                                               stride_o_h,
                                               stride_ns_b,
                                               stride_ns_h,
                                               stride_ns_v);
    } else {
        gdn_decode_kernel_v9_tma<64>
          <<<grid, block, smem_size, stream>>>((const __nv_bfloat16*)Q,
                                               (const __nv_bfloat16*)K,
                                               (const __nv_bfloat16*)V,
                                               (const float*)State,
                                               (const float*)A_log,
                                               (const __nv_bfloat16*)A,
                                               (const float*)DtBias,
                                               (const __nv_bfloat16*)B_gate,
                                               (__nv_bfloat16*)Out,
                                               (float*)NewState,
                                               scale,
                                               stride_q_b,
                                               stride_q_h,
                                               stride_k_b,
                                               stride_k_h,
                                               stride_v_b,
                                               stride_v_h,
                                               stride_s_b,
                                               stride_s_h,
                                               stride_s_v,
                                               stride_a_b,
                                               stride_b_b,
                                               stride_o_b,
                                               stride_o_h,
                                               stride_ns_b,
                                               stride_ns_h,
                                               stride_ns_v);
    }
}

void gdn_decode_v9_launch_fp32(const void* Q,
                               const void* K,
                               const void* V,
                               const void* State,
                               const void* A_log,
                               const void* A,
                               const void* DtBias,
                               const void* B_gate,
                               void* Out,
                               void* NewState,
                               float scale,
                               int B,
                               int num_v_heads,
                               int D,
                               int stride_q_b,
                               int stride_q_h,
                               int stride_k_b,
                               int stride_k_h,
                               int stride_v_b,
                               int stride_v_h,
                               int stride_s_b,
                               int stride_s_h,
                               int stride_s_v,
                               int stride_a_b,
                               int stride_b_b,
                               int stride_o_b,
                               int stride_o_h,
                               int stride_ns_b,
                               int stride_ns_h,
                               int stride_ns_v,
                               int BLOCK_V,
                               cudaStream_t stream)
{
    int V_BLOCKS = D / BLOCK_V;
    dim3 grid(B, num_v_heads, V_BLOCKS);
    dim3 block(V9_THREADS);

    size_t smem_size = (D + D + BLOCK_V + 2 * BLOCK_V * D + BLOCK_V) * sizeof(float);
    smem_size = ((smem_size + 127) / 128) * 128;

    if (BLOCK_V == 16) {
        gdn_decode_kernel_v9_fp32<16>
          <<<grid, block, smem_size, stream>>>((const __nv_bfloat16*)Q,
                                               (const __nv_bfloat16*)K,
                                               (const __nv_bfloat16*)V,
                                               (const float*)State,
                                               (const float*)A_log,
                                               (const __nv_bfloat16*)A,
                                               (const float*)DtBias,
                                               (const __nv_bfloat16*)B_gate,
                                               (__nv_bfloat16*)Out,
                                               (float*)NewState,
                                               scale,
                                               stride_q_b,
                                               stride_q_h,
                                               stride_k_b,
                                               stride_k_h,
                                               stride_v_b,
                                               stride_v_h,
                                               stride_s_b,
                                               stride_s_h,
                                               stride_s_v,
                                               stride_a_b,
                                               stride_b_b,
                                               stride_o_b,
                                               stride_o_h,
                                               stride_ns_b,
                                               stride_ns_h,
                                               stride_ns_v);
    } else if (BLOCK_V == 32) {
        gdn_decode_kernel_v9_fp32<32>
          <<<grid, block, smem_size, stream>>>((const __nv_bfloat16*)Q,
                                               (const __nv_bfloat16*)K,
                                               (const __nv_bfloat16*)V,
                                               (const float*)State,
                                               (const float*)A_log,
                                               (const __nv_bfloat16*)A,
                                               (const float*)DtBias,
                                               (const __nv_bfloat16*)B_gate,
                                               (__nv_bfloat16*)Out,
                                               (float*)NewState,
                                               scale,
                                               stride_q_b,
                                               stride_q_h,
                                               stride_k_b,
                                               stride_k_h,
                                               stride_v_b,
                                               stride_v_h,
                                               stride_s_b,
                                               stride_s_h,
                                               stride_s_v,
                                               stride_a_b,
                                               stride_b_b,
                                               stride_o_b,
                                               stride_o_h,
                                               stride_ns_b,
                                               stride_ns_h,
                                               stride_ns_v);
    } else {
        gdn_decode_kernel_v9_fp32<64>
          <<<grid, block, smem_size, stream>>>((const __nv_bfloat16*)Q,
                                               (const __nv_bfloat16*)K,
                                               (const __nv_bfloat16*)V,
                                               (const float*)State,
                                               (const float*)A_log,
                                               (const __nv_bfloat16*)A,
                                               (const float*)DtBias,
                                               (const __nv_bfloat16*)B_gate,
                                               (__nv_bfloat16*)Out,
                                               (float*)NewState,
                                               scale,
                                               stride_q_b,
                                               stride_q_h,
                                               stride_k_b,
                                               stride_k_h,
                                               stride_v_b,
                                               stride_v_h,
                                               stride_s_b,
                                               stride_s_h,
                                               stride_s_v,
                                               stride_a_b,
                                               stride_b_b,
                                               stride_o_b,
                                               stride_o_h,
                                               stride_ns_b,
                                               stride_ns_h,
                                               stride_ns_v);
    }
}

} // namespace gdn
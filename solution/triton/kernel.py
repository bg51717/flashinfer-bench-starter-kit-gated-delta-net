import math
import os
import torch
import torch.nn.functional as F
import triton
import triton.language as tl
import triton.profiler.language as pl
from flashinfer.gdn_decode import gated_delta_rule_decode_pretranspose

torch.set_float32_matmul_precision("high")

# Scratch allocator used by Triton kernels that rely on TMA buffers.
triton.set_allocator(
    lambda size, align, stream: torch.empty(
        size,
        device="cuda",
        dtype=torch.int8,
    )
)


@torch.no_grad()
def kernel_fla_tma(q, k, v, state, A_log, a, dt_bias, b, scale, output, new_state):
    """
    TMA + warp-specialized decode path.

    Launches one CTA per (batch, v-head) pair.
    Requires Hopper/Blackwell (compute capability >= 9).
    """
    K = q.shape[-1]
    B, T, H, _ = k.shape
    assert T == 1
    V = v.shape[-1]
    HV = v.shape[2]

    if scale is None or scale == 0.0:
        scale = 1.0 / math.sqrt(K)

    BK = 128
    BV = 64  # Tuned B200 config: BV=64, num_warps=4, num_stages=2

    grid = (B * HV,)  # One CTA per (batch, v-head)
    fused_recurrent_gated_delta_rule_tma_kernel[grid](
        q=q,
        k=k,
        v=v,
        A_log=A_log,
        a_gate=a,
        dt_bias=dt_bias,
        b_gate=b,
        o=output,
        h0=state,
        ht=new_state,
        scale=scale,
        B=B,
        H=H,
        HV=HV,
        K=K,
        V=V,
        BK=BK,
        BV=BV,
        num_warps=4,
        num_stages=2,
    )


@triton.jit
def fused_recurrent_gated_delta_rule_tma_kernel(
    q,
    k,
    v,
    A_log,
    a_gate,
    dt_bias,
    b_gate,
    o,
    h0,
    ht,
    scale,
    B: tl.constexpr,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
):
    # Grid: (B * HV,), with one CTA per (batch, v-head).
    # Iterates over V tiles using warp specialization:
    #   producer warps -> TMA state loads/stores
    #   consumer warps -> decay, delta-rule update, output compute
    i_nh = tl.program_id(0)
    i_n = i_nh // HV
    i_hv = i_nh % HV
    i_h = i_hv // (HV // H)  # q/k head index under grouped-value attention

    # Load q and k once; reused for all V tiles in this CTA.
    o_k = tl.arange(0, BK)
    b_q = tl.load(q + (i_n * H + i_h) * K + o_k).to(tl.float32) * scale
    b_k = tl.load(k + (i_n * H + i_h) * K + o_k).to(tl.float32)

    # Load scalar gates once per CTA.
    b_A = tl.load(A_log + i_hv).to(tl.float32)
    b_a = tl.load(a_gate + i_n * HV + i_hv).to(tl.float32)
    b_dt = tl.load(dt_bias + i_hv).to(tl.float32)
    b_b = tl.load(b_gate + i_n * HV + i_hv).to(tl.float32)

    x = b_a + b_dt
    sp = tl.where(x > 20.0, x, tl.log(1.0 + tl.exp(x)))  # Softplus.
    g = tl.exp(-tl.exp(b_A) * sp)  # Decay gate.
    b_beta = 1.0 / (1.0 + tl.exp(-b_b))  # Sigmoid update gate.

    # TMA descriptors for state [V, K], the dominant memory traffic per head.
    h0_desc = tl.make_tensor_descriptor(
        h0 + i_nh * V * K,
        shape=[V, K],
        strides=[K, 1],
        block_shape=[BV, BK],
    )
    ht_desc = tl.make_tensor_descriptor(
        ht + i_nh * V * K,
        shape=[V, K],
        strides=[K, 1],
        block_shape=[BV, BK],
    )

    NV: tl.constexpr = V // BV
    for i_v in tl.range(0, NV, 1, flatten=True, warp_specialize=True):
        # Load one state tile [BV, BK] via TMA.
        b_h = h0_desc.load([i_v * BV, 0]).to(tl.float32)

        # Load matching v tile [BV].
        o_v = i_v * BV + tl.arange(0, BV)
        b_v = tl.load(v + (i_n * HV + i_hv) * V + o_v).to(tl.float32)

        # Apply decay.
        b_h *= g

        # Delta-rule update: retrieve old value, blend with new, write state.
        b_v = b_beta * (b_v - tl.sum(b_h * b_k[None, :], 1))
        b_h += b_v[:, None] * b_k[None, :]

        # Compute output for this V tile.
        b_o = tl.sum(b_h * b_q[None, :], 1)

        # Store updated state tile via TMA.
        ht_desc.store([i_v * BV, 0], b_h.to(ht_desc.dtype))

        # Store output tile [BV].
        tl.store(o + (i_n * HV + i_hv) * V + o_v, b_o.to(tl.bfloat16))

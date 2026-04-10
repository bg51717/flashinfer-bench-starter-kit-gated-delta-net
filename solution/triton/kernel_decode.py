import math

import torch
import triton
import triton.language as tl


@triton.jit
def _gdn_decode_kernel_v2(
    q_ptr,
    k_ptr,
    v_ptr,
    state_ptr,
    a_log_ptr,
    a_ptr,
    dt_bias_ptr,
    b_gate_ptr,
    out_ptr,
    new_state_ptr,
    scale,
    stride_q_b,
    stride_q_h,  # Q  [B, 4, D]
    stride_k_b,
    stride_k_h,  # K  [B, 4, D]
    stride_v_b,
    stride_v_h,  # V  [B, 8, D]
    stride_s_b,
    stride_s_h,
    stride_s_v,  # State [B,8,D,D] k-last
    stride_a_b,  # a  [B, 8]
    stride_b_b,  # b  [B, 8]
    stride_o_b,
    stride_o_h,  # Out [B, 8, D]
    stride_ns_b,
    stride_ns_h,
    stride_ns_v,
    D: tl.constexpr,  # head size, fixed to 128 by definition
    BLOCK_V: tl.constexpr,  # adaptive V-tile size
):
    batch_idx = tl.program_id(0)
    v_head_idx = tl.program_id(1)
    v_block_idx = tl.program_id(2)  # in [0, D // BLOCK_V)
    v_offset = v_block_idx * BLOCK_V
    qk_head_idx = v_head_idx // 2  # GVA: 2 v-heads share one qk-head

    # Per-program scalar gates.
    a_val = tl.load(a_ptr + batch_idx * stride_a_b + v_head_idx).to(tl.float32)
    dt_val = tl.load(dt_bias_ptr + v_head_idx)
    a_log_val = tl.load(a_log_ptr + v_head_idx)
    b_val = tl.load(b_gate_ptr + batch_idx * stride_b_b + v_head_idx).to(tl.float32)

    gate_x = a_val + dt_val
    softplus_x = tl.where(gate_x > 20.0, gate_x, tl.log(1.0 + tl.exp(gate_x)))
    decay_gate = tl.exp(-tl.exp(a_log_val) * softplus_x)
    update_gate = tl.sigmoid(b_val)

    # Load q/k vectors and one V tile.
    feature_idx = tl.arange(0, D)
    tile_idx = tl.arange(0, BLOCK_V)

    q_vec = tl.load(
        q_ptr + batch_idx * stride_q_b + qk_head_idx * stride_q_h + feature_idx
    ).to(tl.float32)
    k_vec = tl.load(
        k_ptr + batch_idx * stride_k_b + qk_head_idx * stride_k_h + feature_idx
    ).to(tl.float32)
    v_vec = tl.load(
        v_ptr + batch_idx * stride_v_b + v_head_idx * stride_v_h + v_offset + tile_idx
    ).to(tl.float32)

    # Load state tile [BLOCK_V, D] for this program.
    tile_row = tl.arange(0, BLOCK_V)[:, None]
    tile_col = tl.arange(0, D)[None, :]
    state_tile_ptr = (
        state_ptr
        + batch_idx * stride_s_b
        + v_head_idx * stride_s_h
        + v_offset * stride_s_v
    )
    state_tile = tl.load(state_tile_ptr + tile_row * stride_s_v + tile_col)

    # GDN delta-rule update and decode output.
    state_tile = decay_gate * state_tile
    old_v = tl.sum(state_tile * k_vec[None, :], axis=1)
    delta = update_gate * (v_vec - old_v)
    state_tile = state_tile + delta[:, None] * k_vec[None, :]
    out_vec = scale * tl.sum(state_tile * q_vec[None, :], axis=1)

    tl.store(
        out_ptr
        + batch_idx * stride_o_b
        + v_head_idx * stride_o_h
        + v_offset
        + tile_idx,
        out_vec.to(tl.bfloat16),
    )
    new_state_tile_ptr = (
        new_state_ptr
        + batch_idx * stride_ns_b
        + v_head_idx * stride_ns_h
        + v_offset * stride_ns_v
    )
    tl.store(new_state_tile_ptr + tile_row * stride_ns_v + tile_col, state_tile)


def _select_block_v(batch_size: int) -> int:
    """Choose BLOCK_V from batch size for launch efficiency."""
    if batch_size <= 16:
        return 16
    if batch_size <= 128:
        return 32
    return 64


def decode_kernel(q, k, v, state, A_log, a, dt_bias, b, scale):
    """Run one-step GDN decode and return (output, new_state)."""
    batch_size, _, _, head_dim = q.shape
    num_v_heads = v.shape[2]
    device = q.device

    block_v = _select_block_v(batch_size)
    num_v_blocks = head_dim // block_v

    if scale is None or scale == 0.0:
        scale = 1.0 / math.sqrt(head_dim)

    q_c = q.squeeze(1).contiguous()  # [B, 4, D]
    k_c = k.squeeze(1).contiguous()
    v_c = v.squeeze(1).contiguous()  # [B, 8, D]
    a_c = a.squeeze(1).contiguous()  # [B, 8]
    b_c = b.squeeze(1).contiguous()

    if state is not None:
        state_tensor = state.contiguous()
    else:
        state_tensor = torch.zeros(
            batch_size,
            num_v_heads,
            head_dim,
            head_dim,
            dtype=torch.float32,
            device=device,
        )

    out = torch.empty(
        batch_size, num_v_heads, head_dim, dtype=torch.bfloat16, device=device
    )
    new_state = torch.empty_like(state_tensor)

    _gdn_decode_kernel_v2[(batch_size, num_v_heads, num_v_blocks)](
        q_c,
        k_c,
        v_c,
        state_tensor,
        A_log,
        a_c,
        dt_bias,
        b_c,
        out,
        new_state,
        float(scale),
        q_c.stride(0),
        q_c.stride(1),
        k_c.stride(0),
        k_c.stride(1),
        v_c.stride(0),
        v_c.stride(1),
        state_tensor.stride(0),
        state_tensor.stride(1),
        state_tensor.stride(2),
        a_c.stride(0),
        b_c.stride(0),
        out.stride(0),
        out.stride(1),
        new_state.stride(0),
        new_state.stride(1),
        new_state.stride(2),
        D=128,
        BLOCK_V=block_v,
        **{"num_warps": 4},
    )

    return out.unsqueeze(1), new_state  # [B,1,8,D], [B,8,D,D]


def run(q, k, v, state, A_log, a, dt_bias, b, scale):
    """Compatibility entry point expected by the benchmark harness."""
    return decode_kernel(q, k, v, state, A_log, a, dt_bias, b, scale)

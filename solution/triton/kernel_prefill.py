import math

import torch
import triton
import triton.language as tl


@triton.jit
def _prefill_step_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    state_ptr,
    a_log_ptr,
    a_ptr,
    dt_bias_ptr,
    b_ptr,
    cu_seq_ptr,
    out_ptr,
    new_state_ptr,
    scale,
    stride_q_t,
    stride_q_h,  # Q [T, 4, D]
    stride_k_t,
    stride_k_h,  # K [T, 4, D]
    stride_v_t,
    stride_v_h,  # V [T, 8, D]
    stride_s_n,
    stride_s_h,
    stride_s_v,  # State [N, 8, D, D] k-last
    stride_a_t,  # a [T, 8]
    stride_b_t,  # b [T, 8]
    stride_o_t,
    stride_o_h,  # Out [T, 8, D]
    stride_ns_n,
    stride_ns_h,
    stride_ns_v,
    D: tl.constexpr,  # head size = 128
    BLOCK_V: tl.constexpr,  # adaptive V-tile size
):
    """Software-pipelined per-sequence GDN prefill kernel."""
    seq_id = tl.program_id(0)
    v_head_id = tl.program_id(1)
    v_block_id = tl.program_id(2)
    v_base = v_block_id * BLOCK_V
    qk_head_id = v_head_id // 2

    seq_start = tl.load(cu_seq_ptr + seq_id).to(tl.int32)
    seq_end = tl.load(cu_seq_ptr + seq_id + 1).to(tl.int32)
    seq_len = seq_end - seq_start

    a_log = tl.load(a_log_ptr + v_head_id)
    dt_bias = tl.load(dt_bias_ptr + v_head_id)

    row_idx = tl.arange(0, BLOCK_V)[:, None]
    col_idx = tl.arange(0, D)[None, :]
    state_tile_ptr = (
        state_ptr + seq_id * stride_s_n + v_head_id * stride_s_h + v_base * stride_s_v
    )
    state_tile = tl.load(state_tile_ptr + row_idx * stride_s_v + col_idx)

    feature_idx = tl.arange(0, D)
    v_tile_idx = tl.arange(0, BLOCK_V)

    if seq_len <= 0:
        new_state_tile_ptr = (
            new_state_ptr
            + seq_id * stride_ns_n
            + v_head_id * stride_ns_h
            + v_base * stride_ns_v
        )
        tl.store(new_state_tile_ptr + row_idx * stride_ns_v + col_idx, state_tile)
        return

    token_idx = seq_start
    a_curr = tl.load(a_ptr + token_idx * stride_a_t + v_head_id).to(tl.float32)
    b_curr = tl.load(b_ptr + token_idx * stride_b_t + v_head_id).to(tl.float32)
    k_curr = tl.load(
        k_ptr + token_idx * stride_k_t + qk_head_id * stride_k_h + feature_idx
    ).to(tl.float32)
    v_curr = tl.load(
        v_ptr + token_idx * stride_v_t + v_head_id * stride_v_h + v_base + v_tile_idx
    ).to(tl.float32)
    q_curr = tl.load(
        q_ptr + token_idx * stride_q_t + qk_head_id * stride_q_h + feature_idx
    ).to(tl.float32)

    for token_offset in range(seq_len):
        token_idx = seq_start + token_offset
        next_token_idx = tl.minimum(token_idx + 1, seq_end - 1)

        a_next = tl.load(a_ptr + next_token_idx * stride_a_t + v_head_id).to(tl.float32)
        b_next = tl.load(b_ptr + next_token_idx * stride_b_t + v_head_id).to(tl.float32)
        k_next = tl.load(
            k_ptr + next_token_idx * stride_k_t + qk_head_id * stride_k_h + feature_idx
        ).to(tl.float32)
        v_next = tl.load(
            v_ptr
            + next_token_idx * stride_v_t
            + v_head_id * stride_v_h
            + v_base
            + v_tile_idx
        ).to(tl.float32)
        q_next = tl.load(
            q_ptr + next_token_idx * stride_q_t + qk_head_id * stride_q_h + feature_idx
        ).to(tl.float32)

        gate_input = a_curr + dt_bias
        softplus_val = tl.where(
            gate_input > 20.0, gate_input, tl.log(1.0 + tl.exp(gate_input))
        )
        decay_gate = tl.exp(-tl.exp(a_log) * softplus_val)
        update_gate = tl.sigmoid(b_curr)

        state_tile = decay_gate * state_tile
        old_v = tl.sum(state_tile * k_curr[None, :], axis=1)
        delta = update_gate * (v_curr - old_v)
        state_tile = state_tile + delta[:, None] * k_curr[None, :]

        output_tile = scale * tl.sum(state_tile * q_curr[None, :], axis=1)
        tl.store(
            out_ptr
            + token_idx * stride_o_t
            + v_head_id * stride_o_h
            + v_base
            + v_tile_idx,
            output_tile.to(tl.bfloat16),
        )

        a_curr = a_next
        b_curr = b_next
        k_curr = k_next
        v_curr = v_next
        q_curr = q_next

    new_state_tile_ptr = (
        new_state_ptr
        + seq_id * stride_ns_n
        + v_head_id * stride_ns_h
        + v_base * stride_ns_v
    )
    tl.store(new_state_tile_ptr + row_idx * stride_ns_v + col_idx, state_tile)


# Backward-compatible alias for the previous internal name.
_prefill_kernel_v5 = _prefill_step_kernel


def _select_block_v(num_sequences: int) -> int:
    """Choose BLOCK_V from sequence count."""
    if num_sequences <= 4:
        return 16
    return 32


def run_prefill(q, k, v, state, A_log, a, dt_bias, b, cu_seqlens, scale):
    """Run GDN prefill and return (output, new_state)."""
    num_tokens, _, head_dim = q.shape
    num_v_heads = v.shape[1]
    num_sequences = cu_seqlens.shape[0] - 1
    device = q.device

    block_v = _select_block_v(num_sequences)
    num_v_blocks = head_dim // block_v

    if scale is None or scale == 0.0:
        scale = 1.0 / math.sqrt(head_dim)

    q_contig = q.contiguous()
    k_contig = k.contiguous()
    v_contig = v.contiguous()
    a_contig = a.contiguous()
    b_contig = b.contiguous()
    cu_seq_contig = cu_seqlens.contiguous()

    if state is not None:
        state_tensor = state.contiguous()
    else:
        state_tensor = torch.zeros(
            num_sequences,
            num_v_heads,
            head_dim,
            head_dim,
            dtype=torch.float32,
            device=device,
        )

    output = torch.empty(
        num_tokens,
        num_v_heads,
        head_dim,
        dtype=torch.bfloat16,
        device=device,
    )
    new_state = torch.empty_like(state_tensor)

    _prefill_step_kernel[(num_sequences, num_v_heads, num_v_blocks)](
        q_contig,
        k_contig,
        v_contig,
        state_tensor,
        A_log,
        a_contig,
        dt_bias,
        b_contig,
        cu_seq_contig,
        output,
        new_state,
        float(scale),
        q_contig.stride(0),
        q_contig.stride(1),
        k_contig.stride(0),
        k_contig.stride(1),
        v_contig.stride(0),
        v_contig.stride(1),
        state_tensor.stride(0),
        state_tensor.stride(1),
        state_tensor.stride(2),
        a_contig.stride(0),
        b_contig.stride(0),
        output.stride(0),
        output.stride(1),
        new_state.stride(0),
        new_state.stride(1),
        new_state.stride(2),
        D=128,
        BLOCK_V=block_v,
        num_warps=4,
    )

    return output, new_state


def run(q, k, v, state, A_log, a, dt_bias, b, cu_seqlens, scale):
    """Compatibility entry point expected by the benchmark harness."""
    return run_prefill(q, k, v, state, A_log, a, dt_bias, b, cu_seqlens, scale)

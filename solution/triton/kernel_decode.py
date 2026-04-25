import math

import triton
import triton.language as tl


D = 128
H = 4
HV = 8


@triton.jit
def _softplus(x):
    return tl.where(x > 20.0, x, tl.log(1.0 + tl.exp(x)))


@triton.jit
def _gdn_decode_fwd_kernel(
    q,
    k,
    v,
    h0,
    A_log,
    a,
    dt_bias,
    beta,
    o,
    ht,
    scale,
    K: tl.constexpr,
    H: tl.constexpr,
    HV: tl.constexpr,
    BV: tl.constexpr,
):
    i_v, i_bh = tl.program_id(0), tl.program_id(1)
    i_b, i_hv = i_bh // HV, i_bh % HV
    i_h = i_hv // (HV // H)

    o_k = tl.arange(0, K)
    o_v = i_v * BV + tl.arange(0, BV)

    p_q = q + (i_b * H + i_h) * K + o_k
    p_k = k + (i_b * H + i_h) * K + o_k
    p_v = v + (i_b * HV + i_hv) * K + o_v
    p_h = h0 + (i_b * HV + i_hv) * K * K + o_v[:, None] * K + o_k[None, :]
    p_o = o + (i_b * HV + i_hv) * K + o_v
    p_ht = ht + (i_b * HV + i_hv) * K * K + o_v[:, None] * K + o_k[None, :]

    b_q = tl.load(p_q).to(tl.float32)
    b_k = tl.load(p_k).to(tl.float32)
    b_v = tl.load(p_v).to(tl.float32)
    b_h = tl.load(p_h).to(tl.float32)

    b_a = tl.load(a + i_b * HV + i_hv).to(tl.float32)
    b_A = tl.load(A_log + i_hv).to(tl.float32)
    b_dt = tl.load(dt_bias + i_hv).to(tl.float32)
    b_beta = tl.load(beta + i_b * HV + i_hv).to(tl.float32)

    b_g = tl.exp(-tl.exp(b_A) * _softplus(b_a + b_dt))
    b_beta = tl.sigmoid(b_beta)

    b_h = b_g * b_h
    b_delta = b_beta * (b_v - tl.sum(b_h * b_k[None, :], axis=1))
    b_h += b_delta[:, None] * b_k[None, :]
    b_o = scale * tl.sum(b_h * b_q[None, :], axis=1)

    tl.store(p_o, b_o.to(tl.bfloat16))
    tl.store(p_ht, b_h)


def _select_bv(batch_size: int) -> int:
    return 16


def gdn_decode_fwd(q, k, v, state, A_log, a, dt_bias, b, scale, output, new_state):
    batch_size = q.shape[0]
    block_v = _select_bv(batch_size)

    if scale == 0.0:
        scale = 1.0 / math.sqrt(D)

    grid = (triton.cdiv(D, block_v), batch_size * HV)
    _gdn_decode_fwd_kernel[grid](
        q,
        k,
        v,
        state,
        A_log,
        a,
        dt_bias,
        b,
        output,
        new_state,
        float(scale),
        K=D,
        H=H,
        HV=HV,
        BV=block_v,
        num_warps=4,
    )


def run(q, k, v, state, A_log, a, dt_bias, b, scale, output, new_state):
    gdn_decode_fwd(q, k, v, state, A_log, a, dt_bias, b, scale, output, new_state)

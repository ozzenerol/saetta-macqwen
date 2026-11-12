// Kernels for the attention and Gated DeltaNet layers, one token at a time.
// They follow src/cpu/ (attention.rs, deltanet.rs), which is checked against
// MLX; src/gpu_layers.rs tests each one against it. They are recorded once
// into a Plan, so what changes per token comes from TokenParams.

#include <metal_stdlib>
using namespace metal;

// Sum of `v` over the whole threadgroup. `tmp` holds one float per simdgroup.
inline float group_sum(float v, threadgroup float *tmp, uint simd, uint lane, uint simds) {
    v = simd_sum(v);
    if (lane == 0) {
        tmp[simd] = v;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float total = 0;
    for (uint i = 0; i < simds; i++) {
        total += tmp[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return total;
}

inline float group_max(float v, threadgroup float *tmp, uint simd, uint lane, uint simds) {
    v = simd_max(v);
    if (lane == 0) {
        tmp[simd] = v;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float m = -INFINITY;
    for (uint i = 0; i < simds; i++) {
        m = max(m, tmp[i]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return m;
}

inline float sigmoid(float x) {
    return 1.0f / (1.0f + exp(-x));
}

inline float silu(float x) {
    return x * sigmoid(x);
}

struct NormParams {
    uint dim;
    uint in_stride;   // floats between rows of x
    uint out_stride;  // floats between rows of out
    float eps;
    float scale;
    uint weighted;    // multiply by w
    uint gated;       // multiply by silu(gate), gate rows are `dim` apart
};

// One threadgroup per row:
// out = x / sqrt(mean(x^2) + eps) * scale [* w] [* silu(gate)].
// out may be x when the strides are equal.
kernel void rms_norm(device const float *x [[buffer(0)]],
                     device const float *w [[buffer(1)]],
                     device const float *gate [[buffer(2)]],
                     device float *out [[buffer(3)]],
                     constant NormParams &p [[buffer(4)]],
                     uint row [[threadgroup_position_in_grid]],
                     uint tid [[thread_position_in_threadgroup]],
                     uint threads [[threads_per_threadgroup]],
                     uint simd [[simdgroup_index_in_threadgroup]],
                     uint lane [[thread_index_in_simdgroup]],
                     uint simds [[simdgroups_per_threadgroup]]) {
    threadgroup float tmp[32];
    device const float *xr = x + row * p.in_stride;
    float sq = 0;
    for (uint i = tid; i < p.dim; i += threads) {
        sq += xr[i] * xr[i];
    }
    float total = group_sum(sq, tmp, simd, lane, simds);
    float inv = p.scale / sqrt(total / p.dim + p.eps);
    device float *o = out + row * p.out_stride;
    for (uint i = tid; i < p.dim; i += threads) {
        float v = xr[i] * inv;
        if (p.weighted) {
            v *= w[i];
        }
        if (p.gated) {
            v *= silu(gate[row * p.dim + i]);
        }
        o[i] = v;
    }
}

struct RopeParams {
    uint stride;     // floats between heads
    uint half_dims;  // rotary dims / 2
    float log2_theta;
};

// In place, one thread per (head, i < half_dims): value i pairs with
// i + half_dims, as `traditional=False` in MLX, at the token's position.
// The angle is computed in f32 as MLX does.
kernel void rope(device float *x [[buffer(0)]],
                 constant RopeParams &p [[buffer(1)]],
                 constant TokenParams &t [[buffer(2)]],
                 uint2 id [[thread_position_in_grid]]) {
    uint i = id.x;
    if (i >= p.half_dims) {
        return;
    }
    device float *h = x + id.y * p.stride;
    float freq = exp2(-float(i) / float(p.half_dims) * p.log2_theta);
    float angle = float(t.position) * freq;
    float s = precise::sin(angle);
    float c = precise::cos(angle);
    float a = h[i];
    float b = h[i + p.half_dims];
    h[i] = a * c - b * s;
    h[i + p.half_dims] = a * s + b * c;
}

// ---- Fused kernels for decode: each replaces several ops and their
// barriers. Same arithmetic as the separate kernels above.

struct ResidualParams {
    uint dim;
    float eps;
    uint logit_stride;  // mix_norm: floats between rows' shared logits
};

// The RMSNorm scale 1 / sqrt(mean(x^2) + eps) from each thread's part `sq`
// of the sum of squares.
inline float norm_scale(float sq, uint dim, float eps, threadgroup float *tmp, uint simd,
                        uint lane, uint simds) {
    float total = group_sum(sq, tmp, simd, lane, simds);
    return 1.0f / sqrt(total / dim + eps);
}

// x += y, then out = rms_norm(x) * w: the residual after the mixer and the
// post-attention norm in one op. One threadgroup per row of x, y and out.
kernel void add_norm(device float *x [[buffer(0)]],
                     device const float *y [[buffer(1)]],
                     device const float *w [[buffer(2)]],
                     device float *out [[buffer(3)]],
                     constant ResidualParams &p [[buffer(4)]],
                     uint row [[threadgroup_position_in_grid]],
                     uint tid [[thread_position_in_threadgroup]],
                     uint threads [[threads_per_threadgroup]],
                     uint simd [[simdgroup_index_in_threadgroup]],
                     uint lane [[thread_index_in_simdgroup]],
                     uint simds [[simdgroups_per_threadgroup]]) {
    threadgroup float tmp[32];
    x += row * p.dim;
    y += row * p.dim;
    out += row * p.dim;
    float sq = 0;
    for (uint i = tid; i < p.dim; i += threads) {
        float v = x[i] + y[i];
        x[i] = v;
        sq += v * v;
    }
    float inv = norm_scale(sq, p.dim, p.eps, tmp, simd, lane, simds);
    for (uint i = tid; i < p.dim; i += threads) {
        out[i] = x[i] * inv * w[i];
    }
}

// x += sigmoid(shared_logit) * shared + routed, then out = rms_norm(x) * w:
// the end of the MoE block and the next layer's input norm (or the final
// norm) in one op. One threadgroup per row.
kernel void mix_norm(device float *x [[buffer(0)]],
                     device const float *shared_logit [[buffer(1)]],
                     device const float *shared [[buffer(2)]],
                     device const float *routed [[buffer(3)]],
                     device const float *w [[buffer(4)]],
                     device float *out [[buffer(5)]],
                     constant ResidualParams &p [[buffer(6)]],
                     uint row [[threadgroup_position_in_grid]],
                     uint tid [[thread_position_in_threadgroup]],
                     uint threads [[threads_per_threadgroup]],
                     uint simd [[simdgroup_index_in_threadgroup]],
                     uint lane [[thread_index_in_simdgroup]],
                     uint simds [[simdgroups_per_threadgroup]]) {
    threadgroup float tmp[32];
    x += row * p.dim;
    shared += row * p.dim;
    routed += row * p.dim;
    out += row * p.dim;
    float gate = sigmoid(shared_logit[row * p.logit_stride]);
    float sq = 0;
    for (uint i = tid; i < p.dim; i += threads) {
        float v = x[i] + gate * shared[i] + routed[i];
        x[i] = v;
        sq += v * v;
    }
    float inv = norm_scale(sq, p.dim, p.eps, tmp, simd, lane, simds);
    for (uint i = tid; i < p.dim; i += threads) {
        out[i] = x[i] * inv * w[i];
    }
}

struct ConvParams {
    uint channels;
    uint kernel_size;
};

// Depthwise causal conv1d and SiLU, one thread per channel. `state` holds
// the last kernel_size - 1 inputs, oldest first, and gets x shifted in.
kernel void deltanet_conv(device const float *x [[buffer(0)]],
                          device const float *w [[buffer(1)]],
                          device float *state [[buffer(2)]],
                          device float *out [[buffer(3)]],
                          constant ConvParams &p [[buffer(4)]],
                          uint c [[thread_position_in_grid]]) {
    if (c >= p.channels) {
        return;
    }
    uint past = p.kernel_size - 1;
    device const float *wc = w + c * p.kernel_size;
    float acc = wc[past] * x[c];
    for (uint t = 0; t < past; t++) {
        acc += wc[t] * state[t * p.channels + c];
    }
    for (uint t = 0; t + 1 < past; t++) {
        state[t * p.channels + c] = state[(t + 1) * p.channels + c];
    }
    state[(past - 1) * p.channels + c] = x[c];
    out[c] = silu(acc);
}


// in_proj_qkv and the conv of one token in one op: qmv, then each output
// row goes through its channel's conv, so the qkv vector is never written.
template <uint BITS>
kernel void qmv_conv(device const uint *words [[buffer(0)]],
                     device const ushort *scales [[buffer(1)]],
                     device const ushort *biases [[buffer(2)]],
                     device const float *x [[buffer(3)]],
                     device const float *w [[buffer(4)]],
                     device float *state [[buffer(5)]],
                     device float *out [[buffer(6)]],
                     constant ConvParams &p [[buffer(7)]],
                     constant QmvShape &shape [[buffer(8)]],
                     uint tg [[threadgroup_position_in_grid]],
                     uint simd [[simdgroup_index_in_threadgroup]],
                     uint lane [[thread_index_in_simdgroup]]) {
    uint row0 = (tg * QMV_SIMDS + simd) * QMV_ROWS;
    uint row_words = shape.cols * BITS / 32;
    uint groups = shape.cols / 64;
    float acc[QMV_ROWS] = {0};

    for (uint k = lane * 32; k < shape.cols; k += 32 * 32) {
        float xv[32];
        float xsum = load_x32(x + k, xv);
#pragma clang loop unroll(full)
        for (uint r = 0; r < QMV_ROWS; r++) {
            uint row = row0 + r;
            if (row >= shape.rows) {
                break;
            }
            acc[r] += dot32<BITS>(words + row * row_words + k * BITS / 32, scales, biases,
                                  row * groups + k / 64, xv, xsum);
        }
    }

    for (uint r = 0; r < QMV_ROWS; r++) {
        float v = simd_sum(acc[r]);
        uint c = row0 + r;
        if (lane != 0 || c >= shape.rows) {
            continue;
        }
        uint past = p.kernel_size - 1;
        device const float *wc = w + c * p.kernel_size;
        float conv = wc[past] * v;
        for (uint t = 0; t < past; t++) {
            conv += wc[t] * state[t * p.channels + c];
        }
        for (uint t = 0; t + 1 < past; t++) {
            state[t * p.channels + c] = state[(t + 1) * p.channels + c];
        }
        state[(past - 1) * p.channels + c] = v;
        out[c] = silu(conv);
    }
}

template [[host_name("qmv_conv_5bit")]] kernel decltype(qmv_conv<5>) qmv_conv<5>;
template [[host_name("qmv_conv_8bit")]] kernel decltype(qmv_conv<8>) qmv_conv<8>;


struct StepParams {
    uint k_heads;
    uint v_heads;
    uint v_dim;
    float eps;
};

// The Gated DeltaNet step after the conv, for k_dim = 128: L2 norm of q and
// k (normalize_qk), the delta rule and the gated RMSNorm, in one op.
// One threadgroup of 1024 threads per value head; each simdgroup owns
// v_dim / 32 rows of the state and each lane 4 of the 128 values of a row,
// so the state is read and written in whole cache lines.
kernel void deltanet_step(device const float *q [[buffer(0)]],
                          device const float *k [[buffer(1)]],
                          device const float *v [[buffer(2)]],
                          device const float *a [[buffer(3)]],
                          device const float *b [[buffer(4)]],
                          device const float *a_log [[buffer(5)]],
                          device const float *dt_bias [[buffer(6)]],
                          device float *state [[buffer(7)]],
                          device const float *z [[buffer(8)]],
                          device const float *norm_w [[buffer(9)]],
                          device float *y [[buffer(10)]],
                          constant StepParams &p [[buffer(11)]],
                          uint h [[threadgroup_position_in_grid]],
                          uint tid [[thread_position_in_threadgroup]],
                          uint simd [[simdgroup_index_in_threadgroup]],
                          uint lane [[thread_index_in_simdgroup]],
                          uint simds [[simdgroups_per_threadgroup]]) {
    constexpr uint K_DIM = 128;
    threadgroup float tmp[32];
    threadgroup float out[256];
    uint kh = h / (p.v_heads / p.k_heads);

    float4 qv = ((device const float4 *)(q + kh * K_DIM))[lane];
    float4 kv = ((device const float4 *)(k + kh * K_DIM))[lane];
    // normalize_qk in MLX: rms_norm with eps / k_dim, q scaled by 1 / k_dim
    // and k by 1 / sqrt(k_dim).
    float eps = p.eps / K_DIM;
    qv *= (1.0f / K_DIM) / sqrt(simd_sum(dot(qv, qv)) / K_DIM + eps);
    kv *= rsqrt(float(K_DIM)) / sqrt(simd_sum(dot(kv, kv)) / K_DIM + eps);

    float x = a[h] + dt_bias[h];
    float softplus = max(x, 0.0f) + log(1.0f + exp(-abs(x)));
    float g = exp(-exp(a_log[h]) * softplus);
    float beta = sigmoid(b[h]);

    device const float *vh = v + h * p.v_dim;
    for (uint i = simd; i < p.v_dim; i += simds) {
        device float4 *row = (device float4 *)(state + (h * p.v_dim + i) * K_DIM) + lane;
        float4 s = *row * g;
        float delta = (vh[i] - simd_sum(dot(s, kv))) * beta;
        s += kv * delta;
        *row = s;
        float o = simd_sum(dot(s, qv));
        if (lane == 0) {
            out[i] = o;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float o = tid < p.v_dim ? out[tid] : 0.0f;
    float inv = norm_scale(o * o, p.v_dim, p.eps, tmp, simd, lane, simds);
    if (tid < p.v_dim) {
        uint n = h * p.v_dim + tid;
        y[n] = o * inv * norm_w[tid] * silu(z[n]);
    }
}


struct RowsStepParams {
    uint k_heads;
    uint v_heads;
    uint v_dim;
    uint kernel_size;  // conv1d taps
    float eps;
    uint rows;         // tokens, one after the other
    uint snapshots;    // 1: save the states after each row
};

// deltanet_conv and deltanet_step for `rows` tokens one after the other,
// for k_dim 128, v_dim 128 and two value heads per key head, in one op:
// - the depthwise causal conv1d and SiLU over this key head's q and k
//   channels and its value heads' v channels, shifting them into `conv`
//   ([kernel_size - 1, channels], oldest first);
// - the L2 norm of q and k (normalize_qk);
// - the delta rule on both value heads' [v_dim, 128] states;
// - the gated RMSNorm into y.
// One threadgroup of 1024 threads per key head. It is the only one that
// reads or writes its conv channels, so the state needs no ordering between
// threadgroups. Each simdgroup owns state rows and each lane 4 of the 128
// values of a row, so the state is read and written in whole cache lines.
//
// `rows` tokens go through in order; qkv, a, b, z and y hold one token per
// row. The state stays in registers between rows. With `snapshots`, the
// states after row r also go to conv_snaps[r] and state_snaps[r], so a
// speculative verify can roll back to any row.
kernel void deltanet_rows(device const float *qkv [[buffer(0)]],
                          device const float *conv_w [[buffer(1)]],
                          device float *conv [[buffer(2)]],
                          device const float *a [[buffer(3)]],
                          device const float *b [[buffer(4)]],
                          device const float *a_log [[buffer(5)]],
                          device const float *dt_bias [[buffer(6)]],
                          device float *state [[buffer(7)]],
                          device const float *z [[buffer(8)]],
                          device const float *norm_w [[buffer(9)]],
                          device float *y [[buffer(10)]],
                          constant RowsStepParams &p [[buffer(11)]],
                          device float *conv_snaps [[buffer(12)]],
                          device float *state_snaps [[buffer(13)]],
                          uint kh [[threadgroup_position_in_grid]],
                          uint tid [[thread_position_in_threadgroup]],
                          uint simd [[simdgroup_index_in_threadgroup]],
                          uint lane [[thread_index_in_simdgroup]],
                          uint simds [[simdgroups_per_threadgroup]]) {
    constexpr uint D = 128;     // k_dim and v_dim
    constexpr uint GROUP = 2;   // value heads per key head
    constexpr uint MINE = 8;    // state rows per simdgroup: GROUP * D / 32
    threadgroup float tmp[32];
    threadgroup float mixed[(2 + GROUP) * D];  // q, k, then the v heads
    threadgroup float out[GROUP * D];
    uint key_size = p.k_heads * D;
    uint channels = 2 * key_size + p.v_heads * D;
    uint values = p.v_heads * D;
    uint h0 = kh * GROUP;
    uint past = p.kernel_size - 1;
    uint conv_size = past * channels;
    uint state_size = values * D;

    // This thread's conv channel.
    uint part = tid / D;
    uint c = part == 0 ? kh * D + tid % D
           : part == 1 ? key_size + kh * D + tid % D
                       : 2 * key_size + (h0 + part - 2) * D + tid % D;
    bool convs = tid < (2 + GROUP) * D;

    // State row j of this simdgroup is r = simd + 32 j, all of head
    // h0 + r / D. Rows of a head share g and beta.
    float4 s[MINE];
#pragma clang loop unroll(full)
    for (uint j = 0; j < MINE; j++) {
        uint r = simd + simds * j;
        s[j] = ((device float4 *)(state + (h0 * D + r) * D))[lane];
    }

    for (uint row = 0; row < p.rows; row++) {
        if (convs) {
            device const float *wc = conv_w + c * p.kernel_size;
            float x = qkv[row * channels + c];
            float acc = wc[past] * x;
            for (uint t = 0; t < past; t++) {
                acc += wc[t] * conv[t * channels + c];
            }
            for (uint t = 0; t + 1 < past; t++) {
                conv[t * channels + c] = conv[(t + 1) * channels + c];
            }
            conv[(past - 1) * channels + c] = x;
            if (p.snapshots) {
                for (uint t = 0; t < past; t++) {
                    conv_snaps[row * conv_size + t * channels + c] = conv[t * channels + c];
                }
            }
            mixed[tid] = silu(acc);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float4 qv = ((threadgroup const float4 *)mixed)[lane];
        float4 kv = ((threadgroup const float4 *)(mixed + D))[lane];
        // normalize_qk in MLX: rms_norm with eps / k_dim, q scaled by 1 / k_dim
        // and k by 1 / sqrt(k_dim).
        float eps = p.eps / D;
        qv *= (1.0f / D) / sqrt(simd_sum(dot(qv, qv)) / D + eps);
        kv *= rsqrt(float(D)) / sqrt(simd_sum(dot(kv, kv)) / D + eps);

#pragma clang loop unroll(full)
        for (uint j = 0; j < MINE; j++) {
            uint r = simd + simds * j;
            uint h = h0 + r / D;
            uint i = r % D;
            float x = a[row * p.v_heads + h] + dt_bias[h];
            float softplus = max(x, 0.0f) + log(1.0f + exp(-abs(x)));
            float g = exp(-exp(a_log[h]) * softplus);
            float beta = sigmoid(b[row * p.v_heads + h]);

            float4 sj = s[j] * g;
            float delta = (mixed[(2 + r / D) * D + i] - simd_sum(dot(sj, kv))) * beta;
            sj += kv * delta;
            s[j] = sj;
            if (p.snapshots) {
                device float4 *snap =
                    (device float4 *)(state_snaps + row * state_size + (h * D + i) * D);
                snap[lane] = sj;
            }
            float o = simd_sum(dot(sj, qv));
            if (lane == 0) {
                out[r] = o;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

#pragma clang loop unroll(full)
        for (uint j = 0; j < GROUP; j++) {
            bool mine = tid / D == j;
            float o = mine ? out[tid] : 0.0f;
            float inv = norm_scale(o * o, D, p.eps, tmp, simd, lane, simds);
            if (mine) {
                uint n = row * values + (h0 + j) * D + tid % D;
                y[n] = o * inv * norm_w[tid % D] * silu(z[n]);
            }
        }
    }

#pragma clang loop unroll(full)
    for (uint j = 0; j < MINE; j++) {
        uint r = simd + simds * j;
        ((device float4 *)(state + (h0 * D + r) * D))[lane] = s[j];
    }
}

// deltanet_rows in two ops. deltanet_conv_rows: one thread per conv
// channel runs the m rows and writes mixed [m, channels] (silu applied).
kernel void deltanet_conv_rows(device const float *qkv [[buffer(0)]],
                               device const float *conv_w [[buffer(1)]],
                               device float *conv [[buffer(2)]],
                               device float *mixed [[buffer(3)]],
                               constant RowsStepParams &p [[buffer(4)]],
                               device float *conv_snaps [[buffer(5)]],
                               uint c [[thread_position_in_grid]]) {
    constexpr uint D = 128;
    uint channels = 2 * p.k_heads * D + p.v_heads * D;
    if (c >= channels) {
        return;
    }
    uint past = p.kernel_size - 1;
    uint conv_size = past * channels;
    device const float *wc = conv_w + c * p.kernel_size;
    for (uint row = 0; row < p.rows; row++) {
        float x = qkv[row * channels + c];
        float acc = wc[past] * x;
        for (uint t = 0; t < past; t++) {
            acc += wc[t] * conv[t * channels + c];
        }
        for (uint t = 0; t + 1 < past; t++) {
            conv[t * channels + c] = conv[(t + 1) * channels + c];
        }
        conv[(past - 1) * channels + c] = x;
        if (p.snapshots) {
            for (uint t = 0; t < past; t++) {
                conv_snaps[row * conv_size + t * channels + c] = conv[t * channels + c];
            }
        }
        mixed[row * channels + c] = silu(acc);
    }
}

// deltanet_step_rows: one threadgroup of SIMDS simdgroups per value head,
// D / SIMDS state rows per simdgroup in registers, the m rows one after the
// other. Same math and snapshots as deltanet_rows.
template <uint SIMDS>
kernel void deltanet_step_rows(device const float *mixed [[buffer(0)]],
                               device const float *a [[buffer(1)]],
                               device const float *b [[buffer(2)]],
                               device const float *a_log [[buffer(3)]],
                               device const float *dt_bias [[buffer(4)]],
                               device float *state [[buffer(5)]],
                               device const float *z [[buffer(6)]],
                               device const float *norm_w [[buffer(7)]],
                               device float *y [[buffer(8)]],
                               constant RowsStepParams &p [[buffer(9)]],
                               device float *state_snaps [[buffer(10)]],
                               uint h [[threadgroup_position_in_grid]],
                               uint tid [[thread_position_in_threadgroup]],
                               uint simd [[simdgroup_index_in_threadgroup]],
                               uint lane [[thread_index_in_simdgroup]]) {
    constexpr uint D = 128;
    constexpr uint MINE = D / SIMDS;
    threadgroup float tmp[32];
    threadgroup float out[D];
    uint key_size = p.k_heads * D;
    uint channels = 2 * key_size + p.v_heads * D;
    uint values = p.v_heads * D;
    uint state_size = values * D;
    uint kh = h / (p.v_heads / p.k_heads);
    float decay = exp(a_log[h]);

    float4 s[MINE];
#pragma clang loop unroll(full)
    for (uint j = 0; j < MINE; j++) {
        uint i = simd + SIMDS * j;
        s[j] = ((device float4 *)(state + (h * D + i) * D))[lane];
    }

    for (uint row = 0; row < p.rows; row++) {
        device const float *mr = mixed + row * channels;
        float4 qv = ((device const float4 *)(mr + kh * D))[lane];
        float4 kv = ((device const float4 *)(mr + key_size + kh * D))[lane];
        device const float *vv = mr + 2 * key_size + h * D;
        float eps = p.eps / D;
        qv *= (1.0f / D) / sqrt(simd_sum(dot(qv, qv)) / D + eps);
        kv *= rsqrt(float(D)) / sqrt(simd_sum(dot(kv, kv)) / D + eps);
        float x = a[row * p.v_heads + h] + dt_bias[h];
        float softplus = max(x, 0.0f) + log(1.0f + exp(-abs(x)));
        float g = exp(-decay * softplus);
        float beta = sigmoid(b[row * p.v_heads + h]);

#pragma clang loop unroll(full)
        for (uint j = 0; j < MINE; j++) {
            uint i = simd + SIMDS * j;
            float4 sj = s[j] * g;
            float delta = (vv[i] - simd_sum(dot(sj, kv))) * beta;
            sj += kv * delta;
            s[j] = sj;
            if (p.snapshots) {
                device float4 *snap =
                    (device float4 *)(state_snaps + row * state_size + (h * D + i) * D);
                snap[lane] = sj;
            }
            float o = simd_sum(dot(sj, qv));
            if (lane == 0) {
                out[i] = o;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float o = tid < D ? out[tid] : 0.0f;
        float inv = norm_scale(o * o, D, p.eps, tmp, simd, lane, SIMDS);
        if (tid < D) {
            uint n = row * values + h * D + tid;
            y[n] = o * inv * norm_w[tid] * silu(z[n]);
        }
    }

#pragma clang loop unroll(full)
    for (uint j = 0; j < MINE; j++) {
        uint i = simd + SIMDS * j;
        ((device float4 *)(state + (h * D + i) * D))[lane] = s[j];
    }
}

template [[host_name("deltanet_step_rows_8")]] kernel decltype(deltanet_step_rows<8>)
    deltanet_step_rows<8>;
template [[host_name("deltanet_step_rows_16")]] kernel decltype(deltanet_step_rows<16>)
    deltanet_step_rows<16>;
template [[host_name("deltanet_step_rows_32")]] kernel decltype(deltanet_step_rows<32>)
    deltanet_step_rows<32>;

struct PrepParams {
    uint heads;
    uint kv_heads;
    uint dim;
    uint half_dims;  // rotary dims / 2
    float log2_theta;
    float eps;
};

// Attention before the scores, in one op: q norm and RoPE into q, k norm
// and RoPE into the cache slot of the token, v copied into its slot.
// Threadgroups 0..heads are query heads, the next kv_heads are key/value
// heads; `dim` threads each (dim <= 1024). With SLOT false, keys and values
// are the one-token k and v instead, written in place (each thread writes
// only what it read), for kv_quantize to take into a quantized cache.
template <typename T, bool SLOT>
kernel void attention_prep(device const float *q_and_gate [[buffer(0)]],
                           device const float *k [[buffer(1)]],
                           device const float *v [[buffer(2)]],
                           device const float *q_norm [[buffer(3)]],
                           device const float *k_norm [[buffer(4)]],
                           device float *q_out [[buffer(5)]],
                           device T *keys [[buffer(6)]],
                           device T *values [[buffer(7)]],
                           constant PrepParams &p [[buffer(8)]],
                           constant TokenParams &t [[buffer(9)]],
                           uint g [[threadgroup_position_in_grid]],
                           uint i [[thread_position_in_threadgroup]],
                           uint simd [[simdgroup_index_in_threadgroup]],
                           uint lane [[thread_index_in_simdgroup]],
                           uint simds [[simdgroups_per_threadgroup]]) {
    threadgroup float tmp[32];
    threadgroup float head[1024];
    bool is_q = g < p.heads;
    uint j = is_q ? g : g - p.heads;
    uint slot = (SLOT ? t.position * p.kv_heads * p.dim : 0) + j * p.dim;

    float x = is_q ? q_and_gate[j * 2 * p.dim + i] : k[j * p.dim + i];
    float inv = norm_scale(x * x, p.dim, p.eps, tmp, simd, lane, simds);
    x *= inv * (is_q ? q_norm[i] : k_norm[i]);
    head[i] = x;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint r = i < p.half_dims ? i : i - p.half_dims;
    if (i < 2 * p.half_dims) {
        float freq = exp2(-float(r) / float(p.half_dims) * p.log2_theta);
        float angle = float(t.position) * freq;
        float s = precise::sin(angle);
        float c = precise::cos(angle);
        float a = head[r];
        float b = head[r + p.half_dims];
        x = i < p.half_dims ? a * c - b * s : a * s + b * c;
    }
    if (is_q) {
        q_out[j * p.dim + i] = x;
    } else {
        T value = T(v[j * p.dim + i]);
        keys[slot + i] = T(x);
        values[slot + i] = value;
    }
}

template [[host_name("attention_prep")]] kernel decltype(attention_prep<half, true>)
    attention_prep<half, true>;
template [[host_name("attention_prep_float")]] kernel decltype(attention_prep<float, false>)
    attention_prep<float, false>;

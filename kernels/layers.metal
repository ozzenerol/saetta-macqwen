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

struct AttentionParams {
    uint heads;
    uint kv_heads;
    uint dim;
    uint capacity;  // tokens the cache and the scores rows hold
    float scale;
};

// Copies the token's key and value (`kv_heads * dim` each) into its slot of
// the cache, one thread per value.
kernel void kv_store(device const float *k [[buffer(0)]],
                     device const float *v [[buffer(1)]],
                     device float *keys [[buffer(2)]],
                     device float *values [[buffer(3)]],
                     constant uint &size [[buffer(4)]],
                     constant TokenParams &t [[buffer(5)]],
                     uint i [[thread_position_in_grid]]) {
    if (i >= size) {
        return;
    }
    keys[t.position * size + i] = k[i];
    values[t.position * size + i] = v[i];
}

// One threadgroup per query head: softmax(q K^T * scale) V over the first
// kv_len tokens of the cache, times sigmoid(gate). The gate of head h sits
// at q_and_gate[h * 2 * dim + dim].
kernel void attention(device const float *q [[buffer(0)]],
                      device const float *keys [[buffer(1)]],
                      device const float *values [[buffer(2)]],
                      device const float *q_and_gate [[buffer(3)]],
                      device float *out [[buffer(4)]],
                      device float *scores [[buffer(5)]],
                      constant AttentionParams &p [[buffer(6)]],
                      constant TokenParams &tp [[buffer(7)]],
                      uint h [[threadgroup_position_in_grid]],
                      uint tid [[thread_position_in_threadgroup]],
                      uint threads [[threads_per_threadgroup]],
                      uint simd [[simdgroup_index_in_threadgroup]],
                      uint lane [[thread_index_in_simdgroup]],
                      uint simds [[simdgroups_per_threadgroup]]) {
    threadgroup float tmp[32];
    uint len = min(tp.kv_len, p.capacity);
    uint kv = h / (p.heads / p.kv_heads);
    uint stride = p.kv_heads * p.dim;
    device const float *qh = q + h * p.dim;
    device float *sh = scores + h * p.capacity;

    for (uint t = simd; t < len; t += simds) {
        device const float *k = keys + t * stride + kv * p.dim;
        float d = 0;
        for (uint i = lane; i < p.dim; i += 32) {
            d += qh[i] * k[i];
        }
        d = simd_sum(d);
        if (lane == 0) {
            sh[t] = d * p.scale;
        }
    }
    threadgroup_barrier(mem_flags::mem_device);

    float m = -INFINITY;
    for (uint t = tid; t < len; t += threads) {
        m = max(m, sh[t]);
    }
    m = group_max(m, tmp, simd, lane, simds);
    float sum = 0;
    for (uint t = tid; t < len; t += threads) {
        float e = exp(sh[t] - m);
        sh[t] = e;
        sum += e;
    }
    sum = group_sum(sum, tmp, simd, lane, simds);
    threadgroup_barrier(mem_flags::mem_device);

    for (uint i = tid; i < p.dim; i += threads) {
        float acc = 0;
        for (uint t = 0; t < len; t++) {
            acc += sh[t] * values[t * stride + kv * p.dim + i];
        }
        out[h * p.dim + i] = acc / sum * sigmoid(q_and_gate[h * 2 * p.dim + p.dim + i]);
    }
}

// ---- Fused kernels for decode: each replaces several ops and their
// barriers. Same arithmetic as the separate kernels above.

struct ResidualParams {
    uint dim;
    float eps;
};

// The RMSNorm scale 1 / sqrt(mean(x^2) + eps) from each thread's part `sq`
// of the sum of squares.
inline float norm_scale(float sq, uint dim, float eps, threadgroup float *tmp, uint simd,
                        uint lane, uint simds) {
    float total = group_sum(sq, tmp, simd, lane, simds);
    return 1.0f / sqrt(total / dim + eps);
}

// x += y, then out = rms_norm(x) * w: the residual after the mixer and the
// post-attention norm in one op. One threadgroup.
kernel void add_norm(device float *x [[buffer(0)]],
                     device const float *y [[buffer(1)]],
                     device const float *w [[buffer(2)]],
                     device float *out [[buffer(3)]],
                     constant ResidualParams &p [[buffer(4)]],
                     uint tid [[thread_position_in_threadgroup]],
                     uint threads [[threads_per_threadgroup]],
                     uint simd [[simdgroup_index_in_threadgroup]],
                     uint lane [[thread_index_in_simdgroup]],
                     uint simds [[simdgroups_per_threadgroup]]) {
    threadgroup float tmp[32];
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
// norm) in one op. One threadgroup.
kernel void mix_norm(device float *x [[buffer(0)]],
                     device const float *shared_logit [[buffer(1)]],
                     device const float *shared [[buffer(2)]],
                     device const float *routed [[buffer(3)]],
                     device const float *w [[buffer(4)]],
                     device float *out [[buffer(5)]],
                     constant ResidualParams &p [[buffer(6)]],
                     uint tid [[thread_position_in_threadgroup]],
                     uint threads [[threads_per_threadgroup]],
                     uint simd [[simdgroup_index_in_threadgroup]],
                     uint lane [[thread_index_in_simdgroup]],
                     uint simds [[simdgroups_per_threadgroup]]) {
    threadgroup float tmp[32];
    float gate = sigmoid(shared_logit[0]);
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

struct StepParams {
    uint k_heads;
    uint v_heads;
    uint v_dim;
    uint kernel_size;  // conv1d taps
    float eps;
};

// The whole Gated DeltaNet step after the projections, for k_dim 128,
// v_dim 128 and two value heads per key head, in one op:
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
kernel void deltanet_step(device const float *qkv [[buffer(0)]],
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
                          constant StepParams &p [[buffer(11)]],
                          uint kh [[threadgroup_position_in_grid]],
                          uint tid [[thread_position_in_threadgroup]],
                          uint simd [[simdgroup_index_in_threadgroup]],
                          uint lane [[thread_index_in_simdgroup]],
                          uint simds [[simdgroups_per_threadgroup]]) {
    constexpr uint D = 128;     // k_dim and v_dim
    constexpr uint GROUP = 2;   // value heads per key head
    threadgroup float tmp[32];
    threadgroup float mixed[(2 + GROUP) * D];  // q, k, then the v heads
    threadgroup float out[GROUP * D];
    uint key_size = p.k_heads * D;
    uint channels = 2 * key_size + p.v_heads * D;
    uint h0 = kh * GROUP;

    if (tid < (2 + GROUP) * D) {
        uint part = tid / D;
        uint i = tid % D;
        uint c = part == 0 ? kh * D + i
               : part == 1 ? key_size + kh * D + i
                           : 2 * key_size + (h0 + part - 2) * D + i;
        uint past = p.kernel_size - 1;
        device const float *wc = conv_w + c * p.kernel_size;
        float x = qkv[c];
        float acc = wc[past] * x;
        for (uint t = 0; t < past; t++) {
            acc += wc[t] * conv[t * channels + c];
        }
        for (uint t = 0; t + 1 < past; t++) {
            conv[t * channels + c] = conv[(t + 1) * channels + c];
        }
        conv[(past - 1) * channels + c] = x;
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

    for (uint r = simd; r < GROUP * D; r += simds) {
        uint h = h0 + r / D;
        uint i = r % D;
        float x = a[h] + dt_bias[h];
        float softplus = max(x, 0.0f) + log(1.0f + exp(-abs(x)));
        float g = exp(-exp(a_log[h]) * softplus);
        float beta = sigmoid(b[h]);

        device float4 *row = (device float4 *)(state + (h * D + i) * D) + lane;
        float4 s = *row * g;
        float delta = (mixed[(2 + r / D) * D + i] - simd_sum(dot(s, kv))) * beta;
        s += kv * delta;
        *row = s;
        float o = simd_sum(dot(s, qv));
        if (lane == 0) {
            out[r] = o;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint j = 0; j < GROUP; j++) {
        bool mine = tid / D == j;
        float o = mine ? out[tid] : 0.0f;
        float inv = norm_scale(o * o, D, p.eps, tmp, simd, lane, simds);
        if (mine) {
            uint n = (h0 + j) * D + tid % D;
            y[n] = o * inv * norm_w[tid % D] * silu(z[n]);
        }
    }
}

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
// heads; `dim` threads each (dim <= 1024).
kernel void attention_prep(device const float *q_and_gate [[buffer(0)]],
                           device const float *k [[buffer(1)]],
                           device const float *v [[buffer(2)]],
                           device const float *q_norm [[buffer(3)]],
                           device const float *k_norm [[buffer(4)]],
                           device float *q_out [[buffer(5)]],
                           device float *keys [[buffer(6)]],
                           device float *values [[buffer(7)]],
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
    uint slot = t.position * p.kv_heads * p.dim + j * p.dim;

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
        keys[slot + i] = x;
        values[slot + i] = v[j * p.dim + i];
    }
}

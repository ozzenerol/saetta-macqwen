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

struct DeltaParams {
    uint k_heads;
    uint v_heads;
    uint k_dim;
    uint v_dim;
};

// One thread per row i of the [v_dim, k_dim] state of value head h:
// decay by g, write v under key k with strength beta, read with q.
kernel void delta_rule(device const float *q [[buffer(0)]],
                       device const float *k [[buffer(1)]],
                       device const float *v [[buffer(2)]],
                       device const float *a [[buffer(3)]],
                       device const float *b [[buffer(4)]],
                       device const float *a_log [[buffer(5)]],
                       device const float *dt_bias [[buffer(6)]],
                       device float *state [[buffer(7)]],
                       device float *y [[buffer(8)]],
                       constant DeltaParams &p [[buffer(9)]],
                       uint n [[thread_position_in_grid]]) {
    if (n >= p.v_heads * p.v_dim) {
        return;
    }
    uint h = n / p.v_dim;
    uint i = n % p.v_dim;
    uint kh = h / (p.v_heads / p.k_heads);
    device const float *qh = q + kh * p.k_dim;
    device const float *kk = k + kh * p.k_dim;

    float x = a[h] + dt_bias[h];
    float softplus = max(x, 0.0f) + log(1.0f + exp(-abs(x)));
    float g = exp(-exp(a_log[h]) * softplus);
    float beta = sigmoid(b[h]);

    device float *s = state + n * p.k_dim;
    float kv = 0;
    for (uint j = 0; j < p.k_dim; j++) {
        s[j] *= g;
        kv += s[j] * kk[j];
    }
    float delta = (v[h * p.v_dim + i] - kv) * beta;
    float out = 0;
    for (uint j = 0; j < p.k_dim; j++) {
        s[j] += kk[j] * delta;
        out += s[j] * qh[j];
    }
    y[n] = out;
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

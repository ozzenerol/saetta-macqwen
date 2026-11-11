// Kernels for the attention and Gated DeltaNet layers, one token at a time.
// They follow src/cpu/ (attention.rs, deltanet.rs), which is checked against
// MLX; src/gpu_layers.rs tests each one against it.

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
    uint pos;
    float log2_theta;
};

// In place, one thread per (head, i < half_dims): value i pairs with
// i + half_dims, as `traditional=False` in MLX. The angle is computed in f32
// as MLX does.
kernel void rope(device float *x [[buffer(0)]],
                 constant RopeParams &p [[buffer(1)]],
                 uint2 id [[thread_position_in_grid]]) {
    uint i = id.x;
    if (i >= p.half_dims) {
        return;
    }
    device float *h = x + id.y * p.stride;
    float freq = exp2(-float(i) / float(p.half_dims) * p.log2_theta);
    float angle = float(p.pos) * freq;
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
    uint len;       // tokens in the cache, this one included
    uint capacity;  // scores row length
    float scale;
};

// One threadgroup per query head: softmax(q K^T * scale) V over the cache,
// times sigmoid(gate). The gate of head h sits at q_and_gate[h * 2 * dim + dim].
kernel void attention(device const float *q [[buffer(0)]],
                      device const float *keys [[buffer(1)]],
                      device const float *values [[buffer(2)]],
                      device const float *q_and_gate [[buffer(3)]],
                      device float *out [[buffer(4)]],
                      device float *scores [[buffer(5)]],
                      constant AttentionParams &p [[buffer(6)]],
                      uint h [[threadgroup_position_in_grid]],
                      uint tid [[thread_position_in_threadgroup]],
                      uint threads [[threads_per_threadgroup]],
                      uint simd [[simdgroup_index_in_threadgroup]],
                      uint lane [[thread_index_in_simdgroup]],
                      uint simds [[simdgroups_per_threadgroup]]) {
    threadgroup float tmp[32];
    uint kv = h / (p.heads / p.kv_heads);
    uint stride = p.kv_heads * p.dim;
    device const float *qh = q + h * p.dim;
    device float *sh = scores + h * p.capacity;

    for (uint t = simd; t < p.len; t += simds) {
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
    for (uint t = tid; t < p.len; t += threads) {
        m = max(m, sh[t]);
    }
    m = group_max(m, tmp, simd, lane, simds);
    float sum = 0;
    for (uint t = tid; t < p.len; t += threads) {
        float e = exp(sh[t] - m);
        sh[t] = e;
        sum += e;
    }
    sum = group_sum(sum, tmp, simd, lane, simds);
    threadgroup_barrier(mem_flags::mem_device);

    for (uint i = tid; i < p.dim; i += threads) {
        float acc = 0;
        for (uint t = 0; t < p.len; t++) {
            acc += sh[t] * values[t * stride + kv * p.dim + i];
        }
        out[h * p.dim + i] = acc / sum * sigmoid(q_and_gate[h * 2 * p.dim + p.dim + i]);
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

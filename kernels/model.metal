// The kernels around the layers that one token of the whole model needs:
// the embedding row, the router's top-k, and the MoE and residual sums.
// They follow src/cpu/model.rs and src/cpu/moe.rs; src/gpu_model.rs tests
// each one against them. Uses group_max and group_sum from layers.metal.

struct EmbedShape {
    uint cols;
};

// out = row `params.token` of a quantized embedding table.
template <uint BITS>
kernel void embed(device const uint *words [[buffer(0)]],
                  device const ushort *scales [[buffer(1)]],
                  device const ushort *biases [[buffer(2)]],
                  constant TokenParams &params [[buffer(3)]],
                  device float *out [[buffer(4)]],
                  constant EmbedShape &shape [[buffer(5)]],
                  uint i [[thread_position_in_grid]]) {
    if (i >= shape.cols) {
        return;
    }
    uint row = params.token;
    device const uint *w = words + row * (shape.cols * BITS / 32);
    uint bit = i * BITS;
    uint word = bit / 32;
    uint shift = bit % 32;
    uint q = w[word] >> shift;
    if (shift + BITS > 32) {
        q |= w[word + 1] << (32 - shift);
    }
    q &= (1u << BITS) - 1;
    uint group = row * (shape.cols / 64) + i / 64;
    out[i] = bf16_to_float(scales[group]) * float(q) + bf16_to_float(biases[group]);
}

template [[host_name("embed_5bit")]] kernel decltype(embed<5>) embed<5>;

struct RouteShape {
    uint experts;
    uint top_k;
};

// Softmax over the router logits, the top_k best (ties go to the lower id,
// as the CPU's stable sort), and their weights rescaled to sum to 1.
// One simdgroup: lane l holds experts l, l + 32, ... in registers, so the
// whole kernel runs without threadgroup memory or barriers.
#define ROUTE_PER_LANE 8

kernel void route_top_k(device const float *logits [[buffer(0)]],
                        device uint *ids [[buffer(1)]],
                        device float *weights [[buffer(2)]],
                        constant RouteShape &shape [[buffer(3)]],
                        uint lane [[thread_index_in_simdgroup]]) {
    float p[ROUTE_PER_LANE];
    float m = -INFINITY;
    for (uint j = 0; j < ROUTE_PER_LANE; j++) {
        uint e = lane + 32 * j;
        p[j] = e < shape.experts ? logits[e] : -INFINITY;
        m = max(m, p[j]);
    }
    m = simd_max(m);
    float z = 0;
    for (uint j = 0; j < ROUTE_PER_LANE; j++) {
        p[j] = exp(p[j] - m);
        z += p[j];
    }
    z = simd_sum(z);
    for (uint j = 0; j < ROUTE_PER_LANE; j++) {
        uint e = lane + 32 * j;
        p[j] = e < shape.experts ? p[j] / z : -1.0f;
    }

    float chosen[16];
    float sum = 0;
    for (uint k = 0; k < shape.top_k; k++) {
        // This lane's best; j goes up with the id, so the first wins ties.
        float bp = p[0];
        uint bj = 0;
        for (uint j = 1; j < ROUTE_PER_LANE; j++) {
            if (p[j] > bp) {
                bp = p[j];
                bj = j;
            }
        }
        float best = simd_max(bp);
        uint id = simd_min(bp == best ? lane + 32 * bj : 0xffffffffu);
        if (id == lane + 32 * bj) {
            p[bj] = -1.0f;
        }
        chosen[k] = best;
        sum += best;
        if (lane == 0) {
            ids[k] = id;
        }
    }
    if (lane < shape.top_k) {
        weights[lane] = chosen[lane] / sum;
    }
}

// Debug (SAETTA_FORCE_EXPERT): at position f[0], slot f[1] takes expert f[2].
kernel void force_expert(device uint *ids [[buffer(0)]],
                         constant uint *f [[buffer(1)]],
                         constant TokenParams &tp [[buffer(2)]]) {
    if (tp.position == f[0]) {
        ids[f[1]] = f[2];
    }
}

// out = silu(gate) * up, the middle of a SwiGLU MLP.
kernel void silu_mul(device const float *gate [[buffer(0)]],
                     device const float *up [[buffer(1)]],
                     device float *out [[buffer(2)]],
                     constant uint &n [[buffer(3)]],
                     uint i [[thread_position_in_grid]]) {
    if (i < n) {
        float g = gate[i];
        out[i] = g * sigmoid(g) * up[i];
    }
}

// x += y, the residual after the attention or DeltaNet mixer.
kernel void add_into(device float *x [[buffer(0)]],
                     device const float *y [[buffer(1)]],
                     constant uint &n [[buffer(2)]],
                     uint i [[thread_position_in_grid]]) {
    if (i < n) {
        x[i] += y[i];
    }
}

// x += sigmoid(shared_logit) * shared + routed, the end of the MoE block.
kernel void moe_mix(device float *x [[buffer(0)]],
                    device const float *shared_logit [[buffer(1)]],
                    device const float *shared [[buffer(2)]],
                    device const float *routed [[buffer(3)]],
                    constant uint &n [[buffer(4)]],
                    uint i [[thread_position_in_grid]]) {
    if (i < n) {
        x[i] += sigmoid(shared_logit[0]) * shared[i] + routed[i];
    }
}

// out[0] = the index of the largest of x[0..n], the lowest index on ties,
// as cpu::model::argmax. One threadgroup of up to 1024 threads.
kernel void argmax(device const float *x [[buffer(0)]],
                   device uint *out [[buffer(1)]],
                   constant uint &n [[buffer(2)]],
                   uint t [[thread_position_in_threadgroup]],
                   uint threads [[threads_per_threadgroup]],
                   uint simd [[simdgroup_index_in_threadgroup]],
                   uint lane [[thread_index_in_simdgroup]],
                   uint simds [[simdgroups_per_threadgroup]]) {
    threadgroup float best_v[32];
    threadgroup uint best_i[32];
    // Each thread walks its indices in order, so it keeps the first maximum.
    float v = -INFINITY;
    uint index = 0xffffffffu;
    for (uint i = t; i < n; i += threads) {
        float y = x[i];
        if (y > v) {
            v = y;
            index = i;
        }
    }
    float sv = simd_max(v);
    uint si = simd_min(v == sv ? index : 0xffffffffu);
    if (lane == 0) {
        best_v[simd] = sv;
        best_i[simd] = si;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (t == 0) {
        float bv = best_v[0];
        uint bi = best_i[0];
        for (uint i = 1; i < simds; i++) {
            if (best_v[i] > bv || (best_v[i] == bv && best_i[i] < bi)) {
                bv = best_v[i];
                bi = best_i[i];
            }
        }
        out[0] = bi;
    }
}

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
// One threadgroup with one thread per expert, at most 1024.
kernel void route_top_k(device const float *logits [[buffer(0)]],
                        device uint *ids [[buffer(1)]],
                        device float *weights [[buffer(2)]],
                        constant RouteShape &shape [[buffer(3)]],
                        uint t [[thread_position_in_threadgroup]],
                        uint simd [[simdgroup_index_in_threadgroup]],
                        uint lane [[thread_index_in_simdgroup]],
                        uint simds [[simdgroups_per_threadgroup]]) {
    threadgroup float tmp[32];
    threadgroup float best_p[32];
    threadgroup uint best_id[32];
    threadgroup float chosen_p[16];

    bool live = t < shape.experts;
    float v = live ? logits[t] : -INFINITY;
    float m = group_max(v, tmp, simd, lane, simds);
    float e = live ? exp(v - m) : 0.0f;
    float z = group_sum(e, tmp, simd, lane, simds);
    float p = live ? e / z : -1.0f;

    for (uint k = 0; k < shape.top_k; k++) {
        // Best in this simdgroup, lowest id on ties.
        float sp = simd_max(p);
        uint sid = simd_min(p == sp ? t : 0xffffffffu);
        if (lane == 0) {
            best_p[simd] = sp;
            best_id[simd] = sid;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float bp = best_p[0];
        uint bid = best_id[0];
        for (uint i = 1; i < simds; i++) {
            if (best_p[i] > bp || (best_p[i] == bp && best_id[i] < bid)) {
                bp = best_p[i];
                bid = best_id[i];
            }
        }
        if (t == bid) {
            p = -1.0f;
        }
        if (t == 0) {
            ids[k] = bid;
            chosen_p[k] = bp;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (t == 0) {
        float sum = 0;
        for (uint k = 0; k < shape.top_k; k++) {
            sum += chosen_p[k];
        }
        for (uint k = 0; k < shape.top_k; k++) {
            weights[k] = chosen_p[k] / sum;
        }
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

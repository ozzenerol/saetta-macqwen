// Quantized KV cache (src/kv_quant.rs). Each token's keys or values,
// kv_heads * 256 floats, are cut into groups of 64; a group stores BITS-bit
// codes q with a half scale s and bias z, and x = s * q + z. The codes of
// group G are u32 words 2 * BITS * G onward, the lowest bits first; the
// scale and bias of G are scales[G] and biases[G]. Token p's groups start
// at G = p * per_token.

struct KvQuantParams {
    uint groups;     // groups to write, from x[0]
    uint per_token;  // groups per token: kv_heads * 256 / 64
};

// One simdgroup per group: lane l takes values 2l and 2l + 1. The group goes
// to the token at tp.position and the ones after it.
template <uint BITS>
kernel void kv_quantize(device const float *x [[buffer(0)]],
                        device uint *data [[buffer(1)]],
                        device half *scales [[buffer(2)]],
                        device half *biases [[buffer(3)]],
                        constant KvQuantParams &p [[buffer(4)]],
                        constant TokenParams &tp [[buffer(5)]],
                        uint g [[threadgroup_position_in_grid]],
                        uint lane [[thread_index_in_simdgroup]]) {
    if (g >= p.groups) {
        return;
    }
    float a = x[g * 64 + 2 * lane];
    float b = x[g * 64 + 2 * lane + 1];
    float lo = simd_min(min(a, b));
    float hi = simd_max(max(a, b));
    const float levels = float((1u << BITS) - 1);
    half s = half((hi - lo) / levels);
    half z = half(lo);
    float inv = float(s) > 0 ? 1.0f / float(s) : 0.0f;
    uint qa = uint(clamp(rint((a - float(z)) * inv), 0.0f, levels));
    uint qb = uint(clamp(rint((b - float(z)) * inv), 0.0f, levels));
    uint dst = tp.position * p.per_token + g;
    if (lane == 0) {
        scales[dst] = s;
        biases[dst] = z;
    }
    uint pair = qa | (qb << BITS);
    if (BITS == 8) {
        uint next = simd_shuffle_down(pair, 1);
        if (lane % 2 == 0) {
            data[dst * 16 + lane / 2] = pair | (next << 16);
        }
    } else {
        uint p1 = simd_shuffle_down(pair, 1);
        uint p2 = simd_shuffle_down(pair, 2);
        uint p3 = simd_shuffle_down(pair, 3);
        if (lane % 4 == 0) {
            data[dst * 8 + lane / 4] = pair | (p1 << 8) | (p2 << 16) | (p3 << 24);
        }
    }
}

template [[host_name("kv_quantize_8")]] kernel decltype(kv_quantize<8>) kv_quantize<8>;
template [[host_name("kv_quantize_4")]] kernel decltype(kv_quantize<4>) kv_quantize<4>;

// The values of groups 0..p.groups as half, one thread per value: the
// prefill attention reads a quantized cache through this copy for now.
template <uint BITS>
kernel void kv_dequantize(device const uint *data [[buffer(0)]],
                          device const half *scales [[buffer(1)]],
                          device const half *biases [[buffer(2)]],
                          device half *out [[buffer(3)]],
                          constant KvQuantParams &p [[buffer(4)]],
                          uint i [[thread_position_in_grid]]) {
    uint g = i / 64;
    if (g >= p.groups) {
        return;
    }
    const uint per_word = 32 / BITS;
    uint j = i % 64;
    uint word = data[g * 2 * BITS + j / per_word];
    uint q = (word >> ((j % per_word) * BITS)) & ((1u << BITS) - 1);
    out[i] = half(float(scales[g]) * float(q) + float(biases[g]));
}

template [[host_name("kv_dequantize_8")]] kernel decltype(kv_dequantize<8>) kv_dequantize<8>;
template [[host_name("kv_dequantize_4")]] kernel decltype(kv_dequantize<4>) kv_dequantize<4>;

// Codes j0..j0 + 3 of group g, as floats s * q + z.
template <uint BITS>
inline float4 kv_load4(device const uint *data, device const half *scales,
                       device const half *biases, uint g, uint j0) {
    const uint per_word = 32 / BITS;
    const uint mask = (1u << BITS) - 1;
    uint word = data[g * 2 * BITS + j0 / per_word];
    uint shift = (j0 % per_word) * BITS;
    float4 q = float4(float((word >> shift) & mask), float((word >> (shift + BITS)) & mask),
                      float((word >> (shift + 2 * BITS)) & mask),
                      float((word >> (shift + 3 * BITS)) & mask));
    return q * float(scales[g]) + float(biases[g]);
}

// attention_partial (kernels/attention.metal) over a quantized cache: the
// same chunks, threads and outputs, reading the codes in place. Track B's
// attention_partial_mma_q8/q4 take the same buffers.
template <uint BITS>
kernel void attention_partial_q(device const float *q [[buffer(0)]],
                                device const uint *keys [[buffer(1)]],
                                device const uint *values [[buffer(2)]],
                                device const half *key_scales [[buffer(3)]],
                                device const half *key_biases [[buffer(4)]],
                                device const half *value_scales [[buffer(5)]],
                                device const half *value_biases [[buffer(6)]],
                                device float *part_o [[buffer(7)]],
                                device float *part_ml [[buffer(8)]],
                                constant DecodeParams &p [[buffer(9)]],
                                constant TokenParams &tp [[buffer(10)]],
                                uint2 tg [[threadgroup_position_in_grid]],
                                uint tid [[thread_index_in_threadgroup]],
                                uint simd [[simdgroup_index_in_threadgroup]],
                                uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float sc[DEC_GROUP][DEC_CHUNK];
    uint len = min(tp.kv_len, p.capacity);
    uint start = tg.x * DEC_CHUNK;
    if (start >= len) {
        return;
    }
    uint n = min(DEC_CHUNK, len - start);
    uint kv = tg.y;
    // Groups of 64 per token, and the first group of this head.
    uint per_token = p.kv_heads * DEC_DIM / 64;
    uint head = kv * DEC_DIM / 64;

    float qv[DEC_GROUP][8];
    for (uint g = 0; g < DEC_GROUP; g++) {
        device const float4 *q4 = (device const float4 *)(q + (kv * DEC_GROUP + g) * DEC_DIM + lane * 8);
        float4 a = q4[0] * p.scale, b = q4[1] * p.scale;
        qv[g][0] = a.x; qv[g][1] = a.y; qv[g][2] = a.z; qv[g][3] = a.w;
        qv[g][4] = b.x; qv[g][5] = b.y; qv[g][6] = b.z; qv[g][7] = b.w;
    }

    // Lane l reads dims 8l..8l + 7: group l / 8, codes 8 (l % 8) onward.
    for (uint t = simd; t < n; t += 8) {
        uint grp = (start + t) * per_token + head + lane / 8;
        uint j0 = 8 * (lane % 8);
        float4 a = kv_load4<BITS>(keys, key_scales, key_biases, grp, j0);
        float4 b = kv_load4<BITS>(keys, key_scales, key_biases, grp, j0 + 4);
        for (uint g = 0; g < DEC_GROUP; g++) {
            float d = qv[g][0] * a.x + qv[g][1] * a.y + qv[g][2] * a.z + qv[g][3] * a.w +
                      qv[g][4] * b.x + qv[g][5] * b.y + qv[g][6] * b.z + qv[g][7] * b.w;
            d = simd_sum(d);
            if (lane == 0) {
                sc[g][t] = d;
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    {
        uint g = simd;
        float m = -INFINITY;
        for (uint t = lane; t < n; t += 32) {
            m = max(m, sc[g][t]);
        }
        m = simd_max(m);
        float l = 0;
        for (uint t = lane; t < n; t += 32) {
            float e = exp(sc[g][t] - m);
            sc[g][t] = e;
            l += e;
        }
        l = simd_sum(l);
        if (lane == 0) {
            uint i = tg.x * p.heads + kv * DEC_GROUP + g;
            part_ml[2 * i] = m;
            part_ml[2 * i + 1] = l;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Thread tid owns dims 4 d4..4 d4 + 3: group d4 / 16, codes 4 (d4 % 16).
    threadgroup float4 sum[DEC_GROUP][DEC_DIM / 4];
    uint d4 = tid % 64, phase = tid / 64;
    float4 acc[DEC_GROUP] = {0};
    for (uint t = phase; t < n; t += 4) {
        uint grp = (start + t) * per_token + head + d4 / 16;
        float4 x = kv_load4<BITS>(values, value_scales, value_biases, grp, 4 * (d4 % 16));
        for (uint g = 0; g < DEC_GROUP; g++) {
            acc[g] += sc[g][t] * x;
        }
    }
    for (uint step = 3; step > 0; step--) {
        if (phase == step) {
            for (uint g = 0; g < DEC_GROUP; g++) {
                sum[g][d4] = step == 3 ? acc[g] : sum[g][d4] + acc[g];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (phase == 0) {
        for (uint g = 0; g < DEC_GROUP; g++) {
            device float4 *o = (device float4 *)(part_o + (tg.x * p.heads + kv * DEC_GROUP + g) * DEC_DIM);
            o[d4] = sum[g][d4] + acc[g];
        }
    }
}

template [[host_name("attention_partial_q8")]] kernel decltype(attention_partial_q<8>) attention_partial_q<8>;
template [[host_name("attention_partial_q4")]] kernel decltype(attention_partial_q<4>) attention_partial_q<4>;

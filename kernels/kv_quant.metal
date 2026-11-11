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

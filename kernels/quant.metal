// Helpers for MLX affine quantized weights, shared by qmv.metal and
// moe.metal. A lane reads 32 values at a time: BITS whole u32 words and half
// a group of 64, so with w = s * q + b the dot product with x is
// s * dot(q, x) + b * sum(x).

#include <metal_stdlib>
using namespace metal;

inline float bf16_to_float(ushort v) {
    return as_type<float>(uint(v) << 16);
}

// The i-th BITS-wide value of a little-endian bit stream held in `w`.
template <uint BITS>
inline uint unpack(thread const uint *w, uint i) {
    uint bit = i * BITS;
    uint word = bit / 32;
    uint shift = bit % 32;
    uint v = w[word] >> shift;
    if (shift + BITS > 32) {
        v |= w[word + 1] << (32 - shift);
    }
    return v & ((1u << BITS) - 1);
}

// Loads 32 floats of x into xv and returns their sum.
inline float load_x32(device const float *x, thread float *xv) {
    device const float4 *x4 = (device const float4 *)x;
    float sum = 0;
#pragma clang loop unroll(full)
    for (uint i = 0; i < 8; i++) {
        float4 v = x4[i];
        xv[4 * i] = v.x;
        xv[4 * i + 1] = v.y;
        xv[4 * i + 2] = v.z;
        xv[4 * i + 3] = v.w;
        sum += v.x + v.y + v.z + v.w;
    }
    return sum;
}

// load_x32 for x times a norm weight: xv = x * nw, returns sum(xv) and
// adds sum(x * x) to sq. The RMS scale is applied after the dot products,
// which are linear in x.
inline float load_x32_norm(device const float *x, device const float *nw, thread float *xv, thread float &sq) {
    device const float4 *x4 = (device const float4 *)x;
    device const float4 *w4 = (device const float4 *)nw;
    float sum = 0;
#pragma clang loop unroll(full)
    for (uint i = 0; i < 8; i++) {
        float4 v = x4[i];
        float4 n = v * w4[i];
        sq += dot(v, v);
        xv[4 * i] = n.x;
        xv[4 * i + 1] = n.y;
        xv[4 * i + 2] = n.z;
        xv[4 * i + 3] = n.w;
        sum += n.x + n.y + n.z + n.w;
    }
    return sum;
}

// The RMS norm scale from the lanes' sums of squares of all `cols` values.
inline float norm_inv(float sq, uint cols, float eps) {
    return rsqrt(simd_sum(sq) / float(cols) + eps);
}

// dot(w, x) over 32 values. `w` points at their BITS packed words, `group`
// is the index of their group in scales and biases.
template <uint BITS>
inline float dot32(device const uint *w,
                   device const ushort *scales,
                   device const ushort *biases,
                   uint group,
                   thread const float *xv,
                   float xsum) {
    uint packed[BITS];
#pragma clang loop unroll(full)
    for (uint j = 0; j < BITS; j++) {
        packed[j] = w[j];
    }
    float qx = 0;
#pragma clang loop unroll(full)
    for (uint i = 0; i < 32; i++) {
        qx += float(unpack<BITS>(packed, i)) * xv[i];
    }
    return bf16_to_float(scales[group]) * qx + bf16_to_float(biases[group]) * xsum;
}

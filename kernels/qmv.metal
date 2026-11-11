#include <metal_stdlib>
using namespace metal;

// Quantized matrix times vector: y = W x for an MLX affine matrix.
//
// One simdgroup computes QMV_ROWS rows. Each lane reads 32 values of a row at
// a time, which is BITS whole u32 words and half a group of 64, so a lane
// never splits a word or a group. With w = s * q + b, the dot product of a
// half group is s * dot(q, x) + b * sum(x).

constant uint QMV_ROWS = 4;
constant uint QMV_SIMDS = 2;

struct QmvShape {
    uint rows;
    uint cols;
};

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

template <uint BITS>
kernel void qmv(device const uint *words [[buffer(0)]],
                device const ushort *scales [[buffer(1)]],
                device const ushort *biases [[buffer(2)]],
                device const float *x [[buffer(3)]],
                device float *y [[buffer(4)]],
                constant QmvShape &shape [[buffer(5)]],
                uint tg [[threadgroup_position_in_grid]],
                uint simd [[simdgroup_index_in_threadgroup]],
                uint lane [[thread_index_in_simdgroup]]) {
    uint row0 = (tg * QMV_SIMDS + simd) * QMV_ROWS;
    uint row_words = shape.cols * BITS / 32;
    uint groups = shape.cols / 64;
    float acc[QMV_ROWS] = {0};

    for (uint k = lane * 32; k < shape.cols; k += 32 * 32) {
        float xv[32];
        float xsum = 0;
        device const float4 *x4 = (device const float4 *)(x + k);
#pragma clang loop unroll(full)
        for (uint i = 0; i < 8; i++) {
            float4 v = x4[i];
            xv[4 * i] = v.x;
            xv[4 * i + 1] = v.y;
            xv[4 * i + 2] = v.z;
            xv[4 * i + 3] = v.w;
            xsum += v.x + v.y + v.z + v.w;
        }
        uint g = k / 64;

#pragma clang loop unroll(full)
        for (uint r = 0; r < QMV_ROWS; r++) {
            uint row = row0 + r;
            if (row >= shape.rows) {
                break;
            }
            device const uint *w = words + row * row_words + k * BITS / 32;
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
            uint sg = row * groups + g;
            acc[r] += bf16_to_float(scales[sg]) * qx + bf16_to_float(biases[sg]) * xsum;
        }
    }

    for (uint r = 0; r < QMV_ROWS; r++) {
        float sum = simd_sum(acc[r]);
        uint row = row0 + r;
        if (lane == 0 && row < shape.rows) {
            y[row] = sum;
        }
    }
}

template [[host_name("qmv_5bit")]] kernel decltype(qmv<5>) qmv<5>;
template [[host_name("qmv_8bit")]] kernel decltype(qmv<8>) qmv<8>;

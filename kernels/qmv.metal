// Quantized matrix times vector: y = W x for an MLX affine matrix.
// One simdgroup computes QMV_ROWS rows; see quant.metal for the lane layout.
// NORM: x is first RMS-normed with weight nw (buffer 6) and eps (7), the
// norm the op before would otherwise write; every simdgroup reads all of x,
// so it gets the scale from its own loads. ACC: y += W x (into a residual).

constant uint QMV_ROWS = 4;
constant uint QMV_SIMDS = 2;

struct QmvShape {
    uint rows;
    uint cols;
};

template <uint BITS, bool NORM, bool ACC>
kernel void qmv(device const uint *words [[buffer(0)]],
                device const ushort *scales [[buffer(1)]],
                device const ushort *biases [[buffer(2)]],
                device const float *x [[buffer(3)]],
                device float *y [[buffer(4)]],
                constant QmvShape &shape [[buffer(5)]],
                device const float *nw [[buffer(6)]],
                constant float &eps [[buffer(7)]],
                uint tg [[threadgroup_position_in_grid]],
                uint simd [[simdgroup_index_in_threadgroup]],
                uint lane [[thread_index_in_simdgroup]]) {
    uint row0 = (tg * QMV_SIMDS + simd) * QMV_ROWS;
    uint row_words = shape.cols * BITS / 32;
    uint groups = shape.cols / 64;
    float acc[QMV_ROWS] = {0};
    float sq = 0;

    for (uint k = lane * 32; k < shape.cols; k += 32 * 32) {
        float xv[32];
        float xsum = NORM ? load_x32_norm(x + k, nw + k, xv, sq) : load_x32(x + k, xv);
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

    float inv = NORM ? norm_inv(sq, shape.cols, eps) : 1.0f;
    for (uint r = 0; r < QMV_ROWS; r++) {
        float sum = simd_sum(acc[r]) * inv;
        uint row = row0 + r;
        if (lane == 0 && row < shape.rows) {
            y[row] = ACC ? y[row] + sum : sum;
        }
    }
}

#define QMV(name, bits, norm, acc) \
    template [[host_name(name)]] kernel decltype(qmv<bits, norm, acc>) qmv<bits, norm, acc>;
QMV("qmv_5bit", 5, false, false)
QMV("qmv_8bit", 8, false, false)
QMV("qmv_norm_5bit", 5, true, false)
QMV("qmv_norm_8bit", 8, true, false)
QMV("qmv_acc_5bit", 5, false, true)
QMV("qmv_acc_8bit", 8, false, true)

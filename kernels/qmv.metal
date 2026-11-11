// Quantized matrix times vector: y = W x for an MLX affine matrix.
// One simdgroup computes QMV_ROWS rows; see quant.metal for the lane layout.

constant uint QMV_ROWS = 4;
constant uint QMV_SIMDS = 2;

struct QmvShape {
    uint rows;
    uint cols;
};

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
        float xsum = load_x32(x + k, xv);
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

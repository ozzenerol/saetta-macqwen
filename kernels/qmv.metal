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

// qmv_tg: qmv with x staged once per threadgroup in threadgroup memory, so
// SIMDS * QMV_ROWS rows share one read of x instead of QMV_ROWS. With NORM,
// x is RMS-normed with weight nw (buffer 6) and eps (7) while it is staged;
// the scale is applied to the sums, which are linear in x. cols <= 4096.
constant uint QMV_TG_COLS = 4096;

inline float load_x32_tg(threadgroup const float *x, thread float *xv) {
    threadgroup const float4 *x4 = (threadgroup const float4 *)x;
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

template <uint BITS, uint SIMDS, bool NORM>
kernel void qmv_tg(device const uint *words [[buffer(0)]],
                   device const ushort *scales [[buffer(1)]],
                   device const ushort *biases [[buffer(2)]],
                   device const float *x [[buffer(3)]],
                   device float *y [[buffer(4)]],
                   constant QmvShape &shape [[buffer(5)]],
                   device const float *nw [[buffer(6)]],
                   constant float &eps [[buffer(7)]],
                   uint tg [[threadgroup_position_in_grid]],
                   uint tid [[thread_index_in_threadgroup]],
                   uint simd [[simdgroup_index_in_threadgroup]],
                   uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float xs[QMV_TG_COLS];
    threadgroup float sqs[SIMDS];
    float sq = 0;
    for (uint i = tid; i < shape.cols / 4; i += SIMDS * 32) {
        float4 v = ((device const float4 *)x)[i];
        if (NORM) {
            sq += dot(v, v);
            v *= ((device const float4 *)nw)[i];
        }
        ((threadgroup float4 *)xs)[i] = v;
    }
    if (NORM) {
        sq = simd_sum(sq);
        if (lane == 0) {
            sqs[simd] = sq;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float inv = 1.0f;
    if (NORM) {
        float total = 0;
        for (uint s = 0; s < SIMDS; s++) {
            total += sqs[s];
        }
        inv = rsqrt(total / float(shape.cols) + eps);
    }

    uint row0 = (tg * SIMDS + simd) * QMV_ROWS;
    uint row_words = shape.cols * BITS / 32;
    uint groups = shape.cols / 64;
    float acc[QMV_ROWS] = {0};
    for (uint k = lane * 32; k < shape.cols; k += 32 * 32) {
        float xv[32];
        float xsum = load_x32_tg(xs + k, xv);
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
        float sum = simd_sum(acc[r]) * inv;
        uint row = row0 + r;
        if (lane == 0 && row < shape.rows) {
            y[row] = sum;
        }
    }
}

#define QMV_TG(name, bits, simds, norm) \
    template [[host_name(name)]] kernel decltype(qmv_tg<bits, simds, norm>) qmv_tg<bits, simds, norm>;
QMV_TG("qmv_tg4_5bit", 5, 4, false)
QMV_TG("qmv_tg8_5bit", 5, 8, false)
QMV_TG("qmv_tg8_8bit", 8, 8, false)
QMV_TG("qmv_tg8_norm_5bit", 5, 8, true)

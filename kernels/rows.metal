// Decode kernels for M rows at once (2 to 8), for the speculative verify:
// each weight is read from memory once and applied to all M rows. The
// summation order per row is the one of the 1-row kernel, so a row gets
// the same value it would get as a decode token.
//
// x is [M, cols] and y is [M, rows], row-major.

// qmv for M rows: the lane layout of qmv (quant.metal), QMV_ROWS weight
// rows per simdgroup, QMV_SIMDS simdgroups.
template <uint BITS, uint M>
kernel void qmv_rows(device const uint *words [[buffer(0)]],
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
    float acc[QMV_ROWS][M] = {{0}};

    for (uint k = lane * 32; k < shape.cols; k += 32 * 32) {
        float xsum[M];
#pragma clang loop unroll(full)
        for (uint j = 0; j < M; j++) {
            device const float4 *x4 = (device const float4 *)(x + j * shape.cols + k);
            float sum = 0;
#pragma clang loop unroll(full)
            for (uint i = 0; i < 8; i++) {
                float4 v = x4[i];
                sum += v.x + v.y + v.z + v.w;
            }
            xsum[j] = sum;
        }
#pragma clang loop unroll(full)
        for (uint r = 0; r < QMV_ROWS; r++) {
            uint row = row0 + r;
            if (row >= shape.rows) {
                break;
            }
            device const uint *w = words + row * row_words + k * BITS / 32;
            uint packed[BITS];
#pragma clang loop unroll(full)
            for (uint i = 0; i < BITS; i++) {
                packed[i] = w[i];
            }
            float q[32];
#pragma clang loop unroll(full)
            for (uint i = 0; i < 32; i++) {
                q[i] = float(unpack<BITS>(packed, i));
            }
            uint group = row * groups + k / 64;
            float s = bf16_to_float(scales[group]);
            float b = bf16_to_float(biases[group]);
#pragma clang loop unroll(full)
            for (uint j = 0; j < M; j++) {
                device const float4 *x4 = (device const float4 *)(x + j * shape.cols + k);
                float qx = 0;
#pragma clang loop unroll(full)
                for (uint i = 0; i < 8; i++) {
                    float4 v = x4[i];
                    qx += q[4 * i] * v.x;
                    qx += q[4 * i + 1] * v.y;
                    qx += q[4 * i + 2] * v.z;
                    qx += q[4 * i + 3] * v.w;
                }
                acc[r][j] += s * qx + b * xsum[j];
            }
        }
    }

#pragma clang loop unroll(full)
    for (uint r = 0; r < QMV_ROWS; r++) {
        uint row = row0 + r;
#pragma clang loop unroll(full)
        for (uint j = 0; j < M; j++) {
            float sum = simd_sum(acc[r][j]);
            if (lane == 0 && row < shape.rows) {
                y[j * shape.rows + row] = sum;
            }
        }
    }
}

#define QMV_ROWS_M(BITS, M) \
    template [[host_name("qmv_rows_" #BITS "bit_" #M)]] kernel decltype(qmv_rows<BITS, M>) \
        qmv_rows<BITS, M>;
#define QMV_ROWS_ALL(BITS) \
    QMV_ROWS_M(BITS, 1) QMV_ROWS_M(BITS, 2) QMV_ROWS_M(BITS, 3) QMV_ROWS_M(BITS, 4) \
    QMV_ROWS_M(BITS, 5) QMV_ROWS_M(BITS, 6) QMV_ROWS_M(BITS, 7) QMV_ROWS_M(BITS, 8)
QMV_ROWS_ALL(5)
QMV_ROWS_ALL(8)

// acc[j] += dot(w, x[j]) over 32 values for M rows of x, times wt[j] when
// WEIGHTED: the math of `acc += dot32(...)` with the weights unpacked once.
// Rows with use[j] false are skipped.
template <uint BITS, uint M, bool WEIGHTED>
inline void dot32_rows(device const uint *w,
                       float s,
                       float b,
                       device const float *const thread *x,
                       thread const bool *use,
                       thread const float *wt,
                       thread float *acc) {
    uint packed[BITS];
#pragma clang loop unroll(full)
    for (uint i = 0; i < BITS; i++) {
        packed[i] = w[i];
    }
    float q[32];
#pragma clang loop unroll(full)
    for (uint i = 0; i < 32; i++) {
        q[i] = float(unpack<BITS>(packed, i));
    }
#pragma clang loop unroll(full)
    for (uint j = 0; j < M; j++) {
        if (!use[j]) {
            continue;
        }
        device const float4 *x4 = (device const float4 *)x[j];
        float xsum = 0;
        float qx = 0;
#pragma clang loop unroll(full)
        for (uint i = 0; i < 8; i++) {
            float4 v = x4[i];
            xsum += v.x + v.y + v.z + v.w;
        }
#pragma clang loop unroll(full)
        for (uint i = 0; i < 8; i++) {
            float4 v = x4[i];
            qx += q[4 * i] * v.x;
            qx += q[4 * i + 1] * v.y;
            qx += q[4 * i + 2] * v.z;
            qx += q[4 * i + 3] * v.w;
        }
        if (WEIGHTED) {
            acc[j] += wt[j] * (s * qx + b * xsum);
        } else {
            acc[j] += s * qx + b * xsum;
        }
    }
}

// router_8bit for M rows: out is [M, rows + 1].
template <uint M>
kernel void router_rows(device const uint *gate_w [[buffer(0)]],
                        device const ushort *gate_s [[buffer(1)]],
                        device const ushort *gate_b [[buffer(2)]],
                        device const uint *shared_w [[buffer(3)]],
                        device const ushort *shared_s [[buffer(4)]],
                        device const ushort *shared_b [[buffer(5)]],
                        device const float *x [[buffer(6)]],
                        device float *out [[buffer(7)]],
                        constant RouterShape &shape [[buffer(8)]],
                        uint tg [[threadgroup_position_in_grid]],
                        uint simd [[simdgroup_index_in_threadgroup]],
                        uint lane [[thread_index_in_simdgroup]]) {
    uint row = tg * ROUTER_SIMDS + simd;
    if (row > shape.rows) {
        return;
    }
    bool shared = row == shape.rows;
    device const uint *w = shared ? shared_w : gate_w;
    device const ushort *s = shared ? shared_s : gate_s;
    device const ushort *b = shared ? shared_b : gate_b;
    uint r = shared ? 0 : row;
    uint row_words = shape.cols * 8 / 32;
    uint groups = shape.cols / 64;
    float acc[M];
#pragma clang loop unroll(full)
    for (uint j = 0; j < M; j++) {
        acc[j] = 0;
    }

    // The router weights are small and stay in cache, so each row calls
    // dot32 as router_8bit does, which keeps the rows bit for bit equal.
    for (uint k = lane * 32; k < shape.cols; k += 32 * 32) {
#pragma clang loop unroll(full)
        for (uint j = 0; j < M; j++) {
            float xv[32];
            float xsum = load_x32(x + j * shape.cols + k, xv);
            acc[j] += dot32<8>(w + r * row_words + k / 4, s, b, r * groups + k / 64, xv, xsum);
        }
    }
#pragma clang loop unroll(full)
    for (uint j = 0; j < M; j++) {
        float sum = simd_sum(acc[j]);
        if (lane == 0) {
            out[j * (shape.rows + 1) + row] = sum;
        }
    }
}

// The routed experts of M rows: each distinct expert is read once.
//
// moe_union turns ids [M, top_k] into the distinct experts in order of first
// appearance: list[0] is their count, list[1 + u] expert u, and
// slots[u * M + j] the slot where row j picked expert u, or NO_SLOT.
constant uint NO_SLOT = 0xffffffff;

struct UnionShape {
    uint rows;   // M
    uint top_k;
};

kernel void moe_union(device const uint *ids [[buffer(0)]],
                      device uint *list [[buffer(1)]],
                      device uint *slots [[buffer(2)]],
                      constant UnionShape &p [[buffer(3)]],
                      uint tid [[thread_position_in_grid]]) {
    if (tid != 0) {
        return;
    }
    uint count = 0;
    for (uint i = 0; i < p.rows * p.top_k; i++) {
        uint e = ids[i];
        uint u = 0;
        while (u < count && list[1 + u] != e) {
            u++;
        }
        if (u == count) {
            list[1 + u] = e;
            for (uint j = 0; j < p.rows; j++) {
                slots[u * p.rows + j] = NO_SLOT;
            }
            count++;
        }
        slots[u * p.rows + i / p.top_k] = i % p.top_k;
    }
    list[0] = count;
}

// moe_gate_up for M rows. Threadgroup row tg.y is the distinct expert u.
// h is [M, top_k, rows], as moe_gate_up writes it for each row.
template <uint M>
kernel void moe_gate_up_rows(device const uint *gate_w [[buffer(0)]],
                             device const ushort *gate_s [[buffer(1)]],
                             device const ushort *gate_b [[buffer(2)]],
                             device const uint *up_w [[buffer(3)]],
                             device const ushort *up_s [[buffer(4)]],
                             device const ushort *up_b [[buffer(5)]],
                             device const float *x [[buffer(6)]],
                             device float *h [[buffer(7)]],
                             constant MoeShape &shape [[buffer(8)]],
                             device const uint *list [[buffer(9)]],
                             device const uint *slots [[buffer(10)]],
                             uint2 tg [[threadgroup_position_in_grid]],
                             uint simd [[simdgroup_index_in_threadgroup]],
                             uint lane [[thread_index_in_simdgroup]]) {
    constexpr uint BITS = 5;
    uint u = tg.y;
    if (u >= list[0]) {
        return;
    }
    uint e = list[1 + u];
    uint row = e * shape.rows + tg.x * MOE_SIMDS + simd;
    uint row_words = shape.cols * BITS / 32;
    uint groups = shape.cols / 64;
    bool use[M];
    float g[M], up[M];
#pragma clang loop unroll(full)
    for (uint j = 0; j < M; j++) {
        use[j] = slots[u * M + j] != NO_SLOT;
        g[j] = 0;
        up[j] = 0;
    }

    for (uint k = lane * 32; k < shape.cols; k += 32 * 32) {
        device const float *xs[M];
#pragma clang loop unroll(full)
        for (uint j = 0; j < M; j++) {
            xs[j] = x + j * shape.cols + k;
        }
        uint w = row * row_words + k * BITS / 32;
        uint grp = row * groups + k / 64;
        dot32_rows<BITS, M, false>(gate_w + w, bf16_to_float(gate_s[grp]),
                                   bf16_to_float(gate_b[grp]), xs, use, nullptr, g);
        dot32_rows<BITS, M, false>(up_w + w, bf16_to_float(up_s[grp]), bf16_to_float(up_b[grp]),
                                   xs, use, nullptr, up);
    }

    uint out_row = tg.x * MOE_SIMDS + simd;
#pragma clang loop unroll(full)
    for (uint j = 0; j < M; j++) {
        float gs = simd_sum(g[j]);
        float us = simd_sum(up[j]);
        if (lane == 0 && use[j]) {
            uint slot = slots[u * M + j];
            h[(j * shape.count + slot) * shape.rows + out_row] = gs / (1.0f + exp(-gs)) * us;
        }
    }
}

// moe_down for M rows: y[j] = sum over row j's slots of weight * down h.
// The lanes walk the 32-value chunks of all distinct experts, so each down
// row of each expert is read once. shape.count is top_k; y is [M, rows].
template <uint M>
kernel void moe_down_rows(device const uint *words [[buffer(0)]],
                          device const ushort *scales [[buffer(1)]],
                          device const ushort *biases [[buffer(2)]],
                          device const float *h [[buffer(3)]],
                          device float *y [[buffer(4)]],
                          constant MoeShape &shape [[buffer(5)]],
                          device const uint *list [[buffer(6)]],
                          device const uint *slots [[buffer(7)]],
                          device const float *weights [[buffer(8)]],
                          uint tg [[threadgroup_position_in_grid]],
                          uint simd [[simdgroup_index_in_threadgroup]],
                          uint lane [[thread_index_in_simdgroup]]) {
    constexpr uint BITS = 5;
    uint out_row = tg * MOE_SIMDS + simd;
    uint row_words = shape.cols * BITS / 32;
    uint groups = shape.cols / 64;
    uint chunks = shape.cols / 32;
    uint count = list[0];
    float acc[M];
#pragma clang loop unroll(full)
    for (uint j = 0; j < M; j++) {
        acc[j] = 0;
    }

    for (uint c = lane; c < count * chunks; c += 32) {
        uint u = c / chunks;
        uint k = (c % chunks) * 32;
        uint row = list[1 + u] * shape.rows + out_row;
        bool use[M];
        float wt[M];
        device const float *xs[M];
#pragma clang loop unroll(full)
        for (uint j = 0; j < M; j++) {
            uint slot = slots[u * M + j];
            use[j] = slot != NO_SLOT;
            wt[j] = use[j] ? weights[j * shape.count + slot] : 0.0f;
            xs[j] = h + (j * shape.count + (use[j] ? slot : 0)) * shape.cols + k;
        }
        uint grp = row * groups + k / 64;
        dot32_rows<BITS, M, true>(words + row * row_words + k * BITS / 32,
                                  bf16_to_float(scales[grp]), bf16_to_float(biases[grp]), xs, use,
                                  wt, acc);
    }

#pragma clang loop unroll(full)
    for (uint j = 0; j < M; j++) {
        float sum = simd_sum(acc[j]);
        if (lane == 0) {
            y[j * shape.rows + out_row] = sum;
        }
    }
}

#define ROWS_M(M)                                                                              \
    template [[host_name("router_rows_" #M)]] kernel decltype(router_rows<M>) router_rows<M>;  \
    template [[host_name("moe_gate_up_rows_" #M)]] kernel decltype(moe_gate_up_rows<M>)        \
        moe_gate_up_rows<M>;                                                                   \
    template [[host_name("moe_down_rows_" #M)]] kernel decltype(moe_down_rows<M>)              \
        moe_down_rows<M>;
ROWS_M(1) ROWS_M(2) ROWS_M(3) ROWS_M(4) ROWS_M(5) ROWS_M(6) ROWS_M(7) ROWS_M(8)

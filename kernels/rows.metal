// Decode kernels for M rows at once (2 to 8), for the speculative verify:
// each weight is read from memory once and applied to all M rows. The
// summation order per row is the one of the 1-row kernel, so a row gets
// the same value it would get as a decode token.
//
// x is [M, cols] and y is [M, rows], row-major.

struct QmvRowsShape {
    uint rows;
    uint cols;
    uint m;
};

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

// qmv for up to 8 rows with simdgroup matrices: Y[rows, m] = W X^T in 8x8
// tiles. A threadgroup owns one tile of 8 weight rows; its MMA_SIMDS
// simdgroups split k, and per 32 values each dequantizes its part of the W
// tiles straight into the tiles' thread elements and loads X^T as four 8x8
// tiles. The slices are summed through threadgroup memory. x must have 8
// readable rows: rows m..7 only reach the columns of Y that are not stored.
// cols must be a multiple of 32 * MMA_SIMDS.
constant uint MMA_SIMDS = 8;

// This lane's elements of the four 8x8 W tiles of weight row `row` (the
// lane's tile row fm), values k .. k + 31, dequantized: s * q + b.
template <uint BITS>
inline void w_tiles(device const uint *words,
                    device const ushort *scales,
                    device const ushort *biases,
                    uint row,
                    uint row_words,
                    uint groups,
                    uint k,
                    uint fn,
                    thread simdgroup_float8x8 *wt) {
    device const uint *w = words + row * row_words + k * BITS / 32;
    uint packed[BITS];
    for (uint i = 0; i < BITS; i++) {
        packed[i] = w[i];
    }
    uint g = row * groups + k / 64;
    float s = bf16_to_float(scales[g]), b = bf16_to_float(biases[g]);
    for (uint i = 0; i < 4; i++) {
        wt[i].thread_elements()[0] = s * float(unpack<BITS>(packed, 8 * i + fn)) + b;
        wt[i].thread_elements()[1] = s * float(unpack<BITS>(packed, 8 * i + fn + 1)) + b;
    }
}

inline uint2 frag(uint lane) {
    uint qid = lane / 4;
    return uint2((qid & 4) + (lane / 2) % 4, (qid & 2) * 2 + (lane % 2) * 2);
}

// The 8x8 tile `acc` of each simdgroup summed over the threadgroup:
// returns, for threads 0..63, element (tid / 8, tid % 8).
inline float sum_tiles(simdgroup_float8x8 acc, threadgroup float *red, uint simd, uint tid) {
    simdgroup_store(acc, red + simd * 64, 8);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float sum = 0;
    if (tid < 64) {
        for (uint i = 0; i < MMA_SIMDS; i++) {
            sum += red[i * 64 + tid];
        }
    }
    return sum;
}

template <uint BITS>
kernel void qmv_mma(device const uint *words [[buffer(0)]],
                    device const ushort *scales [[buffer(1)]],
                    device const ushort *biases [[buffer(2)]],
                    device const float *x [[buffer(3)]],
                    device float *y [[buffer(4)]],
                    constant QmvRowsShape &shape [[buffer(5)]],
                    uint tg [[threadgroup_position_in_grid]],
                    uint tid [[thread_index_in_threadgroup]],
                    uint simd [[simdgroup_index_in_threadgroup]],
                    uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float red[MMA_SIMDS * 64];
    uint2 f = frag(lane);
    uint row_words = shape.cols * BITS / 32;
    uint groups = shape.cols / 64;
    uint row0 = tg * 8;
    uint row = min(row0 + f.x, shape.rows - 1);
    uint slice = shape.cols / MMA_SIMDS;

    simdgroup_float8x8 acc = simdgroup_float8x8(0);
    for (uint k = simd * slice; k < (simd + 1) * slice; k += 32) {
        simdgroup_float8x8 xt[4], wt[4];
        for (uint i = 0; i < 4; i++) {
            simdgroup_load(xt[i], x + k + 8 * i, shape.cols, ulong2(0, 0), true);
        }
        w_tiles<BITS>(words, scales, biases, row, row_words, groups, k, f.y, wt);
        for (uint i = 0; i < 4; i++) {
            simdgroup_multiply_accumulate(acc, wt[i], xt[i], acc);
        }
    }
    float sum = sum_tiles(acc, red, simd, tid);
    uint r = row0 + tid / 8, j = tid % 8;
    if (tid < 64 && r < shape.rows && j < shape.m) {
        y[j * shape.rows + r] = sum;
    }
}

template [[host_name("qmv_mma_5bit")]] kernel decltype(qmv_mma<5>) qmv_mma<5>;
template [[host_name("qmv_mma_8bit")]] kernel decltype(qmv_mma<8>) qmv_mma<8>;

// moe_gate_up_rows with simdgroup matrices, split k like qmv_mma:
// threadgroup (tile of 8 rows, distinct expert u). All 8 rows of x go
// through u and only the rows that picked it are stored. x must have 8
// readable rows.
kernel void moe_gate_up_mma(device const uint *gate_w [[buffer(0)]],
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
                            constant uint &m [[buffer(11)]],
                            uint2 tg [[threadgroup_position_in_grid]],
                            uint tid [[thread_index_in_threadgroup]],
                            uint simd [[simdgroup_index_in_threadgroup]],
                            uint lane [[thread_index_in_simdgroup]]) {
    constexpr uint BITS = 5;
    threadgroup float red[MMA_SIMDS * 64];
    uint u = tg.y;
    if (u >= list[0]) {
        return;
    }
    uint2 f = frag(lane);
    uint e = list[1 + u];
    uint row = e * shape.rows + tg.x * 8 + f.x;
    uint row_words = shape.cols * BITS / 32;
    uint groups = shape.cols / 64;
    uint slice = shape.cols / MMA_SIMDS;
    simdgroup_float8x8 g = simdgroup_float8x8(0), up = simdgroup_float8x8(0);
    for (uint k = simd * slice; k < (simd + 1) * slice; k += 32) {
        simdgroup_float8x8 xt[4], wg[4], wu[4];
        for (uint i = 0; i < 4; i++) {
            simdgroup_load(xt[i], x + k + 8 * i, shape.cols, ulong2(0, 0), true);
        }
        w_tiles<BITS>(gate_w, gate_s, gate_b, row, row_words, groups, k, f.y, wg);
        w_tiles<BITS>(up_w, up_s, up_b, row, row_words, groups, k, f.y, wu);
        for (uint i = 0; i < 4; i++) {
            simdgroup_multiply_accumulate(g, wg[i], xt[i], g);
            simdgroup_multiply_accumulate(up, wu[i], xt[i], up);
        }
    }
    float gs = sum_tiles(g, red, simd, tid);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float us = sum_tiles(up, red, simd, tid);
    uint out_row = tg.x * 8 + tid / 8, j = tid % 8;
    if (tid < 64 && j < m) {
        uint slot = slots[u * m + j];
        if (slot != NO_SLOT) {
            h[(j * shape.count + slot) * shape.rows + out_row] = gs / (1.0f + exp(-gs)) * us;
        }
    }
}

// moe_down_rows with simdgroup matrices, split k like qmv_mma: for each
// distinct expert u, the X^T tile holds weight(j, u) * h[j, slot(j, u)] in
// column j (0 where row j did not pick u), so one accumulator sums all
// experts. Threadgroup tg owns 8 output rows; y is [m, rows].
kernel void moe_down_mma(device const uint *words [[buffer(0)]],
                         device const ushort *scales [[buffer(1)]],
                         device const ushort *biases [[buffer(2)]],
                         device const float *h [[buffer(3)]],
                         device float *y [[buffer(4)]],
                         constant MoeShape &shape [[buffer(5)]],
                         device const uint *list [[buffer(6)]],
                         device const uint *slots [[buffer(7)]],
                         device const float *weights [[buffer(8)]],
                         constant uint &m [[buffer(9)]],
                         uint tg [[threadgroup_position_in_grid]],
                         uint tid [[thread_index_in_threadgroup]],
                         uint simd [[simdgroup_index_in_threadgroup]],
                         uint lane [[thread_index_in_simdgroup]]) {
    constexpr uint BITS = 5;
    threadgroup float red[MMA_SIMDS * 64];
    uint2 f = frag(lane);
    uint row_words = shape.cols * BITS / 32;
    uint groups = shape.cols / 64;
    uint slice = shape.cols / MMA_SIMDS;
    uint count = list[0];
    simdgroup_float8x8 acc = simdgroup_float8x8(0);
    for (uint u = 0; u < count; u++) {
        uint row = list[1 + u] * shape.rows + tg * 8 + f.x;
        // This lane's two columns j = f.y, f.y + 1 of X^T.
        float wt[2];
        device const float *hj[2];
        for (uint i = 0; i < 2; i++) {
            uint j = f.y + i;
            uint slot = j < m ? slots[u * m + j] : NO_SLOT;
            wt[i] = slot != NO_SLOT ? weights[j * shape.count + slot] : 0.0f;
            hj[i] = h + (slot != NO_SLOT ? j * shape.count + slot : 0) * shape.cols;
        }
        for (uint k = simd * slice; k < (simd + 1) * slice; k += 32) {
            simdgroup_float8x8 xt[4], wd[4];
            for (uint i = 0; i < 4; i++) {
                // Element (kk, j) of tile i is h[j, slot][k + 8 i + kk];
                // this lane holds kk = f.x and j = f.y, f.y + 1.
                xt[i].thread_elements()[0] = wt[0] * hj[0][k + 8 * i + f.x];
                xt[i].thread_elements()[1] = wt[1] * hj[1][k + 8 * i + f.x];
            }
            w_tiles<BITS>(words, scales, biases, row, row_words, groups, k, f.y, wd);
            for (uint i = 0; i < 4; i++) {
                simdgroup_multiply_accumulate(acc, wd[i], xt[i], acc);
            }
        }
    }
    float sum = sum_tiles(acc, red, simd, tid);
    uint j = tid % 8;
    if (tid < 64 && j < m) {
        y[j * shape.rows + tg * 8 + tid / 8] = sum;
    }
}

// qmv for M <= 4 rows with x in registers: each lane takes 16 values of k
// per step (32 lanes cover 512), holds those 16 values of all M rows of x,
// and reuses them for the QR weight rows of its simdgroup. 16 BITS-bit
// values start at a word (8-bit) or at bit 0 or 16 of a word (5-bit).
// Weight rows per simdgroup of the reg kernels (qmv_reg takes it as a
// template parameter) and simdgroups per threadgroup.
constant uint MOE_QR = 8;
constant uint QR_SIMDS = 2;

template <uint BITS>
inline uint unpack_at(thread const uint *w, uint bit) {
    uint word = bit / 32;
    uint shift = bit % 32;
    uint v = w[word] >> shift;
    if (shift + BITS > 32) {
        v |= w[word + 1] << (32 - shift);
    }
    return v & ((1u << BITS) - 1);
}

struct QmvRegShape {
    uint rows;
    uint cols;
    uint y_stride;  // floats between rows of y
};

template <uint BITS, uint M, uint QR>
kernel void qmv_reg(device const uint *words [[buffer(0)]],
                    device const ushort *scales [[buffer(1)]],
                    device const ushort *biases [[buffer(2)]],
                    device const float *x [[buffer(3)]],
                    device float *y [[buffer(4)]],
                    constant QmvRegShape &shape [[buffer(5)]],
                    uint tg [[threadgroup_position_in_grid]],
                    uint simd [[simdgroup_index_in_threadgroup]],
                    uint lane [[thread_index_in_simdgroup]]) {
    constexpr uint NW = BITS == 5 ? 3 : 4;
    uint row0 = (tg * QR_SIMDS + simd) * QR;
    uint row_words = shape.cols * BITS / 32;
    uint groups = shape.cols / 64;
    float acc[QR][M];
    for (uint r = 0; r < QR; r++) {
        for (uint j = 0; j < M; j++) {
            acc[r][j] = 0;
        }
    }

    for (uint k = lane * 16; k < shape.cols; k += 16 * 32) {
        float4 xv[M][4];
        float xsum[M];
        for (uint j = 0; j < M; j++) {
            device const float4 *x4 = (device const float4 *)(x + j * shape.cols + k);
            float sum = 0;
            for (uint i = 0; i < 4; i++) {
                xv[j][i] = x4[i];
                sum += xv[j][i].x + xv[j][i].y + xv[j][i].z + xv[j][i].w;
            }
            xsum[j] = sum;
        }
        uint bit0 = k * BITS;
        for (uint r = 0; r < QR; r++) {
            uint row = row0 + r;
            if (row >= shape.rows) {
                break;
            }
            device const uint *w = words + row * row_words + bit0 / 32;
            uint packed[NW];
            for (uint i = 0; i < NW; i++) {
                packed[i] = w[i];
            }
            float4 q[4];
            uint off = bit0 % 32;
            for (uint i = 0; i < 4; i++) {
                q[i] = float4(unpack_at<BITS>(packed, off + (4 * i) * BITS),
                              unpack_at<BITS>(packed, off + (4 * i + 1) * BITS),
                              unpack_at<BITS>(packed, off + (4 * i + 2) * BITS),
                              unpack_at<BITS>(packed, off + (4 * i + 3) * BITS));
            }
            uint g = row * groups + k / 64;
            float s = bf16_to_float(scales[g]), b = bf16_to_float(biases[g]);
            for (uint j = 0; j < M; j++) {
                float qx = dot(q[0], xv[j][0]) + dot(q[1], xv[j][1]) + dot(q[2], xv[j][2]) +
                           dot(q[3], xv[j][3]);
                acc[r][j] += s * qx + b * xsum[j];
            }
        }
    }

    for (uint r = 0; r < QR; r++) {
        uint row = row0 + r;
        for (uint j = 0; j < M; j++) {
            float sum = simd_sum(acc[r][j]);
            if (lane == 0 && row < shape.rows) {
                y[j * shape.y_stride + row] = sum;
            }
        }
    }
}

#define QMV_REG(BITS, M, QR) \
    template [[host_name("qmv_reg_" #BITS "bit_" #M "_" #QR)]] kernel \
        decltype(qmv_reg<BITS, M, QR>) qmv_reg<BITS, M, QR>;
#define QMV_REG_QR(QR) \
    QMV_REG(5, 1, QR) QMV_REG(5, 2, QR) QMV_REG(5, 3, QR) QMV_REG(5, 4, QR) \
    QMV_REG(8, 1, QR) QMV_REG(8, 2, QR) QMV_REG(8, 3, QR) QMV_REG(8, 4, QR)
QMV_REG_QR(2) QMV_REG_QR(4) QMV_REG_QR(8)

// This lane's 16 values of weight row `row` from k, dequantized as q (the
// scale and bias are applied by the caller): the qmv_reg layout.
template <uint BITS>
inline void q16(device const uint *words, uint row, uint row_words, uint k, thread float4 *q) {
    constexpr uint NW = BITS == 5 ? 3 : 4;
    uint bit0 = k * BITS;
    device const uint *w = words + row * row_words + bit0 / 32;
    uint packed[NW];
    for (uint i = 0; i < NW; i++) {
        packed[i] = w[i];
    }
    uint off = bit0 % 32;
    for (uint i = 0; i < 4; i++) {
        q[i] = float4(unpack_at<BITS>(packed, off + (4 * i) * BITS),
                      unpack_at<BITS>(packed, off + (4 * i + 1) * BITS),
                      unpack_at<BITS>(packed, off + (4 * i + 2) * BITS),
                      unpack_at<BITS>(packed, off + (4 * i + 3) * BITS));
    }
}

// moe_gate_up_rows with x in registers, for rows r0 .. r0 + M - 1 of the
// verify (M <= 4; more rows take another dispatch with another r0).
// Threadgroup (block of QR * QR_SIMDS output rows, distinct expert u); a
// threadgroup whose rows did not pick u returns at once.
template <uint M>
kernel void moe_gate_up_reg(device const uint *gate_w [[buffer(0)]],
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
                            constant uint2 &rows [[buffer(11)]],  // m, r0
                            uint2 tg [[threadgroup_position_in_grid]],
                            uint simd [[simdgroup_index_in_threadgroup]],
                            uint lane [[thread_index_in_simdgroup]]) {
    constexpr uint BITS = 5;
    uint u = tg.y;
    if (u >= list[0]) {
        return;
    }
    uint m = rows.x, r0 = rows.y;
    uint slot[M];
    bool any = false;
    for (uint j = 0; j < M; j++) {
        slot[j] = r0 + j < m ? slots[u * m + r0 + j] : NO_SLOT;
        any = any || slot[j] != NO_SLOT;
    }
    if (!any) {
        return;
    }
    uint e = list[1 + u];
    uint out0 = (tg.x * QR_SIMDS + simd) * MOE_QR;
    uint row_words = shape.cols * BITS / 32;
    uint groups = shape.cols / 64;
    float g[MOE_QR][M], up[MOE_QR][M];
    for (uint r = 0; r < MOE_QR; r++) {
        for (uint j = 0; j < M; j++) {
            g[r][j] = 0;
            up[r][j] = 0;
        }
    }

    for (uint k = lane * 16; k < shape.cols; k += 16 * 32) {
        float4 xv[M][4];
        float xsum[M];
        for (uint j = 0; j < M; j++) {
            device const float4 *x4 = (device const float4 *)(x + (r0 + j) * shape.cols + k);
            float sum = 0;
            for (uint i = 0; i < 4; i++) {
                xv[j][i] = slot[j] != NO_SLOT ? x4[i] : float4(0);
                sum += xv[j][i].x + xv[j][i].y + xv[j][i].z + xv[j][i].w;
            }
            xsum[j] = sum;
        }
        for (uint r = 0; r < MOE_QR; r++) {
            uint row = e * shape.rows + out0 + r;
            uint grp = row * groups + k / 64;
            float4 qg[4], qu[4];
            q16<BITS>(gate_w, row, row_words, k, qg);
            q16<BITS>(up_w, row, row_words, k, qu);
            float gs = bf16_to_float(gate_s[grp]), gb = bf16_to_float(gate_b[grp]);
            float us = bf16_to_float(up_s[grp]), ub = bf16_to_float(up_b[grp]);
            for (uint j = 0; j < M; j++) {
                float dg = dot(qg[0], xv[j][0]) + dot(qg[1], xv[j][1]) + dot(qg[2], xv[j][2]) +
                           dot(qg[3], xv[j][3]);
                float du = dot(qu[0], xv[j][0]) + dot(qu[1], xv[j][1]) + dot(qu[2], xv[j][2]) +
                           dot(qu[3], xv[j][3]);
                g[r][j] += gs * dg + gb * xsum[j];
                up[r][j] += us * du + ub * xsum[j];
            }
        }
    }

    for (uint r = 0; r < MOE_QR; r++) {
        for (uint j = 0; j < M; j++) {
            float gsum = simd_sum(g[r][j]);
            float usum = simd_sum(up[r][j]);
            if (lane == 0 && slot[j] != NO_SLOT) {
                uint n = ((r0 + j) * shape.count + slot[j]) * shape.rows + out0 + r;
                h[n] = gsum / (1.0f + exp(-gsum)) * usum;
            }
        }
    }
}

template [[host_name("moe_gate_up_reg_1")]] kernel decltype(moe_gate_up_reg<1>) moe_gate_up_reg<1>;
template [[host_name("moe_gate_up_reg_2")]] kernel decltype(moe_gate_up_reg<2>) moe_gate_up_reg<2>;
template [[host_name("moe_gate_up_reg_3")]] kernel decltype(moe_gate_up_reg<3>) moe_gate_up_reg<3>;
template [[host_name("moe_gate_up_reg_4")]] kernel decltype(moe_gate_up_reg<4>) moe_gate_up_reg<4>;

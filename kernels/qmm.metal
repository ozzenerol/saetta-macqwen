// Quantized matrix times matrix for prefill, the fast path:
//   y[m, n] = sum over k of x[m, k] * W[n, k]
// with x in half, W MLX-quantized (BITS, groups of 64), y in f32. Same
// buffers, gather modes and tiles as qmm in prefill.metal, so it drops in
// there. W is dequantized to half (scale q + bias) on its way into
// threadgroup memory, as MLX does, so the products are plain half matrix
// MACs with f32 sums.
//
// A threadgroup of 4 simdgroups computes a BM x BN tile of y, each
// simdgroup 32 x 32 of it in 8 x 8 matrices. Each step takes QMM_BK of k:
// every thread loads x values and dequantizes weights; double buffered, it
// loads the next step into registers while the simdgroups multiply the
// current one and stores them into the other buffer, one barrier per step.

constant uint QMM_BK = 32;
constant uint QMM_LD = QMM_BK + 8;  // padding against bank conflicts
constant uint QMM_FEW = 8;          // tiles this small go to qmm_few

struct QmmShape {
    uint m;
    uint n;
    uint k;
    uint gather;
};

template <uint BM, uint BN>
struct QmmStage {
    half x[BM * QMM_LD];
    half w[BN * QMM_LD];
};

// Thread t loads XE values of x row t / (32 / XE) and dequantizes WE
// weights of W row t / (32 / WE) for each step.
template <uint BITS, uint BM, uint BN>
struct QmmLoader {
    device const half *xrow;
    device const uint *wrow;
    device const ushort *srow;
    device const ushort *brow;
    bool x_ok;
    bool w_ok;
    uint xr, xc, wr, wc;

    half4 xv[BM / 16];
    uint packed[BITS];
    float scale;
    float bias;

    void load(uint k0) {
        if (x_ok) {
            device const half4 *p = (device const half4 *)(xrow + k0 + xc);
            for (uint i = 0; i < BM / 16; i++) {
                xv[i] = p[i];
            }
        }
        if (w_ok) {
            device const uint *p = wrow + k0 / 32 * BITS;
            for (uint j = 0; j < BITS; j++) {
                packed[j] = p[j];
            }
            scale = bf16_to_float(srow[k0 / 64]);
            bias = bf16_to_float(brow[k0 / 64]);
        }
    }

    void store(threadgroup QmmStage<BM, BN> &s) {
        threadgroup half4 *xd = (threadgroup half4 *)(s.x + xr * QMM_LD + xc);
        for (uint i = 0; i < BM / 16; i++) {
            xd[i] = x_ok ? xv[i] : half4(0);
        }
        threadgroup half4 *wd = (threadgroup half4 *)(s.w + wr * QMM_LD + wc);
#pragma clang loop unroll(full)
        for (uint i = 0; i < BN / 16; i++) {
            uint c = wc + 4 * i;
            float4 q = float4(unpack<BITS>(packed, c), unpack<BITS>(packed, c + 1),
                              unpack<BITS>(packed, c + 2), unpack<BITS>(packed, c + 3));
            wd[i] = w_ok ? half4(q * scale + bias) : half4(0);
        }
    }
};

// BM x BN tile, 4 simdgroups of TM x TN (2 x 2 of 32 x 32 for 64 x 64,
// 1 x 4 of 32 x 32 for 32 x 128, 1 x 4 of 16 x 32 for 16 x 128).
// DB: double buffered. SUB: threadgroups per expert tile in gather mode,
// each taking BM rows of a tile of up to SUB BM.
template <uint BITS, uint BM, uint BN, uint TM, uint TN, bool DB, uint SUB = 1>
kernel void qmm2(device const uint *words [[buffer(0)]],
                 device const ushort *scales [[buffer(1)]],
                 device const ushort *biases [[buffer(2)]],
                 device const half *x [[buffer(3)]],
                 device float *y [[buffer(4)]],
                 constant QmmShape &s [[buffer(5)]],
                 device const uint4 *tiles [[buffer(6)]],
                 device const uint *rows [[buffer(7)]],
                 device const uint *tile_count [[buffer(8)]],
                 uint2 tg [[threadgroup_position_in_grid]],
                 uint tid [[thread_index_in_threadgroup]],
                 uint simd [[simdgroup_index_in_threadgroup]],
                 uint lane [[thread_index_in_simdgroup]]) {
    constexpr uint WN = BN / TN;
    constexpr uint FM = TM / 8, FN = TN / 8;
    threadgroup QmmStage<BM, BN> stage[DB ? 2 : 1];
    threadgroup float edge[4][64];

    uint expert = 0, first, count;
    if (s.gather != 0) {
        uint ti = tg.y / SUB, part = tg.y % SUB;
        if (ti >= tile_count[0]) {
            return;
        }
        uint4 tile = tiles[ti];
        if (tile.z <= QMM_FEW || part * BM >= tile.z) {
            return;  // qmm_few takes it, or the tile has no rows here
        }
        expert = tile.x;
        first = tile.y + part * BM;
        count = min(BM, tile.z - part * BM);
    } else {
        first = tg.y * BM;
        count = min(BM, s.m - first);
    }
    uint n0 = tg.x * BN;

    constexpr uint XE = BM / 4, WE = BN / 4;
    QmmLoader<BITS, BM, BN> ld;
    ld.xr = tid / (32 / XE);
    ld.xc = (tid % (32 / XE)) * XE;
    ld.wr = tid / (32 / WE);
    ld.wc = (tid % (32 / WE)) * WE;
    ld.x_ok = ld.xr < count;
    ld.xrow = x;
    if (ld.x_ok) {
        uint r = s.gather == 1 ? rows[first + ld.xr] : first + ld.xr;
        ld.xrow = x + ulong(r) * s.k;
    }
    ld.w_ok = n0 + ld.wr < s.n;
    ulong wrow = ulong(expert) * s.n + (ld.w_ok ? n0 + ld.wr : 0);
    ld.wrow = words + wrow * (s.k * BITS / 32);
    ld.srow = scales + wrow * (s.k / 64);
    ld.brow = biases + wrow * (s.k / 64);

    uint sm = (simd / WN) * TM;
    uint sn = (simd % WN) * TN;
    // Rows past the end of a partial tile: no MACs, nothing to write.
    bool active = sm < count;
    // 8-row blocks of this simdgroup that hold rows; the rest skip their MACs.
    uint live = active ? min(FM, (count - sm + 7) / 8) : 0;
    simdgroup_float8x8 acc[FM][FN];
    for (uint i = 0; i < FM; i++) {
        for (uint j = 0; j < FN; j++) {
            acc[i][j] = simdgroup_float8x8(0);
        }
    }

    if (DB) {
        ld.load(0);
        ld.store(stage[0]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    uint cur = 0;
    for (uint k0 = 0; k0 < s.k; k0 += QMM_BK) {
        bool more = k0 + QMM_BK < s.k;
        if (DB) {
            if (more) {
                ld.load(k0 + QMM_BK);
            }
        } else {
            ld.load(k0);
            ld.store(stage[0]);
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        threadgroup const QmmStage<BM, BN> &st = stage[cur];
        if (active) {
#pragma clang loop unroll(full)
            for (uint kk = 0; kk < QMM_BK; kk += 8) {
                simdgroup_half8x8 b[FN];
                for (uint j = 0; j < FN; j++) {
                    simdgroup_load(b[j], st.w + (sn + 8 * j) * QMM_LD + kk, QMM_LD, ulong2(0, 0), true);
                }
                for (uint i = 0; i < FM; i++) {
                    if (i >= live) {
                        break;
                    }
                    simdgroup_half8x8 a;
                    simdgroup_load(a, st.x + (sm + 8 * i) * QMM_LD + kk, QMM_LD);
                    for (uint j = 0; j < FN; j++) {
                        simdgroup_multiply_accumulate(acc[i][j], a, b[j], acc[i][j]);
                    }
                }
            }
        }
        if (DB) {
            if (more) {
                ld.store(stage[cur ^ 1]);
            }
            cur ^= 1;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (!active) {
        return;
    }
    device float *y0 = y + ulong(first) * s.n + n0;
    if (count == BM && n0 + BN <= s.n) {
        for (uint i = 0; i < FM; i++) {
            for (uint j = 0; j < FN; j++) {
                simdgroup_store(acc[i][j], y0 + ulong(sm + 8 * i) * s.n + sn + 8 * j, s.n);
            }
        }
        return;
    }
    // A partial tile goes out 8 x 8 at a time with bounds checks. Lane l
    // writes row l / 4, columns 2 (l % 4) and + 1.
    threadgroup float *e = edge[simd];
    uint er = lane / 4, ec = (lane % 4) * 2;
    for (uint i = 0; i < FM; i++) {
        for (uint j = 0; j < FN; j++) {
            simdgroup_store(acc[i][j], e, 8);
            simdgroup_barrier(mem_flags::mem_threadgroup);
            uint row = sm + 8 * i + er;
            uint col = sn + 8 * j + ec;
            if (row < count) {
                for (uint c = 0; c < 2; c++) {
                    if (n0 + col + c < s.n) {
                        y0[ulong(row) * s.n + col + c] = e[er * 8 + ec + c];
                    }
                }
            }
            simdgroup_barrier(mem_flags::mem_threadgroup);
        }
    }
}

#define QMM2(name, bits, bm, bn, tm, tn, db) \
    template [[host_name(name)]] kernel decltype(qmm2<bits, bm, bn, tm, tn, db>) qmm2<bits, bm, bn, tm, tn, db>;
#define QMM2_SUB(name, bits, sub) \
    template [[host_name(name)]] kernel decltype(qmm2<bits, 32, 128, 32, 32, false, sub>) \
        qmm2<bits, 32, 128, 32, 32, false, sub>;
QMM2_SUB("qmm2_32x2_5bit", 5, 2)
QMM2_SUB("qmm2_32x2_8bit", 8, 2)
QMM2("qmm2_5bit", 5, 64, 64, 32, 32, false)
QMM2("qmm2_8bit", 8, 64, 64, 32, 32, false)
QMM2("qmm2_db_5bit", 5, 64, 64, 32, 32, true)
QMM2("qmm2_db_8bit", 8, 64, 64, 32, 32, true)
QMM2("qmm2_32_5bit", 5, 32, 128, 32, 32, false)
QMM2("qmm2_32_8bit", 8, 32, 128, 32, 32, false)
QMM2("qmm2_32_db_5bit", 5, 32, 128, 32, 32, true)
QMM2("qmm2_32_db_8bit", 8, 32, 128, 32, 32, true)
QMM2("qmm2_16_5bit", 5, 16, 128, 16, 32, false)
QMM2("qmm2_16_8bit", 8, 16, 128, 16, 32, false)

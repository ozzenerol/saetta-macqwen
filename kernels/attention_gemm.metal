// Prefill attention as matrix products, one block of up to
// PREFILL_AG_KEYS keys at a time (src/attention_gemm.rs). For kv head g
// the query rows are r = token * 8 + h for its 8 q heads, so all of them
// share the keys and values of g. Per block:
//   prefill_ag_scores   S = Q K^T for the block, in 64 x 64 tiles
//   prefill_ag_softmax  per row: running max m and sum l, P = exp(S - m)
//                       in half, and the factor alpha the old output gets
//   prefill_ag_pv       O = alpha O + P V, in 64 x 64 tiles
// then prefill_ag_final writes y = O / l * sigmoid(gate). Key j is visible
// to the row of token i when j <= pos + i. The _q8 and _q4 kernels read
// an 8- or 4-bit cache (kv_quant.metal) in place: the codes, scales and
// biases go in the buffers after the params, and the half cache buffer is
// not read.

constant uint PREFILL_AG_KEYS = 1024;  // keys per block, the S row stride
constant uint PREFILL_AG_D = 256;      // head dim
constant uint PREFILL_AG_GROUP = 8;    // q heads per kv head
constant uint PREFILL_AG_BK = 32;
constant uint PREFILL_AG_LD = PREFILL_AG_BK + 8;
constant uint PREFILL_AG_VLD = 64 + 8;

struct AttGemmParams {
    uint rows;      // tokens * PREFILL_AG_GROUP
    uint rows_pad;  // rows rounded up to 64: the row count of S, P and O
    uint heads;
    uint kv_heads;
    uint pos;       // position of the chunk's first token
    uint j0;        // first key of the block
    uint n;         // keys in the block
    float scale;
};

// The last key row r may see, relative to the block: pos + token - j0,
// negative when it sees none of the block.
inline int prefill_ag_limit(constant AttGemmParams &p, uint r) {
    return int(p.pos + r / PREFILL_AG_GROUP) - int(p.j0);
}

// The keys of the block the 64-row tile of row r reads, rounded up to
// PREFILL_AG_BK: up to the last key its last row sees. On the diagonal
// that is about half the block, so pv does not multiply zeros.
inline uint prefill_ag_tile_keys(constant AttGemmParams &p, uint r) {
    uint last_row = min((r / 64) * 64 + 63, p.rows - 1);
    int limit = prefill_ag_limit(p, last_row);
    uint seen = limit < 0 ? 0 : min(p.n, uint(limit) + 1);
    return (seen + PREFILL_AG_BK - 1) / PREFILL_AG_BK * PREFILL_AG_BK;
}

// Offset of query row r of kv head g in q [tokens][heads][D].
inline ulong prefill_ag_q_row(constant AttGemmParams &p, uint g, uint r) {
    uint token = r / PREFILL_AG_GROUP, h = r % PREFILL_AG_GROUP;
    return (ulong(token) * p.heads + g * PREFILL_AG_GROUP + h) * PREFILL_AG_D;
}

// Offset of key (or value) j of kv head g in the cache [pos][kv_heads][D].
inline ulong prefill_ag_kv_row(constant AttGemmParams &p, uint g, uint j) {
    return (ulong(j) * p.kv_heads + g) * PREFILL_AG_D;
}

// Dims dim .. dim + 3 of key (or value) j of kv head g: from the half
// cache h, or with BITS 8 or 4 from the codes in place.
template <uint BITS>
inline half4 prefill_ag_kv4(device const half *h, device const uint *codes,
                            device const half *scales, device const half *biases,
                            constant AttGemmParams &p, uint g, uint j, uint dim) {
    if (BITS == 16) {
        return *(device const half4 *)(h + prefill_ag_kv_row(p, g, j) + dim);
    }
    uint group = (j * p.kv_heads + g) * (PREFILL_AG_D / 64) + dim / 64;
    return half4(kv_load4<BITS>(codes, scales, biases, group, dim % 64));
}

// S[g][r][jj] (unscaled) for rows tg.y * 64 .., block keys tg.x * 64 ..
// Tiles no row sees are skipped; the softmax never reads them.
template <uint BITS>
kernel void prefill_ag_scores(device const half *q [[buffer(0)]],
                              device const half *keys [[buffer(1)]],
                              device float *s [[buffer(2)]],
                              constant AttGemmParams &p [[buffer(3)]],
                              device const uint *codes [[buffer(4)]],
                              device const half *scales [[buffer(5)]],
                              device const half *biases [[buffer(6)]],
                              uint3 tg [[threadgroup_position_in_grid]],
                              uint tid [[thread_index_in_threadgroup]],
                              uint simd [[simdgroup_index_in_threadgroup]]) {
    threadgroup half qt[64 * PREFILL_AG_LD];
    threadgroup half kt[64 * PREFILL_AG_LD];
    uint g = tg.z, r0 = tg.y * 64, k0 = tg.x * 64;
    uint last_row = min(r0 + 63, p.rows - 1);
    if (int(k0) > prefill_ag_limit(p, last_row)) {
        return;
    }

    // Thread t loads 16 values of q row t / 2 and of key row t / 2.
    uint lr = tid / 2, lc = (tid % 2) * 16;
    bool q_ok = r0 + lr < p.rows;
    bool k_ok = k0 + lr < p.n;
    device const half *qp = q + (q_ok ? prefill_ag_q_row(p, g, r0 + lr) : 0) + lc;
    uint key = p.j0 + k0 + lr;
    threadgroup half4 *qd = (threadgroup half4 *)(qt + lr * PREFILL_AG_LD + lc);
    threadgroup half4 *kd = (threadgroup half4 *)(kt + lr * PREFILL_AG_LD + lc);

    uint sm = (simd / 2) * 32, sn = (simd % 2) * 32;
    simdgroup_float8x8 acc[4][4];
    for (uint i = 0; i < 4; i++) {
        for (uint j = 0; j < 4; j++) {
            acc[i][j] = simdgroup_float8x8(0);
        }
    }
    for (uint d0 = 0; d0 < PREFILL_AG_D; d0 += PREFILL_AG_BK) {
        for (uint i = 0; i < 4; i++) {
            qd[i] = q_ok ? ((device const half4 *)(qp + d0))[i] : half4(0);
            kd[i] = k_ok ? prefill_ag_kv4<BITS>(keys, codes, scales, biases, p, g, key, d0 + lc + 4 * i)
                         : half4(0);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
#pragma clang loop unroll(full)
        for (uint kk = 0; kk < PREFILL_AG_BK; kk += 8) {
            simdgroup_half8x8 a[4];
            simdgroup_half8x8 b[4];
            for (uint i = 0; i < 4; i++) {
                simdgroup_load(a[i], qt + (sm + 8 * i) * PREFILL_AG_LD + kk, PREFILL_AG_LD);
                simdgroup_load(b[i], kt + (sn + 8 * i) * PREFILL_AG_LD + kk, PREFILL_AG_LD, ulong2(0, 0), true);
            }
            for (uint i = 0; i < 4; i++) {
                for (uint j = 0; j < 4; j++) {
                    simdgroup_multiply_accumulate(acc[i][j], a[i], b[j], acc[i][j]);
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    device float *s0 = s + (ulong(g) * p.rows_pad + r0) * PREFILL_AG_KEYS + k0;
    for (uint i = 0; i < 4; i++) {
        for (uint j = 0; j < 4; j++) {
            simdgroup_store(acc[i][j], s0 + ulong(sm + 8 * i) * PREFILL_AG_KEYS + sn + 8 * j, PREFILL_AG_KEYS);
        }
    }
}

// One simdgroup per row r of kv head tg.y: the online softmax step for the
// block. P gets exp(scale S - m) for the keys the row sees and 0 for the
// rest up to prefill_ag_tile_keys, which prefill_ag_pv reads.
kernel void prefill_ag_softmax(device const float *s [[buffer(0)]],
                               device half *pm [[buffer(1)]],
                               device float *m [[buffer(2)]],
                               device float *l [[buffer(3)]],
                               device float *alpha [[buffer(4)]],
                               constant AttGemmParams &p [[buffer(5)]],
                               uint2 tg [[threadgroup_position_in_grid]],
                               uint simd [[simdgroup_index_in_threadgroup]],
                               uint lane [[thread_index_in_simdgroup]]) {
    uint g = tg.y, r = tg.x * 8 + simd;
    if (r >= p.rows) {
        return;
    }
    ulong row = ulong(g) * p.rows_pad + r;
    device const float *sr = s + row * PREFILL_AG_KEYS;
    device half *pr = pm + row * PREFILL_AG_KEYS;
    int limit = prefill_ag_limit(p, r);
    uint count = limit < 0 ? 0 : min(p.n, uint(limit) + 1);
    uint padded = prefill_ag_tile_keys(p, r);
    bool first = p.j0 == 0;
    float m_old = first ? -INFINITY : m[row];
    float l_old = first ? 0.0f : l[row];

    float mx = -INFINITY;
    for (uint j = lane; j < count; j += 32) {
        mx = max(mx, sr[j] * p.scale);
    }
    mx = simd_max(mx);
    if (count == 0) {
        for (uint j = lane; j < padded; j += 32) {
            pr[j] = half(0);
        }
        if (lane == 0) {
            alpha[row] = 1.0f;
        }
        return;
    }
    float m_new = max(m_old, mx);
    float sum = 0.0f;
    for (uint j = lane; j < padded; j += 32) {
        float e = j < count ? exp(sr[j] * p.scale - m_new) : 0.0f;
        sum += e;
        pr[j] = half(e);
    }
    sum = simd_sum(sum);
    if (lane == 0) {
        float a = first ? 0.0f : exp(m_old - m_new);
        m[row] = m_new;
        l[row] = l_old * a + sum;
        alpha[row] = a;
    }
}

// O[g][r][d] = alpha[r] O + P V for rows tg.y * 64 .., dims tg.x * 64 ..
// alpha 0 (the first block) ignores the old O, which is not set yet.
template <uint BITS>
kernel void prefill_ag_pv(device const half *pm [[buffer(0)]],
                          device const half *values [[buffer(1)]],
                          device float *o [[buffer(2)]],
                          device const float *alpha [[buffer(3)]],
                          constant AttGemmParams &p [[buffer(4)]],
                          device const uint *codes [[buffer(5)]],
                          device const half *scales [[buffer(6)]],
                          device const half *biases [[buffer(7)]],
                          uint3 tg [[threadgroup_position_in_grid]],
                          uint tid [[thread_index_in_threadgroup]],
                          uint simd [[simdgroup_index_in_threadgroup]]) {
    // The P and value tiles while the product runs, then the 64 x 64 sums
    // for the update: one 16 KB buffer, not 26 KB (5 + 4.5 + 16).
    threadgroup float c[64 * 64];
    threadgroup half *pt = (threadgroup half *)c;
    threadgroup half *vt = pt + 64 * PREFILL_AG_LD;
    uint g = tg.z, r0 = tg.y * 64, d0 = tg.x * 64;
    if (r0 >= p.rows) {
        return;
    }
    uint padded = prefill_ag_tile_keys(p, r0);
    if (padded == 0) {
        return;  // no row sees the block: alpha is 1 and P is 0
    }

    // Thread t loads 16 values of P row t / 2 and 16 dims of value row t / 4.
    uint pr = tid / 2, pc = (tid % 2) * 16;
    device const half *pp = pm + (ulong(g) * p.rows_pad + r0 + pr) * PREFILL_AG_KEYS + pc;
    threadgroup half4 *pd = (threadgroup half4 *)(pt + pr * PREFILL_AG_LD + pc);
    uint vr = tid / 4, vc = (tid % 4) * 16;
    threadgroup half4 *vd = (threadgroup half4 *)(vt + vr * PREFILL_AG_VLD + vc);

    uint sm = (simd / 2) * 32, sn = (simd % 2) * 32;
    simdgroup_float8x8 acc[4][4];
    for (uint i = 0; i < 4; i++) {
        for (uint j = 0; j < 4; j++) {
            acc[i][j] = simdgroup_float8x8(0);
        }
    }
    for (uint k0 = 0; k0 < padded; k0 += PREFILL_AG_BK) {
        for (uint i = 0; i < 4; i++) {
            pd[i] = ((device const half4 *)(pp + k0))[i];
        }
        bool v_ok = k0 + vr < p.n;
        for (uint i = 0; i < 4; i++) {
            vd[i] = v_ok ? prefill_ag_kv4<BITS>(values, codes, scales, biases, p, g, p.j0 + k0 + vr,
                                                d0 + vc + 4 * i)
                         : half4(0);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
#pragma clang loop unroll(full)
        for (uint kk = 0; kk < PREFILL_AG_BK; kk += 8) {
            simdgroup_half8x8 a[4];
            simdgroup_half8x8 b[4];
            for (uint i = 0; i < 4; i++) {
                simdgroup_load(a[i], pt + (sm + 8 * i) * PREFILL_AG_LD + kk, PREFILL_AG_LD);
                simdgroup_load(b[i], vt + kk * PREFILL_AG_VLD + sn + 8 * i, PREFILL_AG_VLD);
            }
            for (uint i = 0; i < 4; i++) {
                for (uint j = 0; j < 4; j++) {
                    simdgroup_multiply_accumulate(acc[i][j], a[i], b[j], acc[i][j]);
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    // The loop ended on a barrier, so the tiles are free for the sums.
    for (uint i = 0; i < 4; i++) {
        for (uint j = 0; j < 4; j++) {
            simdgroup_store(acc[i][j], c + (sm + 8 * i) * 64 + sn + 8 * j, 64);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Thread t writes 32 dims of row t / 2.
    uint r = r0 + tid / 2, cc = (tid % 2) * 32;
    if (r < p.rows) {
        ulong row = ulong(g) * p.rows_pad + r;
        float a = alpha[row];
        device float *orow = o + row * PREFILL_AG_D + d0 + cc;
        threadgroup const float *crow = c + (tid / 2) * 64 + cc;
        for (uint j = 0; j < 32; j++) {
            orow[j] = a == 0.0f ? crow[j] : a * orow[j] + crow[j];
        }
    }
}

template [[host_name("prefill_ag_scores")]] kernel decltype(prefill_ag_scores<16>) prefill_ag_scores<16>;
template [[host_name("prefill_ag_scores_q8")]] kernel decltype(prefill_ag_scores<8>) prefill_ag_scores<8>;
template [[host_name("prefill_ag_scores_q4")]] kernel decltype(prefill_ag_scores<4>) prefill_ag_scores<4>;
template [[host_name("prefill_ag_pv")]] kernel decltype(prefill_ag_pv<16>) prefill_ag_pv<16>;
template [[host_name("prefill_ag_pv_q8")]] kernel decltype(prefill_ag_pv<8>) prefill_ag_pv<8>;
template [[host_name("prefill_ag_pv_q4")]] kernel decltype(prefill_ag_pv<4>) prefill_ag_pv<4>;

// y[token][head][d] = O / l * sigmoid(gate), thread id = (d, r, g).
kernel void prefill_ag_final(device const float *o [[buffer(0)]],
                             device const float *l [[buffer(1)]],
                             device const float *q_and_gate [[buffer(2)]],
                             device float *y [[buffer(3)]],
                             constant AttGemmParams &p [[buffer(4)]],
                             uint3 id [[thread_position_in_grid]]) {
    uint d = id.x, r = id.y, g = id.z;
    if (r >= p.rows) {
        return;
    }
    ulong row = ulong(g) * p.rows_pad + r;
    ulong at = prefill_ag_q_row(p, g, r);
    float gate = q_and_gate[2 * at + PREFILL_AG_D + d];
    y[at + d] = o[row * PREFILL_AG_D + d] / l[row] * sigmoid(gate);
}

// The fused variant (SAETTA_AG_FUSED=1): no softmax pass. The scores kernel
// writes sh = scale S - t in half, with t the row's max over the keys it
// sees in the 64-key tile (-inf for the keys it does not see), and t to
// tmax[g][r][tile]. pv takes the row's new max m from m_in and the tile
// maxes, loads P = exp(sh + t - m) into its tile, and the dim-0 threadgroup
// writes m and l to m_out and l_out. Each thread loads, scales and writes
// back the same row, so the row state stays in registers.
constant uint PREFILL_AG_TILES = PREFILL_AG_KEYS / 64;

template <uint BITS>
kernel void prefill_ag_scores_f(device const half *q [[buffer(0)]],
                                device const half *keys [[buffer(1)]],
                                device half *sh [[buffer(2)]],
                                constant AttGemmParams &p [[buffer(3)]],
                                device const uint *codes [[buffer(4)]],
                                device const half *scales [[buffer(5)]],
                                device const half *biases [[buffer(6)]],
                                device float *tmax [[buffer(7)]],
                                uint3 tg [[threadgroup_position_in_grid]],
                                uint tid [[thread_index_in_threadgroup]],
                                uint simd [[simdgroup_index_in_threadgroup]]) {
    // The q and key tiles while the product runs, then 32 rows of scores
    // at a time (8 KB), so the kernel needs no more threadgroup memory
    // than prefill_ag_scores.
    threadgroup half qt[2 * 64 * PREFILL_AG_LD];
    threadgroup half *kt = qt + 64 * PREFILL_AG_LD;
    threadgroup float *c = (threadgroup float *)qt;
    uint g = tg.z, r0 = tg.y * 64, k0 = tg.x * 64;
    uint last_row = min(r0 + 63, p.rows - 1);
    if (int(k0) > prefill_ag_limit(p, last_row)) {
        return;
    }

    uint lr = tid / 2, lc = (tid % 2) * 16;
    bool q_ok = r0 + lr < p.rows;
    bool k_ok = k0 + lr < p.n;
    device const half *qp = q + (q_ok ? prefill_ag_q_row(p, g, r0 + lr) : 0) + lc;
    uint key = p.j0 + k0 + lr;
    threadgroup half4 *qd = (threadgroup half4 *)(qt + lr * PREFILL_AG_LD + lc);
    threadgroup half4 *kd = (threadgroup half4 *)(kt + lr * PREFILL_AG_LD + lc);

    uint sm = (simd / 2) * 32, sn = (simd % 2) * 32;
    simdgroup_float8x8 acc[4][4];
    for (uint i = 0; i < 4; i++) {
        for (uint j = 0; j < 4; j++) {
            acc[i][j] = simdgroup_float8x8(0);
        }
    }
    for (uint d0 = 0; d0 < PREFILL_AG_D; d0 += PREFILL_AG_BK) {
        for (uint i = 0; i < 4; i++) {
            qd[i] = q_ok ? ((device const half4 *)(qp + d0))[i] : half4(0);
            kd[i] = k_ok ? prefill_ag_kv4<BITS>(keys, codes, scales, biases, p, g, key, d0 + lc + 4 * i)
                         : half4(0);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
#pragma clang loop unroll(full)
        for (uint kk = 0; kk < PREFILL_AG_BK; kk += 8) {
            simdgroup_half8x8 a[4];
            simdgroup_half8x8 b[4];
            for (uint i = 0; i < 4; i++) {
                simdgroup_load(a[i], qt + (sm + 8 * i) * PREFILL_AG_LD + kk, PREFILL_AG_LD);
                simdgroup_load(b[i], kt + (sn + 8 * i) * PREFILL_AG_LD + kk, PREFILL_AG_LD, ulong2(0, 0), true);
            }
            for (uint i = 0; i < 4; i++) {
                for (uint j = 0; j < 4; j++) {
                    simdgroup_multiply_accumulate(acc[i][j], a[i], b[j], acc[i][j]);
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    // Simdgroups 0 and 1 hold rows 0 .. 31, 2 and 3 rows 32 .. 63. Thread t
    // then takes 16 keys of row t / 4; the four share the row max.
    uint qr = tid / 4, cc = (tid % 4) * 16;
    for (uint half_ = 0; half_ < 2; half_++) {
        if (simd / 2 == half_) {
            for (uint i = 0; i < 4; i++) {
                for (uint j = 0; j < 4; j++) {
                    simdgroup_store(acc[i][j], c + (8 * i) * 64 + sn + 8 * j, 64);
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        uint r = r0 + half_ * 32 + qr;
        int limit = prefill_ag_limit(p, r) - int(k0);
        int seen = min(limit + 1, int(p.n) - int(k0));
        threadgroup const float *crow = c + qr * 64 + cc;
        float mx = -INFINITY;
        for (uint j = 0; j < 16; j++) {
            if (int(cc + j) < seen) {
                mx = max(mx, crow[j] * p.scale);
            }
        }
        mx = max(mx, simd_shuffle_xor(mx, 1));
        mx = max(mx, simd_shuffle_xor(mx, 2));
        ulong row = ulong(g) * p.rows_pad + r;
        if (tid % 4 == 0) {
            tmax[row * PREFILL_AG_TILES + tg.x] = mx;
        }
        device half4 *out = (device half4 *)(sh + row * PREFILL_AG_KEYS + k0 + cc);
        for (uint j = 0; j < 16; j += 4) {
            half4 v;
            for (uint i = 0; i < 4; i++) {
                v[i] = int(cc + j + i) < seen ? half(crow[j + i] * p.scale - mx) : half(-INFINITY);
            }
            out[j / 4] = v;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

template <uint BITS>
kernel void prefill_ag_pv_f(device const half *sh [[buffer(0)]],
                            device const half *values [[buffer(1)]],
                            device float *o [[buffer(2)]],
                            device const float *tmax [[buffer(3)]],
                            constant AttGemmParams &p [[buffer(4)]],
                            device const uint *codes [[buffer(5)]],
                            device const half *scales [[buffer(6)]],
                            device const half *biases [[buffer(7)]],
                            device const float *m_in [[buffer(8)]],
                            device float *m_out [[buffer(9)]],
                            device const float *l_in [[buffer(10)]],
                            device float *l_out [[buffer(11)]],
                            uint3 tg [[threadgroup_position_in_grid]],
                            uint tid [[thread_index_in_threadgroup]],
                            uint simd [[simdgroup_index_in_threadgroup]]) {
    threadgroup float c[64 * 64];
    threadgroup half *pt = (threadgroup half *)c;
    threadgroup half *vt = pt + 64 * PREFILL_AG_LD;
    uint g = tg.z, r0 = tg.y * 64, d0 = tg.x * 64;
    if (r0 >= p.rows) {
        return;
    }
    uint padded = prefill_ag_tile_keys(p, r0);

    // Thread t loads P row t / 2 and later updates the same row of O.
    uint pr = tid / 2, pc = (tid % 2) * 16;
    uint r = r0 + pr;
    ulong row = ulong(g) * p.rows_pad + r;
    bool first = p.j0 == 0;
    float m_old = first ? -INFINITY : m_in[row];
    float mx = -INFINITY;
    for (uint t = 0; t * 64 < padded; t++) {
        mx = max(mx, tmax[row * PREFILL_AG_TILES + t]);
    }
    float m_new = max(m_old, mx);
    // A row that has seen no key yet (only padding rows) keeps 0s.
    float m_sub = m_new == -INFINITY ? 0.0f : m_new;
    float a = first ? 0.0f : (m_new == -INFINITY ? 1.0f : exp(m_old - m_new));

    if (padded == 0) {
        // No row of the tile sees the block: O, m and l stay.
        if (tg.x == 0 && tid % 2 == 0 && r < p.rows) {
            m_out[row] = m_old;
            l_out[row] = l_in[row];
        }
        return;
    }

    device const half *pp = sh + row * PREFILL_AG_KEYS + pc;
    threadgroup half4 *pd = (threadgroup half4 *)(pt + pr * PREFILL_AG_LD + pc);
    uint vr = tid / 4, vc = (tid % 4) * 16;
    threadgroup half4 *vd = (threadgroup half4 *)(vt + vr * PREFILL_AG_VLD + vc);

    uint sm = (simd / 2) * 32, sn = (simd % 2) * 32;
    simdgroup_float8x8 acc[4][4];
    for (uint i = 0; i < 4; i++) {
        for (uint j = 0; j < 4; j++) {
            acc[i][j] = simdgroup_float8x8(0);
        }
    }
    float sum = 0.0f;
    for (uint k0 = 0; k0 < padded; k0 += PREFILL_AG_BK) {
        float off = tmax[row * PREFILL_AG_TILES + k0 / 64] - m_sub;
        for (uint i = 0; i < 4; i++) {
            float4 e = exp(float4(((device const half4 *)(pp + k0))[i]) + off);
            sum += e.x + e.y + e.z + e.w;
            pd[i] = half4(e);
        }
        bool v_ok = k0 + vr < p.n;
        for (uint i = 0; i < 4; i++) {
            vd[i] = v_ok ? prefill_ag_kv4<BITS>(values, codes, scales, biases, p, g, p.j0 + k0 + vr,
                                                d0 + vc + 4 * i)
                         : half4(0);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
#pragma clang loop unroll(full)
        for (uint kk = 0; kk < PREFILL_AG_BK; kk += 8) {
            simdgroup_half8x8 a8[4];
            simdgroup_half8x8 b8[4];
            for (uint i = 0; i < 4; i++) {
                simdgroup_load(a8[i], pt + (sm + 8 * i) * PREFILL_AG_LD + kk, PREFILL_AG_LD);
                simdgroup_load(b8[i], vt + kk * PREFILL_AG_VLD + sn + 8 * i, PREFILL_AG_VLD);
            }
            for (uint i = 0; i < 4; i++) {
                for (uint j = 0; j < 4; j++) {
                    simdgroup_multiply_accumulate(acc[i][j], a8[i], b8[j], acc[i][j]);
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    sum += simd_shuffle_xor(sum, 1);
    for (uint i = 0; i < 4; i++) {
        for (uint j = 0; j < 4; j++) {
            simdgroup_store(acc[i][j], c + (sm + 8 * i) * 64 + sn + 8 * j, 64);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (r < p.rows) {
        if (tg.x == 0 && tid % 2 == 0) {
            m_out[row] = m_new;
            l_out[row] = (first ? 0.0f : l_in[row] * a) + sum;
        }
        uint cc = (tid % 2) * 32;
        device float *orow = o + row * PREFILL_AG_D + d0 + cc;
        threadgroup const float *crow = c + pr * 64 + cc;
        for (uint j = 0; j < 32; j++) {
            orow[j] = first ? crow[j] : a * orow[j] + crow[j];
        }
    }
}

template <uint BITS>
kernel void prefill_ag_scores_t(device const half *q [[buffer(0)]],
                                device const half *keys [[buffer(1)]],
                                device half *sh [[buffer(2)]],
                                constant AttGemmParams &p [[buffer(3)]],
                                device const uint *codes [[buffer(4)]],
                                device const half *scales [[buffer(5)]],
                                device const half *biases [[buffer(6)]],
                                device float *tmax [[buffer(7)]],
                                device float *tsum [[buffer(8)]],
                                uint3 tg [[threadgroup_position_in_grid]],
                                uint tid [[thread_index_in_threadgroup]],
                                uint simd [[simdgroup_index_in_threadgroup]]) {
    // The q and key tiles while the product runs, then 32 rows of scores
    // at a time (8 KB), so the kernel needs no more threadgroup memory
    // than prefill_ag_scores.
    threadgroup half qt[2 * 64 * PREFILL_AG_LD];
    threadgroup half *kt = qt + 64 * PREFILL_AG_LD;
    threadgroup float *c = (threadgroup float *)qt;
    uint g = tg.z, r0 = tg.y * 64, k0 = tg.x * 64;
    uint last_row = min(r0 + 63, p.rows - 1);
    if (int(k0) > prefill_ag_limit(p, last_row)) {
        return;
    }

    uint lr = tid / 2, lc = (tid % 2) * 16;
    bool q_ok = r0 + lr < p.rows;
    bool k_ok = k0 + lr < p.n;
    device const half *qp = q + (q_ok ? prefill_ag_q_row(p, g, r0 + lr) : 0) + lc;
    uint key = p.j0 + k0 + lr;
    threadgroup half4 *qd = (threadgroup half4 *)(qt + lr * PREFILL_AG_LD + lc);
    threadgroup half4 *kd = (threadgroup half4 *)(kt + lr * PREFILL_AG_LD + lc);

    uint sm = (simd / 2) * 32, sn = (simd % 2) * 32;
    simdgroup_float8x8 acc[4][4];
    for (uint i = 0; i < 4; i++) {
        for (uint j = 0; j < 4; j++) {
            acc[i][j] = simdgroup_float8x8(0);
        }
    }
    for (uint d0 = 0; d0 < PREFILL_AG_D; d0 += PREFILL_AG_BK) {
        for (uint i = 0; i < 4; i++) {
            qd[i] = q_ok ? ((device const half4 *)(qp + d0))[i] : half4(0);
            kd[i] = k_ok ? prefill_ag_kv4<BITS>(keys, codes, scales, biases, p, g, key, d0 + lc + 4 * i)
                         : half4(0);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
#pragma clang loop unroll(full)
        for (uint kk = 0; kk < PREFILL_AG_BK; kk += 8) {
            simdgroup_half8x8 a[4];
            simdgroup_half8x8 b[4];
            for (uint i = 0; i < 4; i++) {
                simdgroup_load(a[i], qt + (sm + 8 * i) * PREFILL_AG_LD + kk, PREFILL_AG_LD);
                simdgroup_load(b[i], kt + (sn + 8 * i) * PREFILL_AG_LD + kk, PREFILL_AG_LD, ulong2(0, 0), true);
            }
            for (uint i = 0; i < 4; i++) {
                for (uint j = 0; j < 4; j++) {
                    simdgroup_multiply_accumulate(acc[i][j], a[i], b[j], acc[i][j]);
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    // Simdgroups 0 and 1 hold rows 0 .. 31, 2 and 3 rows 32 .. 63. Thread t
    // then takes 16 keys of row t / 4; the four share the row max.
    uint qr = tid / 4, cc = (tid % 4) * 16;
    for (uint half_ = 0; half_ < 2; half_++) {
        if (simd / 2 == half_) {
            for (uint i = 0; i < 4; i++) {
                for (uint j = 0; j < 4; j++) {
                    simdgroup_store(acc[i][j], c + (8 * i) * 64 + sn + 8 * j, 64);
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        uint r = r0 + half_ * 32 + qr;
        int limit = prefill_ag_limit(p, r) - int(k0);
        int seen = min(limit + 1, int(p.n) - int(k0));
        threadgroup const float *crow = c + qr * 64 + cc;
        float mx = -INFINITY;
        for (uint j = 0; j < 16; j++) {
            if (int(cc + j) < seen) {
                mx = max(mx, crow[j] * p.scale);
            }
        }
        mx = max(mx, simd_shuffle_xor(mx, 1));
        mx = max(mx, simd_shuffle_xor(mx, 2));
        ulong row = ulong(g) * p.rows_pad + r;
        device half4 *out = (device half4 *)(sh + row * PREFILL_AG_KEYS + k0 + cc);
        float sum = 0.0f;
        for (uint j = 0; j < 16; j += 4) {
            half4 v;
            for (uint i = 0; i < 4; i++) {
                float e = int(cc + j + i) < seen ? exp(crow[j + i] * p.scale - mx) : 0.0f;
                sum += e;
                v[i] = half(e);
            }
            out[j / 4] = v;
        }
        sum += simd_shuffle_xor(sum, 1);
        sum += simd_shuffle_xor(sum, 2);
        if (tid % 4 == 0) {
            tmax[row * PREFILL_AG_TILES + tg.x] = mx;
            tsum[row * PREFILL_AG_TILES + tg.x] = sum;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

template <uint BITS>
kernel void prefill_ag_pv_t(device const half *sh [[buffer(0)]],
                            device const half *values [[buffer(1)]],
                            device float *o [[buffer(2)]],
                            device const float *tmax [[buffer(3)]],
                            constant AttGemmParams &p [[buffer(4)]],
                            device const uint *codes [[buffer(5)]],
                            device const half *scales [[buffer(6)]],
                            device const half *biases [[buffer(7)]],
                            device const float *m_in [[buffer(8)]],
                            device float *m_out [[buffer(9)]],
                            device const float *l_in [[buffer(10)]],
                            device float *l_out [[buffer(11)]],
                            device const float *tsum [[buffer(12)]],
                            uint3 tg [[threadgroup_position_in_grid]],
                            uint tid [[thread_index_in_threadgroup]],
                            uint simd [[simdgroup_index_in_threadgroup]]) {
    threadgroup float c[64 * 64];
    threadgroup half *pt = (threadgroup half *)c;
    threadgroup half *vt = pt + 64 * PREFILL_AG_LD;
    uint g = tg.z, r0 = tg.y * 64, d0 = tg.x * 64;
    if (r0 >= p.rows) {
        return;
    }
    uint padded = prefill_ag_tile_keys(p, r0);

    // Thread t loads P row t / 2 and later updates the same row of O.
    uint pr = tid / 2, pc = (tid % 2) * 16;
    uint r = r0 + pr;
    ulong row = ulong(g) * p.rows_pad + r;
    bool first = p.j0 == 0;
    float m_old = first ? -INFINITY : m_in[row];
    float mx = -INFINITY;
    for (uint t = 0; t * 64 < padded; t++) {
        mx = max(mx, tmax[row * PREFILL_AG_TILES + t]);
    }
    float m_new = max(m_old, mx);
    // A row that has seen no key yet (only padding rows) keeps 0s.
    float m_sub = m_new == -INFINITY ? 0.0f : m_new;
    float a = first ? 0.0f : (m_new == -INFINITY ? 1.0f : exp(m_old - m_new));
    float sum = 0.0f;
    for (uint t = 0; t * 64 < padded; t++) {
        sum += exp(tmax[row * PREFILL_AG_TILES + t] - m_sub) * tsum[row * PREFILL_AG_TILES + t];
    }

    if (padded == 0) {
        // No row of the tile sees the block: O, m and l stay.
        if (tg.x == 0 && tid % 2 == 0 && r < p.rows) {
            m_out[row] = m_old;
            l_out[row] = l_in[row];
        }
        return;
    }

    device const half *pp = sh + row * PREFILL_AG_KEYS + pc;
    threadgroup half4 *pd = (threadgroup half4 *)(pt + pr * PREFILL_AG_LD + pc);
    uint vr = tid / 4, vc = (tid % 4) * 16;
    threadgroup half4 *vd = (threadgroup half4 *)(vt + vr * PREFILL_AG_VLD + vc);

    uint sm = (simd / 2) * 32, sn = (simd % 2) * 32;
    simdgroup_float8x8 acc[4][4];
    for (uint i = 0; i < 4; i++) {
        for (uint j = 0; j < 4; j++) {
            acc[i][j] = simdgroup_float8x8(0);
        }
    }
    for (uint k0 = 0; k0 < padded; k0 += PREFILL_AG_BK) {
        half f = half(exp(tmax[row * PREFILL_AG_TILES + k0 / 64] - m_sub));
        for (uint i = 0; i < 4; i++) {
            pd[i] = ((device const half4 *)(pp + k0))[i] * f;
        }
        bool v_ok = k0 + vr < p.n;
        for (uint i = 0; i < 4; i++) {
            vd[i] = v_ok ? prefill_ag_kv4<BITS>(values, codes, scales, biases, p, g, p.j0 + k0 + vr,
                                                d0 + vc + 4 * i)
                         : half4(0);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
#pragma clang loop unroll(full)
        for (uint kk = 0; kk < PREFILL_AG_BK; kk += 8) {
            simdgroup_half8x8 a8[4];
            simdgroup_half8x8 b8[4];
            for (uint i = 0; i < 4; i++) {
                simdgroup_load(a8[i], pt + (sm + 8 * i) * PREFILL_AG_LD + kk, PREFILL_AG_LD);
                simdgroup_load(b8[i], vt + kk * PREFILL_AG_VLD + sn + 8 * i, PREFILL_AG_VLD);
            }
            for (uint i = 0; i < 4; i++) {
                for (uint j = 0; j < 4; j++) {
                    simdgroup_multiply_accumulate(acc[i][j], a8[i], b8[j], acc[i][j]);
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for (uint i = 0; i < 4; i++) {
        for (uint j = 0; j < 4; j++) {
            simdgroup_store(acc[i][j], c + (sm + 8 * i) * 64 + sn + 8 * j, 64);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (r < p.rows) {
        if (tg.x == 0 && tid % 2 == 0) {
            m_out[row] = m_new;
            l_out[row] = (first ? 0.0f : l_in[row] * a) + sum;
        }
        uint cc = (tid % 2) * 32;
        device float *orow = o + row * PREFILL_AG_D + d0 + cc;
        threadgroup const float *crow = c + pr * 64 + cc;
        for (uint j = 0; j < 32; j++) {
            orow[j] = first ? crow[j] : a * orow[j] + crow[j];
        }
    }
}

template [[host_name("prefill_ag_scores_f")]] kernel decltype(prefill_ag_scores_f<16>) prefill_ag_scores_f<16>;
template [[host_name("prefill_ag_scores_f_q8")]] kernel decltype(prefill_ag_scores_f<8>) prefill_ag_scores_f<8>;
template [[host_name("prefill_ag_scores_f_q4")]] kernel decltype(prefill_ag_scores_f<4>) prefill_ag_scores_f<4>;
template [[host_name("prefill_ag_pv_f")]] kernel decltype(prefill_ag_pv_f<16>) prefill_ag_pv_f<16>;
template [[host_name("prefill_ag_pv_f_q8")]] kernel decltype(prefill_ag_pv_f<8>) prefill_ag_pv_f<8>;
template [[host_name("prefill_ag_pv_f_q4")]] kernel decltype(prefill_ag_pv_f<4>) prefill_ag_pv_f<4>;
template [[host_name("prefill_ag_scores_t")]] kernel decltype(prefill_ag_scores_t<16>) prefill_ag_scores_t<16>;
template [[host_name("prefill_ag_scores_t_q8")]] kernel decltype(prefill_ag_scores_t<8>) prefill_ag_scores_t<8>;
template [[host_name("prefill_ag_scores_t_q4")]] kernel decltype(prefill_ag_scores_t<4>) prefill_ag_scores_t<4>;
template [[host_name("prefill_ag_pv_t")]] kernel decltype(prefill_ag_pv_t<16>) prefill_ag_pv_t<16>;
template [[host_name("prefill_ag_pv_t_q8")]] kernel decltype(prefill_ag_pv_t<8>) prefill_ag_pv_t<8>;
template [[host_name("prefill_ag_pv_t_q4")]] kernel decltype(prefill_ag_pv_t<4>) prefill_ag_pv_t<4>;

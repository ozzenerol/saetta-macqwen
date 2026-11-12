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
// rest up to n rounded to PREFILL_AG_BK, which prefill_ag_pv reads.
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
    uint padded = (p.n + PREFILL_AG_BK - 1) / PREFILL_AG_BK * PREFILL_AG_BK;
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
    threadgroup half pt[64 * PREFILL_AG_LD];
    threadgroup half vt[PREFILL_AG_BK * PREFILL_AG_VLD];
    threadgroup float c[64 * 64];
    uint g = tg.z, r0 = tg.y * 64, d0 = tg.x * 64;
    if (r0 >= p.rows) {
        return;
    }
    uint last_row = min(r0 + 63, p.rows - 1);
    if (prefill_ag_limit(p, last_row) < 0) {
        return;  // no row sees the block: alpha is 1 and P is 0
    }
    uint padded = (p.n + PREFILL_AG_BK - 1) / PREFILL_AG_BK * PREFILL_AG_BK;

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

// The fused variant (SAETTA_AG_FUSED=1): no f32 scores and no softmax pass.
//   prefill_ag_scores_p  S = Q K^T for a 64 x 64 tile, then per row of the
//                        tile its max mt over the keys the row sees, its
//                        sum lt of exp(scale S - mt), and P = exp(scale S -
//                        mt) in half, 0 for keys the row does not see
//   prefill_ag_pv_f      per row the block's max over its 16 tiles and the
//                        new running max m, l and alpha, then O = alpha O +
//                        sum over tiles of exp(mt - m) P V
// m and l are read from one pair of buffers and written to the other, so
// the four threadgroups of a row (one per 64 dims) all read the old values;
// the host swaps the pair every block.
constant uint PREFILL_AG_TILES = PREFILL_AG_KEYS / 64;  // mt and lt per row and block

template <uint BITS>
kernel void prefill_ag_scores_p(device const half *q [[buffer(0)]],
                                device const half *keys [[buffer(1)]],
                                device half *pm [[buffer(2)]],
                                constant AttGemmParams &p [[buffer(3)]],
                                device const uint *codes [[buffer(4)]],
                                device const half *scales [[buffer(5)]],
                                device const half *biases [[buffer(6)]],
                                device float *mt [[buffer(7)]],
                                device float *lt [[buffer(8)]],
                                uint3 tg [[threadgroup_position_in_grid]],
                                uint tid [[thread_index_in_threadgroup]],
                                uint simd [[simdgroup_index_in_threadgroup]]) {
    // The q and key tiles while the product runs, then the 64 x 64 scores.
    threadgroup float buf[64 * 64];
    threadgroup half *qt = (threadgroup half *)buf;
    threadgroup half *kt = qt + 64 * PREFILL_AG_LD;
    uint g = tg.z, r0 = tg.y * 64, k0 = tg.x * 64;
    uint tile = tg.x;
    // Thread t takes row t / 2 of the tile, keys 32 (t % 2) .. + 31.
    uint row_t = r0 + tid / 2, half_t = (tid % 2) * 32;
    ulong row = ulong(g) * p.rows_pad + row_t;
    uint last_row = min(r0 + 63, p.rows - 1);
    if (int(k0) > prefill_ag_limit(p, last_row)) {
        // No row sees the tile: pv_f reads mt = -inf and drops P.
        if (half_t == 0 && row_t < p.rows) {
            mt[row * PREFILL_AG_TILES + tile] = -INFINITY;
            lt[row * PREFILL_AG_TILES + tile] = 0.0f;
        }
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
    // The loop ended on a barrier, so buf is free for the scores.
    for (uint i = 0; i < 4; i++) {
        for (uint j = 0; j < 4; j++) {
            simdgroup_store(acc[i][j], buf + (sm + 8 * i) * 64 + sn + 8 * j, 64);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Keys k0 + half_t + c the row sees: c < count.
    int limit = prefill_ag_limit(p, row_t);
    int seen = min(int(p.n), limit + 1) - int(k0 + half_t);
    uint count = row_t < p.rows ? uint(clamp(seen, 0, 32)) : 0;
    threadgroup const float *sr = buf + (tid / 2) * 64 + half_t;
    float mx = -INFINITY;
    for (uint c = 0; c < count; c++) {
        mx = max(mx, sr[c] * p.scale);
    }
    // Threads t and t ^ 1 share a row and are neighbours in a simdgroup.
    mx = max(mx, simd_shuffle_xor(mx, 1));
    float sum = 0.0f;
    device half4 *pr = (device half4 *)(pm + row * PREFILL_AG_KEYS + k0 + half_t);
    for (uint c4 = 0; c4 < 8; c4++) {
        float4 e;
        for (uint k = 0; k < 4; k++) {
            uint c = 4 * c4 + k;
            e[k] = c < count ? exp(sr[c] * p.scale - mx) : 0.0f;
        }
        sum += e.x + e.y + e.z + e.w;
        if (row_t < p.rows) {
            pr[c4] = half4(e);
        }
    }
    sum += simd_shuffle_xor(sum, 1);
    if (half_t == 0 && row_t < p.rows) {
        mt[row * PREFILL_AG_TILES + tile] = mx;
        lt[row * PREFILL_AG_TILES + tile] = sum;
    }
}

template <uint BITS>
kernel void prefill_ag_pv_f(device const half *pm [[buffer(0)]],
                            device const half *values [[buffer(1)]],
                            device float *o [[buffer(2)]],
                            device const float *mt [[buffer(3)]],
                            constant AttGemmParams &p [[buffer(4)]],
                            device const uint *codes [[buffer(5)]],
                            device const half *scales [[buffer(6)]],
                            device const half *biases [[buffer(7)]],
                            device const float *lt [[buffer(8)]],
                            device const float *m_in [[buffer(9)]],
                            device const float *l_in [[buffer(10)]],
                            device float *m_out [[buffer(11)]],
                            device float *l_out [[buffer(12)]],
                            uint3 tg [[threadgroup_position_in_grid]],
                            uint tid [[thread_index_in_threadgroup]],
                            uint simd [[simdgroup_index_in_threadgroup]]) {
    threadgroup half pt[64 * PREFILL_AG_LD];
    threadgroup half vt[PREFILL_AG_BK * PREFILL_AG_VLD];
    threadgroup float c[64 * 64];
    // exp(mt - m) per row and tile, 0 for a tile the row does not see.
    threadgroup float factor[64][PREFILL_AG_TILES];
    threadgroup float alpha_t[64];
    uint g = tg.z, r0 = tg.y * 64, d0 = tg.x * 64;
    if (r0 >= p.rows) {
        return;
    }
    uint padded = (p.n + PREFILL_AG_BK - 1) / PREFILL_AG_BK * PREFILL_AG_BK;
    uint tiles = (p.n + 63) / 64;
    bool first = p.j0 == 0;

    // The softmax state: thread t < 64 takes row r0 + t.
    if (tid < 64) {
        uint r = r0 + tid;
        ulong row = ulong(g) * p.rows_pad + r;
        bool sees = r < p.rows && prefill_ag_limit(p, r) >= 0;
        float m_old = first ? -INFINITY : m_in[row];
        float l_old = first ? 0.0f : l_in[row];
        float mb = -INFINITY;
        if (sees) {
            for (uint t = 0; t < tiles; t++) {
                mb = max(mb, mt[row * PREFILL_AG_TILES + t]);
            }
        }
        float m_new = max(m_old, mb);
        float lb = 0.0f;
        for (uint t = 0; t < PREFILL_AG_TILES; t++) {
            float mtt = sees && t < tiles ? mt[row * PREFILL_AG_TILES + t] : -INFINITY;
            float f = mtt == -INFINITY ? 0.0f : exp(mtt - m_new);
            factor[tid][t] = f;
            if (f != 0.0f) {
                lb += f * lt[row * PREFILL_AG_TILES + t];
            }
        }
        // alpha 0 for the first block, where O is not set yet; 1 for a row
        // that sees none of this block.
        float a = first ? 0.0f : (m_old == m_new ? 1.0f : exp(m_old - m_new));
        alpha_t[tid] = a;
        if (tg.x == 0 && r < p.rows) {
            m_out[row] = m_new;
            l_out[row] = l_old * a + lb;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint last_row = min(r0 + 63, p.rows - 1);
    if (prefill_ag_limit(p, last_row) < 0) {
        return;  // no row sees the block: alpha is 1 and O stays
    }

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
        // The 16 keys of this thread lie in one 64-key tile. A factor of 0
        // also drops P a scores_p tile never wrote.
        float f = factor[pr][(k0 + pc) / 64];
        for (uint i = 0; i < 4; i++) {
            pd[i] = f == 0.0f ? half4(0) : half4(float4(((device const half4 *)(pp + k0))[i]) * f);
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
    for (uint i = 0; i < 4; i++) {
        for (uint j = 0; j < 4; j++) {
            simdgroup_store(acc[i][j], c + (sm + 8 * i) * 64 + sn + 8 * j, 64);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint r = r0 + tid / 2, cc = (tid % 2) * 32;
    if (r < p.rows) {
        ulong row = ulong(g) * p.rows_pad + r;
        float a = alpha_t[tid / 2];
        device float *orow = o + row * PREFILL_AG_D + d0 + cc;
        threadgroup const float *crow = c + (tid / 2) * 64 + cc;
        for (uint j = 0; j < 32; j++) {
            orow[j] = a == 0.0f ? crow[j] : a * orow[j] + crow[j];
        }
    }
}

template [[host_name("prefill_ag_scores_p")]] kernel decltype(prefill_ag_scores_p<16>) prefill_ag_scores_p<16>;
template [[host_name("prefill_ag_scores_p_q8")]] kernel decltype(prefill_ag_scores_p<8>) prefill_ag_scores_p<8>;
template [[host_name("prefill_ag_scores_p_q4")]] kernel decltype(prefill_ag_scores_p<4>) prefill_ag_scores_p<4>;
template [[host_name("prefill_ag_pv_f")]] kernel decltype(prefill_ag_pv_f<16>) prefill_ag_pv_f<16>;
template [[host_name("prefill_ag_pv_f_q8")]] kernel decltype(prefill_ag_pv_f<8>) prefill_ag_pv_f<8>;
template [[host_name("prefill_ag_pv_f_q4")]] kernel decltype(prefill_ag_pv_f<4>) prefill_ag_pv_f<4>;

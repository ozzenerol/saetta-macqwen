// Split-K decode attention for m rows at once (the speculative verify),
// with the splits of attention.metal: threadgroup (s, kv) takes the chunks
// s, s + splits, ... of kv head kv and keeps an online softmax per row, so
// each chunk's keys and values are read once for up to ROWS_PASS rows.
// Scores S^T = K Q^T and O = P V in 8x8 simdgroup matrices; more rows take
// more passes. Row r is at position pos + r and sees kv_len = pos + r + 1
// tokens. part_o and part_ml hold one result per (row, split, q head), and
// attention_reduce_rows merges each row.
//
// BITS 16 reads a half cache from buffers 1 and 2 (3 to 6 are bound but not
// read); 8 and 4 read a quantized cache in the layout of kv_quant.metal.
// Either has room for whole chunks (GpuKvCache rounds it up), so a chunk
// never reaches past it.

constant uint ROWS_PASS = 2;
// Row stride of the score tile: 128 + 4 floats, so the 8 rows of an 8x8
// tile load fall in different banks.
constant uint SC_STRIDE = DEC_CHUNK + 4;

struct RowsAttentionParams {
    uint heads;
    uint kv_heads;
    uint capacity;
    float scale;
    uint rows;    // m
    uint splits;  // threadgroups per kv head, results per row and q head
};

// One kv head of the keys or the values.
struct KvHead {
    device const uint *data;
    device const half *scales;
    device const half *biases;
    uint kv_heads;
    uint kv;
};

// Scale and bias of the group of `token` that holds `dim`; 1 and 0 for a
// half cache.
template <uint BITS>
inline float2 kv_group(KvHead c, uint token, uint dim) {
    if (BITS == 16) {
        return float2(1, 0);
    }
    uint g = (token * c.kv_heads + c.kv) * 4 + dim / 64;
    return float2(float(c.scales[g]), float(c.biases[g]));
}

// This lane's two elements of a tile, dims dim and dim + 1 of `token`, with
// sz from kv_group.
template <uint BITS>
inline void kv_pair(thread simdgroup_float8x8 &t, KvHead c, uint token, uint dim, float2 sz) {
    if (BITS == 16) {
        device const half *h = (device const half *)c.data;
        half2 v = *(device const half2 *)(h + (token * c.kv_heads + c.kv) * DEC_DIM + dim);
        t.thread_elements()[0] = float(v.x);
        t.thread_elements()[1] = float(v.y);
    } else {
        constexpr uint PER_WORD = 32 / BITS;
        constexpr uint MASK = (1u << BITS) - 1;
        uint g = (token * c.kv_heads + c.kv) * 4 + dim / 64;
        uint i = dim % 64;
        uint word = c.data[2 * BITS * g + i / PER_WORD];
        uint shift = (i % PER_WORD) * BITS;
        t.thread_elements()[0] = sz.x * float((word >> shift) & MASK) + sz.y;
        t.thread_elements()[1] = sz.x * float((word >> (shift + BITS)) & MASK) + sz.y;
    }
}

template <uint BITS>
kernel void attention_partial_rows(device const float *q [[buffer(0)]],
                                   device const uint *keys [[buffer(1)]],
                                   device const uint *values [[buffer(2)]],
                                   device const half *key_scales [[buffer(3)]],
                                   device const half *key_biases [[buffer(4)]],
                                   device const half *value_scales [[buffer(5)]],
                                   device const half *value_biases [[buffer(6)]],
                                   device float *part_o [[buffer(7)]],
                                   device float *part_ml [[buffer(8)]],
                                   constant RowsAttentionParams &p [[buffer(9)]],
                                   constant TokenParams &tp [[buffer(10)]],
                                   uint2 tg [[threadgroup_position_in_grid]],
                                   uint simd [[simdgroup_index_in_threadgroup]],
                                   uint lane [[thread_index_in_simdgroup]]) {
    // Scores of the pass's rows: sc[r * 8 + g][t].
    threadgroup float sc[ROWS_PASS * DEC_GROUP][SC_STRIDE];
    threadgroup float alpha[ROWS_PASS * DEC_GROUP];
    uint pos = tp.position;
    uint first = tg.x * DEC_CHUNK;
    if (first >= min(pos + p.rows, p.capacity)) {
        return;
    }
    uint kv = tg.y;
    uint2 f = frag(lane);
    KvHead kc = {keys, key_scales, key_biases, p.kv_heads, kv};
    KvHead vc = {values, value_scales, value_biases, p.kv_heads, kv};

    for (uint r0 = 0; r0 < p.rows; r0 += ROWS_PASS) {
        uint rows = min(ROWS_PASS, p.rows - r0);
        // The pass's last row sees the most tokens.
        uint pass_len = min(pos + r0 + rows, p.capacity);
        float m[ROWS_PASS], l[ROWS_PASS];
        simdgroup_float8x8 o[ROWS_PASS][4];
#pragma clang loop unroll(full)
        for (uint r = 0; r < ROWS_PASS; r++) {
            m[r] = DEC_NO_SCORE;
            l[r] = 0;
#pragma clang loop unroll(full)
            for (uint i = 0; i < 4; i++) {
                o[r][i] = simdgroup_float8x8(0);
            }
        }

        for (uint start = first; start < pass_len; start += p.splits * DEC_CHUNK) {
            // Scores: simdgroup s takes tokens 16s .. 16s + 15, two tiles of
            // 8, for all rows of the pass; this lane's rows are ta and tb.
            {
                uint ta = start + 16 * simd + f.x, tb = ta + 8;
                simdgroup_float8x8 s[ROWS_PASS][2];
#pragma clang loop unroll(full)
                for (uint r = 0; r < ROWS_PASS; r++) {
                    s[r][0] = simdgroup_float8x8(0);
                    s[r][1] = simdgroup_float8x8(0);
                }
                // 32 dims per step, inside one group: eight tile loads in
                // flight.
                for (uint d = 0; d < DEC_DIM; d += 32) {
                    float2 za = kv_group<BITS>(kc, ta, d), zb = kv_group<BITS>(kc, tb, d);
                    simdgroup_float8x8 k0[4], k1[4];
#pragma clang loop unroll(full)
                    for (uint i = 0; i < 4; i++) {
                        kv_pair<BITS>(k0[i], kc, ta, d + 8 * i + f.y, za);
                        kv_pair<BITS>(k1[i], kc, tb, d + 8 * i + f.y, zb);
                    }
#pragma clang loop unroll(full)
                    for (uint r = 0; r < ROWS_PASS; r++) {
                        if (r >= rows) {
                            break;
                        }
                        device const float *qr =
                            q + ((r0 + r) * p.heads + kv * DEC_GROUP) * DEC_DIM + d;
#pragma clang loop unroll(full)
                        for (uint i = 0; i < 4; i++) {
                            simdgroup_float8x8 qt;
                            simdgroup_load(qt, qr + 8 * i, DEC_DIM, ulong2(0, 0), true);
                            simdgroup_multiply_accumulate(s[r][0], k0[i], qt, s[r][0]);
                            simdgroup_multiply_accumulate(s[r][1], k1[i], qt, s[r][1]);
                        }
                    }
                }
#pragma clang loop unroll(full)
                for (uint r = 0; r < ROWS_PASS; r++) {
                    if (r >= rows) {
                        break;
                    }
                    // s[r][i] is S^T: (token, head); store as (head, token).
                    simdgroup_store(s[r][0], &sc[r * DEC_GROUP][16 * simd], SC_STRIDE,
                                    ulong2(0, 0), true);
                    simdgroup_store(s[r][1], &sc[r * DEC_GROUP][16 * simd + 8], SC_STRIDE,
                                    ulong2(0, 0), true);
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            // Softmax step: simdgroup g takes query head g of each row. A
            // row may end before this chunk (n = 0).
#pragma clang loop unroll(full)
            for (uint r = 0; r < ROWS_PASS; r++) {
                if (r >= rows) {
                    break;
                }
                uint len = min(pos + r0 + r + 1, p.capacity);
                uint n = start < len ? min(DEC_CHUNK, len - start) : 0;
                chunk_softmax(sc[r * DEC_GROUP + simd], n, p.scale, m[r], l[r],
                              alpha + r * DEC_GROUP, simd, lane);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            // Values: simdgroup s takes dims 32s .. 32s + 31 (one group),
            // four tiles of 8, for all rows of the pass.
            {
                uint vd = 32 * simd;
#pragma clang loop unroll(full)
                for (uint r = 0; r < ROWS_PASS; r++) {
                    float a = alpha[r * DEC_GROUP + f.x];
#pragma clang loop unroll(full)
                    for (uint i = 0; i < 4; i++) {
                        scale_tile(o[r][i], a);
                    }
                }
                // 16 tokens per step: eight value tile loads in flight.
                for (uint t = 0; t < DEC_CHUNK; t += 16) {
                    simdgroup_float8x8 vt[2][4];
#pragma clang loop unroll(full)
                    for (uint h = 0; h < 2; h++) {
                        uint token = start + t + 8 * h + f.x;
                        float2 z = kv_group<BITS>(vc, token, vd);
#pragma clang loop unroll(full)
                        for (uint i = 0; i < 4; i++) {
                            kv_pair<BITS>(vt[h][i], vc, token, vd + 8 * i + f.y, z);
                        }
                    }
#pragma clang loop unroll(full)
                    for (uint r = 0; r < ROWS_PASS; r++) {
                        if (r >= rows) {
                            break;
                        }
#pragma clang loop unroll(full)
                        for (uint h = 0; h < 2; h++) {
                            simdgroup_float8x8 pt;
                            simdgroup_load(pt, &sc[r * DEC_GROUP][t + 8 * h], SC_STRIDE);
#pragma clang loop unroll(full)
                            for (uint i = 0; i < 4; i++) {
                                simdgroup_multiply_accumulate(o[r][i], pt, vt[h][i], o[r][i]);
                            }
                        }
                    }
                }
            }
            // The next chunk's scores overwrite sc and alpha.
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        // Rows that end before this split's first chunk have no result here,
        // and attention_reduce_rows does not read one.
#pragma clang loop unroll(full)
        for (uint r = 0; r < ROWS_PASS; r++) {
            if (r >= rows) {
                break;
            }
            uint row = r0 + r;
            if (first >= min(pos + row + 1, p.capacity)) {
                continue;
            }
            uint i = (row * p.splits + tg.x) * p.heads + kv * DEC_GROUP;
            if (lane == 0) {
                part_ml[2 * (i + simd)] = m[r];
                part_ml[2 * (i + simd) + 1] = l[r];
            }
            device float *out = part_o + i * DEC_DIM + 32 * simd;
#pragma clang loop unroll(full)
            for (uint j = 0; j < 4; j++) {
                simdgroup_store(o[r][j], out + 8 * j, DEC_DIM);
            }
        }
    }
}

template [[host_name("attention_partial_rows")]] kernel decltype(attention_partial_rows<16>)
    attention_partial_rows<16>;
template [[host_name("attention_partial_rows_q8")]] kernel decltype(attention_partial_rows<8>)
    attention_partial_rows<8>;
template [[host_name("attention_partial_rows_q4")]] kernel decltype(attention_partial_rows<4>)
    attention_partial_rows<4>;

// attention_reduce for all m rows in one launch: threadgroup (head, row),
// 1024 threads as 4 slices of 256 dims, each slice taking every 4th split.
kernel void attention_reduce_rows(device const float *part_o [[buffer(0)]],
                                  device const float *part_ml [[buffer(1)]],
                                  device const float *q_and_gate [[buffer(2)]],
                                  device float *out [[buffer(3)]],
                                  constant RowsAttentionParams &p [[buffer(4)]],
                                  constant TokenParams &tp [[buffer(5)]],
                                  uint2 tg [[threadgroup_position_in_grid]],
                                  uint tid [[thread_index_in_threadgroup]]) {
    threadgroup float slice_m[4];
    threadgroup float slice_l[4];
    threadgroup float slice_o[4][DEC_DIM];
    uint h = tg.x, row = tg.y;
    uint d = tid % DEC_DIM, s = tid / DEC_DIM;
    uint len = min(tp.position + row + 1, p.capacity);
    uint parts = min(p.splits, (len + DEC_CHUNK - 1) / DEC_CHUNK);
    device const float *ml = part_ml + 2 * row * p.splits * p.heads;
    device const float *po = part_o + row * p.splits * p.heads * DEC_DIM;

    float m = -INFINITY;
    for (uint c = s; c < parts; c += 4) {
        m = max(m, ml[2 * (c * p.heads + h)]);
    }
    if (d == 0) {
        slice_m[s] = m;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    m = max(max(slice_m[0], slice_m[1]), max(slice_m[2], slice_m[3]));

    float l = 0, acc = 0;
    for (uint c = s; c < parts; c += 4) {
        uint i = c * p.heads + h;
        float w = exp(ml[2 * i] - m);
        l += w * ml[2 * i + 1];
        acc += w * po[i * DEC_DIM + d];
    }
    slice_o[s][d] = acc;
    if (d == 0) {
        slice_l[s] = l;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (s == 0) {
        float o = slice_o[0][d] + slice_o[1][d] + slice_o[2][d] + slice_o[3][d];
        float total = slice_l[0] + slice_l[1] + slice_l[2] + slice_l[3];
        uint qh = row * p.heads + h;
        out[qh * DEC_DIM + d] = o / total * sigmoid(q_and_gate[qh * 2 * DEC_DIM + DEC_DIM + d]);
    }
}

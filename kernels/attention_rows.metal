// Split-K decode attention for m rows at once (the speculative verify),
// with the layout of attention.metal: one threadgroup per (128-token chunk,
// kv head), and part_o / part_ml per row as attention_partial writes them,
// so attention_reduce merges each row. Row r is at position pos + r and sees
// kv_len = pos + r + 1 tokens. The chunk's keys and values are read once
// for up to ROWS_PASS rows: scores S^T = K Q^T and O = P V in 8x8
// simdgroup matrices. More rows take more passes.
//
// The cache is f32 like attention.metal's, with room for whole chunks
// (GpuKvCache rounds it up), so a chunk never reaches past it.

constant uint ROWS_PASS = 4;
// Row stride of the score tile: 128 + 4 floats, so the 8 rows of an 8x8
// tile load fall in different banks.
constant uint SC_STRIDE = DEC_CHUNK + 4;

struct RowsAttentionParams {
    uint heads;
    uint kv_heads;
    uint capacity;
    float scale;
    uint rows;    // m
    uint chunks;  // per row in part_o and part_ml
};

kernel void attention_partial_rows(device const float *q [[buffer(0)]],
                                   device const float *keys [[buffer(1)]],
                                   device const float *values [[buffer(2)]],
                                   device float *part_o [[buffer(3)]],
                                   device float *part_ml [[buffer(4)]],
                                   constant RowsAttentionParams &p [[buffer(5)]],
                                   constant TokenParams &tp [[buffer(6)]],
                                   uint2 tg [[threadgroup_position_in_grid]],
                                   uint simd [[simdgroup_index_in_threadgroup]],
                                   uint lane [[thread_index_in_simdgroup]]) {
    // Scores of the pass's rows: sc[r * 8 + g][t].
    threadgroup float sc[ROWS_PASS * DEC_GROUP][SC_STRIDE];
    uint pos = tp.position;
    uint start = tg.x * DEC_CHUNK;
    uint last_len = min(pos + p.rows, p.capacity);
    if (start >= last_len) {
        return;
    }
    uint kv = tg.y;
    uint stride = p.kv_heads * DEC_DIM;
    uint chunks = p.chunks;

    for (uint r0 = 0; r0 < p.rows; r0 += ROWS_PASS) {
        uint rows = min(ROWS_PASS, p.rows - r0);

        // Scores: simdgroup s takes tokens 16s .. 16s + 15, two tiles of 8,
        // for all rows of the pass.
        {
            device const float *k = keys + (start + 16 * simd) * stride + kv * DEC_DIM;
            simdgroup_float8x8 s[ROWS_PASS][2];
#pragma clang loop unroll(full)
            for (uint r = 0; r < ROWS_PASS; r++) {
                s[r][0] = simdgroup_float8x8(0);
                s[r][1] = simdgroup_float8x8(0);
            }
            // 32 dims per step: 128 contiguous bytes of each key row, eight
            // tile loads in flight.
            for (uint d = 0; d < DEC_DIM; d += 32) {
                simdgroup_float8x8 k0[4], k1[4];
#pragma clang loop unroll(full)
                for (uint i = 0; i < 4; i++) {
                    simdgroup_load(k0[i], k + d + 8 * i, stride);
                    simdgroup_load(k1[i], k + 8 * stride + d + 8 * i, stride);
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
                simdgroup_store(s[r][0], &sc[r * DEC_GROUP][16 * simd], SC_STRIDE, ulong2(0, 0), true);
                simdgroup_store(s[r][1], &sc[r * DEC_GROUP][16 * simd + 8], SC_STRIDE, ulong2(0, 0), true);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Softmax pieces: simdgroup g takes query head g of each row.
        #pragma clang loop unroll(full)
        for (uint r = 0; r < ROWS_PASS; r++) {
            if (r >= rows) {
                break;
            }
            uint g = simd;
            uint row = r0 + r;
            uint len = min(pos + row + 1, p.capacity);
            uint n = start < len ? min(DEC_CHUNK, len - start) : 0;
            threadgroup float *sr = sc[r * DEC_GROUP + g];
            float m = -INFINITY;
            for (uint t = lane; t < n; t += 32) {
                m = max(m, sr[t] * p.scale);
            }
            m = simd_max(m);
            float l = 0;
            for (uint t = lane; t < DEC_CHUNK; t += 32) {
                float e = t < n ? exp(sr[t] * p.scale - m) : 0;
                sr[t] = e;
                l += e;
            }
            l = simd_sum(l);
            if (lane == 0 && n > 0) {
                uint i = (row * chunks + tg.x) * p.heads + kv * DEC_GROUP + g;
                part_ml[2 * i] = m;
                part_ml[2 * i + 1] = l;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Values: simdgroup s takes dims 32s .. 32s + 31, four tiles of 8,
        // for all rows of the pass.
        {
            device const float *v = values + start * stride + kv * DEC_DIM + 32 * simd;
            simdgroup_float8x8 o[ROWS_PASS][4];
#pragma clang loop unroll(full)
            for (uint r = 0; r < ROWS_PASS; r++) {
#pragma clang loop unroll(full)
                for (uint i = 0; i < 4; i++) {
                    o[r][i] = simdgroup_float8x8(0);
                }
            }
            // 16 tokens per step: eight value tile loads in flight.
            for (uint t = 0; t < DEC_CHUNK; t += 16) {
                simdgroup_float8x8 vt[2][4];
#pragma clang loop unroll(full)
                for (uint h = 0; h < 2; h++) {
#pragma clang loop unroll(full)
                    for (uint i = 0; i < 4; i++) {
                        simdgroup_load(vt[h][i], v + (t + 8 * h) * stride + 8 * i, stride);
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
            #pragma clang loop unroll(full)
        for (uint r = 0; r < ROWS_PASS; r++) {
            if (r >= rows) {
                break;
            }
                uint row = r0 + r;
                if (start >= min(pos + row + 1, p.capacity)) {
                    continue;
                }
                device float *out =
                    part_o + ((row * chunks + tg.x) * p.heads + kv * DEC_GROUP) * DEC_DIM + 32 * simd;
#pragma clang loop unroll(full)
                for (uint i = 0; i < 4; i++) {
                    simdgroup_store(o[r][i], out + 8 * i, DEC_DIM);
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

// attention_reduce for all m rows in one launch: threadgroup (head, row),
// 1024 threads as 4 slices of 256 dims, each slice taking every 4th chunk,
// so the chunk loop is 4 times shorter than attention_reduce's.
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
    uint chunks = (len + DEC_CHUNK - 1) / DEC_CHUNK;
    device const float *ml = part_ml + 2 * row * p.chunks * p.heads;
    device const float *po = part_o + row * p.chunks * p.heads * DEC_DIM;

    float m = -INFINITY;
    for (uint c = s; c < chunks; c += 4) {
        m = max(m, ml[2 * (c * p.heads + h)]);
    }
    if (d == 0) {
        slice_m[s] = m;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    m = max(max(slice_m[0], slice_m[1]), max(slice_m[2], slice_m[3]));

    float l = 0, acc = 0;
    for (uint c = s; c < chunks; c += 4) {
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

// Timing only: attention_partial_rows without its score loop.
kernel void attention_partial_rows_nos(device const float *q [[buffer(0)]],
                                   device const float *keys [[buffer(1)]],
                                   device const float *values [[buffer(2)]],
                                   device float *part_o [[buffer(3)]],
                                   device float *part_ml [[buffer(4)]],
                                   constant RowsAttentionParams &p [[buffer(5)]],
                                   constant TokenParams &tp [[buffer(6)]],
                                   uint2 tg [[threadgroup_position_in_grid]],
                                   uint simd [[simdgroup_index_in_threadgroup]],
                                   uint lane [[thread_index_in_simdgroup]]) {
    // Scores of the pass's rows: sc[r * 8 + g][t].
    threadgroup float sc[ROWS_PASS * DEC_GROUP][SC_STRIDE];
    uint pos = tp.position;
    uint start = tg.x * DEC_CHUNK;
    uint last_len = min(pos + p.rows, p.capacity);
    if (start >= last_len) {
        return;
    }
    uint kv = tg.y;
    uint stride = p.kv_heads * DEC_DIM;
    uint chunks = p.chunks;

    for (uint r0 = 0; r0 < p.rows; r0 += ROWS_PASS) {
        uint rows = min(ROWS_PASS, p.rows - r0);

        // Scores: simdgroup s takes tokens 16s .. 16s + 15, two tiles of 8,
        // for all rows of the pass.
        {
            device const float *k = keys + (start + 16 * simd) * stride + kv * DEC_DIM;
            simdgroup_float8x8 s[ROWS_PASS][2];
#pragma clang loop unroll(full)
            for (uint r = 0; r < ROWS_PASS; r++) {
                s[r][0] = simdgroup_float8x8(0);
                s[r][1] = simdgroup_float8x8(0);
            }
            // 32 dims per step: 128 contiguous bytes of each key row, eight
            // tile loads in flight.
            #pragma clang loop unroll(full)
        for (uint r = 0; r < ROWS_PASS; r++) {
            if (r >= rows) {
                break;
            }
                // s[r][i] is S^T: (token, head); store as (head, token).
                simdgroup_store(s[r][0], &sc[r * DEC_GROUP][16 * simd], SC_STRIDE, ulong2(0, 0), true);
                simdgroup_store(s[r][1], &sc[r * DEC_GROUP][16 * simd + 8], SC_STRIDE, ulong2(0, 0), true);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Softmax pieces: simdgroup g takes query head g of each row.
        #pragma clang loop unroll(full)
        for (uint r = 0; r < ROWS_PASS; r++) {
            if (r >= rows) {
                break;
            }
            uint g = simd;
            uint row = r0 + r;
            uint len = min(pos + row + 1, p.capacity);
            uint n = start < len ? min(DEC_CHUNK, len - start) : 0;
            threadgroup float *sr = sc[r * DEC_GROUP + g];
            float m = -INFINITY;
            for (uint t = lane; t < n; t += 32) {
                m = max(m, sr[t] * p.scale);
            }
            m = simd_max(m);
            float l = 0;
            for (uint t = lane; t < DEC_CHUNK; t += 32) {
                float e = t < n ? exp(sr[t] * p.scale - m) : 0;
                sr[t] = e;
                l += e;
            }
            l = simd_sum(l);
            if (lane == 0 && n > 0) {
                uint i = (row * chunks + tg.x) * p.heads + kv * DEC_GROUP + g;
                part_ml[2 * i] = m;
                part_ml[2 * i + 1] = l;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Values: simdgroup s takes dims 32s .. 32s + 31, four tiles of 8,
        // for all rows of the pass.
        {
            device const float *v = values + start * stride + kv * DEC_DIM + 32 * simd;
            simdgroup_float8x8 o[ROWS_PASS][4];
#pragma clang loop unroll(full)
            for (uint r = 0; r < ROWS_PASS; r++) {
#pragma clang loop unroll(full)
                for (uint i = 0; i < 4; i++) {
                    o[r][i] = simdgroup_float8x8(0);
                }
            }
            // 16 tokens per step: eight value tile loads in flight.
            for (uint t = 0; t < DEC_CHUNK; t += 16) {
                simdgroup_float8x8 vt[2][4];
#pragma clang loop unroll(full)
                for (uint h = 0; h < 2; h++) {
#pragma clang loop unroll(full)
                    for (uint i = 0; i < 4; i++) {
                        simdgroup_load(vt[h][i], v + (t + 8 * h) * stride + 8 * i, stride);
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
            #pragma clang loop unroll(full)
        for (uint r = 0; r < ROWS_PASS; r++) {
            if (r >= rows) {
                break;
            }
                uint row = r0 + r;
                if (start >= min(pos + row + 1, p.capacity)) {
                    continue;
                }
                device float *out =
                    part_o + ((row * chunks + tg.x) * p.heads + kv * DEC_GROUP) * DEC_DIM + 32 * simd;
#pragma clang loop unroll(full)
                for (uint i = 0; i < 4; i++) {
                    simdgroup_store(o[r][i], out + 8 * i, DEC_DIM);
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

// Timing only: without its value loop.
kernel void attention_partial_rows_nov(device const float *q [[buffer(0)]],
                                   device const float *keys [[buffer(1)]],
                                   device const float *values [[buffer(2)]],
                                   device float *part_o [[buffer(3)]],
                                   device float *part_ml [[buffer(4)]],
                                   constant RowsAttentionParams &p [[buffer(5)]],
                                   constant TokenParams &tp [[buffer(6)]],
                                   uint2 tg [[threadgroup_position_in_grid]],
                                   uint simd [[simdgroup_index_in_threadgroup]],
                                   uint lane [[thread_index_in_simdgroup]]) {
    // Scores of the pass's rows: sc[r * 8 + g][t].
    threadgroup float sc[ROWS_PASS * DEC_GROUP][SC_STRIDE];
    uint pos = tp.position;
    uint start = tg.x * DEC_CHUNK;
    uint last_len = min(pos + p.rows, p.capacity);
    if (start >= last_len) {
        return;
    }
    uint kv = tg.y;
    uint stride = p.kv_heads * DEC_DIM;
    uint chunks = p.chunks;

    for (uint r0 = 0; r0 < p.rows; r0 += ROWS_PASS) {
        uint rows = min(ROWS_PASS, p.rows - r0);

        // Scores: simdgroup s takes tokens 16s .. 16s + 15, two tiles of 8,
        // for all rows of the pass.
        {
            device const float *k = keys + (start + 16 * simd) * stride + kv * DEC_DIM;
            simdgroup_float8x8 s[ROWS_PASS][2];
#pragma clang loop unroll(full)
            for (uint r = 0; r < ROWS_PASS; r++) {
                s[r][0] = simdgroup_float8x8(0);
                s[r][1] = simdgroup_float8x8(0);
            }
            // 32 dims per step: 128 contiguous bytes of each key row, eight
            // tile loads in flight.
            for (uint d = 0; d < DEC_DIM; d += 32) {
                simdgroup_float8x8 k0[4], k1[4];
#pragma clang loop unroll(full)
                for (uint i = 0; i < 4; i++) {
                    simdgroup_load(k0[i], k + d + 8 * i, stride);
                    simdgroup_load(k1[i], k + 8 * stride + d + 8 * i, stride);
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
                simdgroup_store(s[r][0], &sc[r * DEC_GROUP][16 * simd], SC_STRIDE, ulong2(0, 0), true);
                simdgroup_store(s[r][1], &sc[r * DEC_GROUP][16 * simd + 8], SC_STRIDE, ulong2(0, 0), true);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Softmax pieces: simdgroup g takes query head g of each row.
        #pragma clang loop unroll(full)
        for (uint r = 0; r < ROWS_PASS; r++) {
            if (r >= rows) {
                break;
            }
            uint g = simd;
            uint row = r0 + r;
            uint len = min(pos + row + 1, p.capacity);
            uint n = start < len ? min(DEC_CHUNK, len - start) : 0;
            threadgroup float *sr = sc[r * DEC_GROUP + g];
            float m = -INFINITY;
            for (uint t = lane; t < n; t += 32) {
                m = max(m, sr[t] * p.scale);
            }
            m = simd_max(m);
            float l = 0;
            for (uint t = lane; t < DEC_CHUNK; t += 32) {
                float e = t < n ? exp(sr[t] * p.scale - m) : 0;
                sr[t] = e;
                l += e;
            }
            l = simd_sum(l);
            if (lane == 0 && n > 0) {
                uint i = (row * chunks + tg.x) * p.heads + kv * DEC_GROUP + g;
                part_ml[2 * i] = m;
                part_ml[2 * i + 1] = l;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Values: simdgroup s takes dims 32s .. 32s + 31, four tiles of 8,
        // for all rows of the pass.
        {
            device const float *v = values + start * stride + kv * DEC_DIM + 32 * simd;
            simdgroup_float8x8 o[ROWS_PASS][4];
#pragma clang loop unroll(full)
            for (uint r = 0; r < ROWS_PASS; r++) {
#pragma clang loop unroll(full)
                for (uint i = 0; i < 4; i++) {
                    o[r][i] = simdgroup_float8x8(0);
                }
            }
            #pragma clang loop unroll(full)
        for (uint r = 0; r < ROWS_PASS; r++) {
            if (r >= rows) {
                break;
            }
                uint row = r0 + r;
                if (start >= min(pos + row + 1, p.capacity)) {
                    continue;
                }
                device float *out =
                    part_o + ((row * chunks + tg.x) * p.heads + kv * DEC_GROUP) * DEC_DIM + 32 * simd;
#pragma clang loop unroll(full)
                for (uint i = 0; i < 4; i++) {
                    simdgroup_store(o[r][i], out + 8 * i, DEC_DIM);
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

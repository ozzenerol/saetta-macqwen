// Causal attention for a prefill chunk, flash style with simdgroup matrices:
//   out[t, h] = softmax(q[t, h] K^T * scale) V * sigmoid(gate[t, h])
// over keys 0 .. pos + t. q, keys and values are half, out f32; buffers and
// PrefillAttentionParams as attention_rows in prefill.metal.
//
// A threadgroup takes AF_TOK tokens of one kv head: 8 tokens x its 8 query
// heads = 64 rows, which share every key and value. Per block of AF_BC keys:
//   S: simdgroup s computes the scores of token s's 8 heads (q of those
//      rows lives in registers) against the block, which sits in threadgroup
//      memory, and runs the online softmax on them in registers.
//   PV: simdgroup s accumulates dims 32 s .. + 31 of all 64 rows, with P
//      from threadgroup memory and V read from the cache.
// Head dim 256 and 8 q heads per kv head are fixed; src/attention_prefill.rs
// checks them. The cache must hold AF_BC - 1 keys past pos + tokens (its
// capacity is a multiple of 64).

constant uint AF_TOK = 8;
constant uint AF_ROWS = AF_TOK * 8;
constant uint AF_BC = 32;
constant uint AF_DIM = 256;
constant uint AF_LDK = AF_DIM + 8;
constant uint AF_LDP = AF_BC + 8;

struct FlashParams {
    uint heads;
    uint kv_heads;
    uint pos;
    uint tokens;
    float scale;
};

// Row and first column of a lane's two values in an 8 x 8 simdgroup matrix.
inline uint2 af_frag(uint lane) {
    uint qid = lane / 4;
    return uint2((qid & 4) + (lane / 2) % 4, (qid & 2) * 2 + (lane % 2) * 2);
}

kernel void attention_flash(device const half *q [[buffer(0)]],
                            device const half *keys [[buffer(1)]],
                            device const half *values [[buffer(2)]],
                            device const float *q_and_gate [[buffer(3)]],
                            device float *out [[buffer(4)]],
                            constant FlashParams &p [[buffer(5)]],
                            uint2 tg [[threadgroup_position_in_grid]],
                            uint tid [[thread_index_in_threadgroup]],
                            uint simd [[simdgroup_index_in_threadgroup]],
                            uint lane [[thread_index_in_simdgroup]]) {
    threadgroup half kt[AF_BC * AF_LDK];
    threadgroup half pt[AF_ROWS * AF_LDP];
    threadgroup float alpha[AF_ROWS];
    threadgroup float lsum[AF_ROWS];

    uint kv = tg.y;
    uint tok0 = tg.x * AF_TOK;
    uint stride = p.kv_heads * AF_DIM;
    uint2 fr = af_frag(lane);
    uint tok = tok0 + simd;
    bool row_lane = (lane & 1) == 0 && (lane & 8) == 0;  // one lane per row

    simdgroup_half8x8 qf[AF_DIM / 8];
    {
        uint t = min(tok, p.tokens - 1);
        device const half *qp = q + (ulong(t) * p.heads + kv * 8) * AF_DIM;
        for (uint d = 0; d < AF_DIM / 8; d++) {
            simdgroup_load(qf[d], qp + 8 * d, AF_DIM);
        }
    }
    simdgroup_float8x8 o[AF_TOK][4];
    for (uint i = 0; i < AF_TOK; i++) {
        for (uint j = 0; j < 4; j++) {
            o[i][j] = simdgroup_float8x8(0);
        }
    }
    float m = -INFINITY, l = 0;  // of row fr.x of this simdgroup's 8 rows
    float sl2 = p.scale * M_LOG2E_F;
    uint limit = p.pos + tok;  // keys this simdgroup's rows may see
    uint last = p.pos + min(tok0 + AF_TOK, p.tokens) - 1;
    uint blocks = last / AF_BC + 1;

    for (uint b = 0; b < blocks; b++) {
        uint j0 = b * AF_BC;
        // The key block: 32 keys of 32 uint4, 4 uint4 per thread.
        for (uint i = 0; i < 4; i++) {
            uint idx = tid * 4 + i;
            uint key = idx / 32, c = idx % 32;
            ((threadgroup uint4 *)(kt + key * AF_LDK))[c] =
                ((device const uint4 *)(keys + ulong(j0 + key) * stride + kv * AF_DIM))[c];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        simdgroup_float8x8 sf[AF_BC / 8];
        for (uint f = 0; f < AF_BC / 8; f++) {
            sf[f] = simdgroup_float8x8(0);
        }
        for (uint d = 0; d < AF_DIM / 8; d++) {
            for (uint f = 0; f < AF_BC / 8; f++) {
                simdgroup_half8x8 kf;
                simdgroup_load(kf, kt + 8 * f * AF_LDK + 8 * d, AF_LDK, ulong2(0, 0), true);
                simdgroup_multiply_accumulate(sf[f], qf[d], kf, sf[f]);
            }
        }

        float s[AF_BC / 4];
        float bm = -INFINITY;
        for (uint f = 0; f < AF_BC / 8; f++) {
            for (uint e = 0; e < 2; e++) {
                uint key = j0 + 8 * f + fr.y + e;
                float v = sf[f].thread_elements()[e] * sl2;
                v = key <= limit ? v : -INFINITY;
                s[2 * f + e] = v;
                bm = max(bm, v);
            }
        }
        bm = max(bm, simd_shuffle_xor(bm, 1));
        bm = max(bm, simd_shuffle_xor(bm, 8));
        float mn = max(m, bm);
        float a = exp2(m - mn);
        float rs = 0;
        threadgroup half *prow = pt + (8 * simd + fr.x) * AF_LDP;
        for (uint f = 0; f < AF_BC / 8; f++) {
            float p0 = exp2(s[2 * f] - mn), p1 = exp2(s[2 * f + 1] - mn);
            rs += p0 + p1;
            *(threadgroup half2 *)(prow + 8 * f + fr.y) = half2(p0, p1);
        }
        rs += simd_shuffle_xor(rs, 1);
        rs += simd_shuffle_xor(rs, 8);
        l = l * a + rs;
        m = mn;
        if (row_lane) {
            alpha[8 * simd + fr.x] = a;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint i = 0; i < AF_TOK; i++) {
            float ai = alpha[8 * i + fr.x];
            for (uint j = 0; j < 4; j++) {
                thread auto &e = o[i][j].thread_elements();
                e[0] *= ai;
                e[1] *= ai;
            }
        }
        for (uint kk = 0; kk < AF_BC / 8; kk++) {
            simdgroup_half8x8 vf[4];
            device const half *vp = values + ulong(j0 + 8 * kk) * stride + kv * AF_DIM + 32 * simd;
            for (uint j = 0; j < 4; j++) {
                simdgroup_load(vf[j], vp + 8 * j, stride);
            }
            for (uint i = 0; i < AF_TOK; i++) {
                simdgroup_half8x8 pf;
                simdgroup_load(pf, pt + 8 * i * AF_LDP + 8 * kk, AF_LDP);
                for (uint j = 0; j < 4; j++) {
                    simdgroup_multiply_accumulate(o[i][j], pf, vf[j], o[i][j]);
                }
            }
        }
    }

    if (row_lane) {
        lsum[8 * simd + fr.x] = l;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = 0; i < AF_TOK; i++) {
        uint t = tok0 + i;
        if (t >= p.tokens) {
            break;
        }
        uint h = kv * 8 + fr.x;
        float inv = 1.0f / lsum[8 * i + fr.x];
        ulong row = ulong(t) * p.heads + h;
        for (uint j = 0; j < 4; j++) {
            uint d = 32 * simd + 8 * j + fr.y;
            device const float *g = q_and_gate + row * 2 * AF_DIM + AF_DIM + d;
            thread auto &e = o[i][j].thread_elements();
            float2 v = float2(e[0], e[1]) * inv;
            *(device float2 *)(out + row * AF_DIM + d) = v * float2(sigmoid(g[0]), sigmoid(g[1]));
        }
    }
}

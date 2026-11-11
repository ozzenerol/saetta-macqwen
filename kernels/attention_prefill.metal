// Flash attention for a prefill chunk on simdgroup matrices:
//   out[t, h] = softmax(q[t, h] K^T * scale) V * sigmoid(gate[t, h])
// over keys 0 .. pos + t, reading the half KV cache. Buffers and parameters
// as attention_rows_half in prefill.metal: q half [tokens][heads][256],
// keys and values half [capacity][kv_heads][256], q_and_gate f32
// [tokens][heads][512] (gate in the second half of each head), out f32
// [tokens][heads][256].
//
// A threadgroup takes TQ tokens of one kv head. Simdgroup s owns token
// tok0 + s and its 8 q heads (the GQA group), so its 8 rows share one causal
// limit. Per block of FA_BK keys the threadgroup copies K and V into
// threadgroup memory once, for all TQ * 8 rows. Q (8 x 256) and O (8 x 256)
// stay in registers, S and P too: in an 8 x 8 simdgroup matrix lane l holds
// two values of row fa_frag(l).x, for float and half alike, so the softmax
// and the rescale of O need no threadgroup memory. Head dim 256 and 8 q heads
// per kv head are fixed (src/attention_prefill.rs checks them). Keys are
// read up to the end of the block that holds key pos + tokens - 1, so the
// cache capacity must be a multiple of FA_BK.

constant uint FA_DIM = 256;
constant uint FA_BK = 16;
constant uint FA_LD = FA_DIM + 8;

struct FlashParams {
    uint heads;
    uint kv_heads;
    uint pos;
    uint tokens;
    float scale;
};

// Row and first column of a lane's two values in an 8 x 8 simdgroup matrix.
inline uint2 fa_frag(uint lane) {
    uint qid = lane / 4;
    return uint2((qid & 4) + (lane / 2) % 4, (qid & 2) * 2 + (lane % 2) * 2);
}

template <uint TQ>
kernel void attention_fa(device const half *q [[buffer(0)]],
                         device const half *keys [[buffer(1)]],
                         device const half *values [[buffer(2)]],
                         device const float *q_and_gate [[buffer(3)]],
                         device float *out [[buffer(4)]],
                         constant FlashParams &p [[buffer(5)]],
                         uint2 tg [[threadgroup_position_in_grid]],
                         uint tid [[thread_index_in_threadgroup]],
                         uint simd [[simdgroup_index_in_threadgroup]],
                         uint lane [[thread_index_in_simdgroup]]) {
    threadgroup half kt[FA_BK * FA_LD];
    threadgroup half vt[FA_BK * FA_LD];

    uint kv = tg.y;
    uint tok0 = tg.x * TQ;
    uint tok = min(tok0 + simd, p.tokens - 1);
    uint stride = p.kv_heads * FA_DIM;
    uint2 fr = fa_frag(lane);

    simdgroup_half8x8 qf[FA_DIM / 8];
    {
        device const half *qp = q + (ulong(tok) * p.heads + kv * 8) * FA_DIM;
        for (uint d = 0; d < FA_DIM / 8; d++) {
            simdgroup_load(qf[d], qp + 8 * d, FA_DIM);
        }
    }
    simdgroup_float8x8 o[FA_DIM / 8];
    for (uint j = 0; j < FA_DIM / 8; j++) {
        o[j] = simdgroup_float8x8(0);
    }

    float m = -INFINITY, l = 0;  // of row fr.x
    float sl2 = p.scale * M_LOG2E_F;
    uint limit = p.pos + tok;  // last key this simdgroup's rows see
    uint last = p.pos + min(tok0 + TQ, p.tokens) - 1;
    uint blocks = last / FA_BK + 1;
    device const half *kbase = keys + kv * FA_DIM;
    device const half *vbase = values + kv * FA_DIM;

    // A block is FA_BK keys x 32 uint4, for K and for V.
    constexpr uint CHUNKS = FA_BK * FA_DIM / 8;
    constexpr uint THREADS = TQ * 32;

    for (uint b = 0; b < blocks; b++) {
        uint j0 = b * FA_BK;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint i = tid; i < CHUNKS; i += THREADS) {
            uint key = i / 32, c = i % 32;
            ulong src = ulong(j0 + key) * stride;
            ((threadgroup uint4 *)(kt + key * FA_LD))[c] = ((device const uint4 *)(kbase + src))[c];
            ((threadgroup uint4 *)(vt + key * FA_LD))[c] = ((device const uint4 *)(vbase + src))[c];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (j0 > limit) {
            continue;  // the whole block is past this simdgroup's token
        }

        // S = Q K^T for the block: 8 rows x FA_BK keys.
        simdgroup_float8x8 sf[FA_BK / 8];
        for (uint f = 0; f < FA_BK / 8; f++) {
            sf[f] = simdgroup_float8x8(0);
        }
        for (uint d = 0; d < FA_DIM / 8; d++) {
            for (uint f = 0; f < FA_BK / 8; f++) {
                simdgroup_half8x8 kf;
                simdgroup_load(kf, kt + 8 * f * FA_LD + 8 * d, FA_LD, ulong2(0, 0), true);
                simdgroup_multiply_accumulate(sf[f], qf[d], kf, sf[f]);
            }
        }

        // Online softmax of row fr.x; its values sit in lanes l ^ 1, l ^ 8.
        float s[FA_BK / 4];
        float bm = -INFINITY;
        for (uint f = 0; f < FA_BK / 8; f++) {
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
        float mn = max(m, bm);  // finite: key j0 <= limit
        float a = exp2(m - mn);
        float rs = 0;
        simdgroup_half8x8 pf[FA_BK / 8];
        for (uint f = 0; f < FA_BK / 8; f++) {
            float p0 = exp2(s[2 * f] - mn), p1 = exp2(s[2 * f + 1] - mn);
            rs += p0 + p1;
            thread auto &pe = pf[f].thread_elements();
            pe[0] = half(p0);
            pe[1] = half(p1);
        }
        rs += simd_shuffle_xor(rs, 1);
        rs += simd_shuffle_xor(rs, 8);
        l = l * a + rs;
        m = mn;

        // O = O * a + P V.
        for (uint j = 0; j < FA_DIM / 8; j++) {
            thread auto &e = o[j].thread_elements();
            e[0] *= a;
            e[1] *= a;
            for (uint f = 0; f < FA_BK / 8; f++) {
                simdgroup_half8x8 vf;
                simdgroup_load(vf, vt + 8 * f * FA_LD + 8 * j, FA_LD);
                simdgroup_multiply_accumulate(o[j], pf[f], vf, o[j]);
            }
        }
    }

    if (tok0 + simd >= p.tokens) {
        return;
    }
    float inv = 1.0f / l;
    ulong row = ulong(tok) * p.heads + kv * 8 + fr.x;
    device const float *gate = q_and_gate + row * 2 * FA_DIM + FA_DIM;
    device float *dst = out + row * FA_DIM;
    for (uint j = 0; j < FA_DIM / 8; j++) {
        uint d = 8 * j + fr.y;
        thread auto &e = o[j].thread_elements();
        float2 g = float2(gate[d], gate[d + 1]);
        float2 sg = 1.0f / (1.0f + exp(-g));
        *(device float2 *)(dst + d) = float2(e[0], e[1]) * inv * sg;
    }
}

template [[host_name("attention_fa4")]] kernel decltype(attention_fa<4>) attention_fa<4>;
template [[host_name("attention_fa8")]] kernel decltype(attention_fa<8>) attention_fa<8>;

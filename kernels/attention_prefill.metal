// Causal attention for a prefill chunk on simdgroup matrices:
//   out[t, h] = softmax(q[t, h] K^T * scale) V * sigmoid(gate[t, h])
// over keys 0 .. pos + t. q, keys and values are half, out f32; buffers and
// parameters as attention_rows in prefill.metal.
//
// attention_rows2 from prefill.metal (scores and P through threadgroup
// memory, q in registers) with the scale folded into exp2 and blocks of KB
// keys. A simdgroup takes 8 rows: 8 tokens of one head, or with GQA the 8 q
// heads of one kv head for one token. A threadgroup has SIMDS simdgroups.
// Head dim 256 is fixed; GQA needs 8 q heads per kv head. Keys are read up
// to the 8-key block holding the last one, so a cache whose capacity is a
// multiple of 8 is enough.

constant uint AF_DIM = 256;

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

template <uint SIMDS, uint KB, bool GQA>
kernel void attention_g(device const half *q [[buffer(0)]],
                        device const half *keys [[buffer(1)]],
                        device const half *values [[buffer(2)]],
                        device const float *q_and_gate [[buffer(3)]],
                        device float *out [[buffer(4)]],
                        constant FlashParams &p [[buffer(5)]],
                        uint2 tg [[threadgroup_position_in_grid]],
                        uint simd [[simdgroup_index_in_threadgroup]],
                        uint lane [[thread_index_in_simdgroup]]) {
    constexpr uint D = AF_DIM;
    threadgroup float s_all[SIMDS][8 * KB];
    threadgroup half p_all[SIMDS][8 * KB];
    threadgroup float a_all[SIMDS][8];
    threadgroup float *sc = s_all[simd];
    threadgroup half *pt = p_all[simd];
    threadgroup float *al = a_all[simd];

    uint h = tg.y * SIMDS + simd;
    uint kv = GQA ? tg.y : h / (p.heads / p.kv_heads);
    uint q0 = GQA ? tg.x * SIMDS + simd : tg.x * 8;  // first token of the rows
    if (GQA && q0 >= p.tokens) {
        return;
    }
    uint q_stride = p.heads * D;
    uint kv_stride = p.kv_heads * D;
    device const half *kh = keys + kv * D;
    device const half *vh = values + kv * D;
    uint end = GQA ? p.pos + q0 + 1 : p.pos + min(q0 + 8, p.tokens);

    simdgroup_half8x8 qm[D / 8];
    {
        device const half *qh = GQA ? q + (ulong(q0) * p.heads + kv * 8) * D : q + ulong(q0) * q_stride + h * D;
        uint rs = GQA ? D : q_stride;
        for (uint d = 0; d < D / 8; d++) {
            simdgroup_load(qm[d], qh + 8 * d, rs);
        }
    }

    constexpr uint PER = KB / 4;
    uint r = lane / 4;
    uint c0 = (lane % 4) * PER;
    uint last = GQA ? end - 1 : p.pos + q0 + r;
    uint2 fp = af_frag(lane);
    float sl2 = p.scale * M_LOG2E_F;
    float m = -INFINITY;
    float l = 0;

    simdgroup_float8x8 o[D / 8];
    for (uint i = 0; i < D / 8; i++) {
        o[i] = simdgroup_float8x8(0);
    }

    for (uint k0 = 0; k0 < end; k0 += KB) {
        // The cache may stop after the 8-key block holding key end - 1, so
        // blocks past it reread that one; their keys are masked out of P.
        uint top = (end - 1) & ~7u;
        simdgroup_float8x8 s[KB / 8];
        for (uint j = 0; j < KB / 8; j++) {
            s[j] = simdgroup_float8x8(0);
        }
        for (uint d = 0; d < D / 8; d++) {
            for (uint j = 0; j < KB / 8; j++) {
                simdgroup_half8x8 km;
                simdgroup_load(km, kh + ulong(min(k0 + 8 * j, top)) * kv_stride + 8 * d, kv_stride, ulong2(0, 0), true);
                simdgroup_multiply_accumulate(s[j], qm[d], km, s[j]);
            }
        }
        for (uint j = 0; j < KB / 8; j++) {
            simdgroup_store(s[j], sc + 8 * j, KB);
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);

        float v[PER];
        float bm = -INFINITY;
        for (uint i = 0; i < PER; i++) {
            uint key = k0 + c0 + i;
            v[i] = key <= last ? sc[r * KB + c0 + i] * sl2 : -INFINITY;
            bm = max(bm, v[i]);
        }
        bm = max(bm, simd_shuffle_xor(bm, 1));
        bm = max(bm, simd_shuffle_xor(bm, 2));
        float m_new = max(m, bm);
        float sum = 0;
        for (uint i = 0; i < PER; i++) {
            float e = exp2(v[i] - m_new);
            pt[r * KB + c0 + i] = half(e);
            sum += e;
        }
        sum += simd_shuffle_xor(sum, 1);
        sum += simd_shuffle_xor(sum, 2);
        float alpha = exp2(m - m_new);
        l = l * alpha + sum;
        m = m_new;
        if (lane % 4 == 0) {
            al[r] = alpha;
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);

        float a = al[fp.x];
        simdgroup_half8x8 pm[KB / 8];
        for (uint j = 0; j < KB / 8; j++) {
            simdgroup_load(pm[j], pt + 8 * j, KB);
        }
        for (uint d = 0; d < D / 8; d++) {
            thread auto &oe = o[d].thread_elements();
            oe[0] *= a;
            oe[1] *= a;
            for (uint j = 0; j < KB / 8; j++) {
                simdgroup_half8x8 vm;
                simdgroup_load(vm, vh + ulong(min(k0 + 8 * j, top)) * kv_stride + 8 * d, kv_stride);
                simdgroup_multiply_accumulate(o[d], pm[j], vm, o[d]);
            }
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
    }

    float inv = 1.0f / l;
    uint row_t = GQA ? q0 : q0 + r;
    uint row_h = GQA ? kv * 8 + r : h;
    for (uint d = 0; d < D / 8; d++) {
        simdgroup_store(o[d], sc, 8);
        simdgroup_barrier(mem_flags::mem_threadgroup);
        if (row_t < p.tokens) {
            for (uint i = (lane % 4) * 2; i < (lane % 4) * 2 + 2; i++) {
                uint col = 8 * d + i;
                float gate = q_and_gate[ulong(row_t) * 2 * q_stride + row_h * 2 * D + D + col];
                out[ulong(row_t) * q_stride + row_h * D + col] = sc[r * 8 + i] * inv * sigmoid(gate);
            }
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
    }
}

#define ATT_G(name, simds, kb, gqa) \
    template [[host_name(name)]] kernel decltype(attention_g<simds, kb, gqa>) attention_g<simds, kb, gqa>;
ATT_G("attention_gq4_128", 4, 128, true)
ATT_G("attention_g4_128", 4, 128, false)

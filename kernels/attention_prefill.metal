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

// PRE: the next key block is copied in while the simdgroups do PV.
// KTG: keys staged in threadgroup memory; without, S reads them from the
// cache directly (less threadgroup memory, more threadgroups per core).
template <bool PRE, bool KTG>
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
    threadgroup half kt[KTG ? AF_BC * AF_LDK : 1];
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

    // The key block from key j0: 32 keys of 32 uint4, 4 uint4 per thread.
    auto load_keys = [&](uint j0) {
        if (!KTG) {
            return;
        }
        for (uint i = 0; i < 4; i++) {
            uint idx = tid * 4 + i;
            uint key = idx / 32, c = idx % 32;
            ((threadgroup uint4 *)(kt + key * AF_LDK))[c] =
                ((device const uint4 *)(keys + ulong(j0 + key) * stride + kv * AF_DIM))[c];
        }
    };
    if (PRE && KTG) {
        load_keys(0);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for (uint b = 0; b < blocks; b++) {
        uint j0 = b * AF_BC;
        if (!PRE && KTG) {
            load_keys(j0);
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        simdgroup_float8x8 sf[AF_BC / 8];
        for (uint f = 0; f < AF_BC / 8; f++) {
            sf[f] = simdgroup_float8x8(0);
        }
        for (uint d = 0; d < AF_DIM / 8; d++) {
            for (uint f = 0; f < AF_BC / 8; f++) {
                simdgroup_half8x8 kf;
                if (KTG) {
                    simdgroup_load(kf, kt + 8 * f * AF_LDK + 8 * d, AF_LDK, ulong2(0, 0), true);
                } else {
                    simdgroup_load(kf, keys + ulong(j0 + 8 * f) * stride + kv * AF_DIM + 8 * d, stride,
                                   ulong2(0, 0), true);
                }
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
        if (PRE && b + 1 < blocks) {
            load_keys(j0 + AF_BC);
        }

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
        if (PRE || !KTG) {
            threadgroup_barrier(mem_flags::mem_threadgroup);
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

template [[host_name("attention_flash")]] kernel decltype(attention_flash<false, true>) attention_flash<false, true>;
template [[host_name("attention_flash_pre")]] kernel decltype(attention_flash<true, true>) attention_flash<true, true>;
template [[host_name("attention_flash_dev")]] kernel decltype(attention_flash<false, false>) attention_flash<false, false>;

// attention_flash_reg: no threadgroup memory and no barriers. Simdgroup s of
// a threadgroup takes token 4 g + s and its 8 heads of kv head tg.y, the 8
// rows of one matrix. q and the keys and values are read as matrices from
// memory (the 4 simdgroups share them through the cache), the scores stay
// in registers and go straight into P V as the A operand.
template <bool QREG, uint BC, uint SPLIT>
kernel void attention_flash_reg(device const half *q [[buffer(0)]],
                                device const half *keys [[buffer(1)]],
                                device const half *values [[buffer(2)]],
                                device const float *q_and_gate [[buffer(3)]],
                                device float *out [[buffer(4)]],
                                constant FlashParams &p [[buffer(5)]],
                                uint2 tg [[threadgroup_position_in_grid]],
                                uint simd [[simdgroup_index_in_threadgroup]],
                                uint lane [[thread_index_in_simdgroup]]) {
    uint kv = tg.y;
    uint tok = tg.x * 4 + simd;
    if (tok >= p.tokens) {
        return;
    }
    uint stride = p.kv_heads * AF_DIM;
    uint2 fr = af_frag(lane);
    device const half *qp = q + (ulong(tok) * p.heads + kv * 8) * AF_DIM;
    device const half *kb = keys + kv * AF_DIM;
    device const half *vb = values + kv * AF_DIM;

    simdgroup_half8x8 qr[QREG ? AF_DIM / 8 : 1];
    if (QREG) {
        for (uint d = 0; d < AF_DIM / 8; d++) {
            simdgroup_load(qr[d], qp + 8 * d, AF_DIM);
        }
    }
    simdgroup_float8x8 o[AF_DIM / 8];
    for (uint j = 0; j < AF_DIM / 8; j++) {
        o[j] = simdgroup_float8x8(0);
    }
    float m = -INFINITY, l = 0;
    float sl2 = p.scale * M_LOG2E_F;
    uint limit = p.pos + tok;
    uint blocks = limit / BC + 1;

    for (uint b = 0; b < blocks; b++) {
        uint j0 = b * BC;
        // SPLIT partial sums over d per score matrix, so more MACs are in
        // flight than the BC / 8 chains alone.
        simdgroup_float8x8 sp[SPLIT][BC / 8];
        for (uint u = 0; u < SPLIT; u++) {
            for (uint f = 0; f < BC / 8; f++) {
                sp[u][f] = simdgroup_float8x8(0);
            }
        }
        for (uint d0 = 0; d0 < AF_DIM / 8; d0 += SPLIT) {
            for (uint u = 0; u < SPLIT; u++) {
                uint d = d0 + u;
                simdgroup_half8x8 qf;
                if (QREG) {
                    qf = qr[d];
                } else {
                    simdgroup_load(qf, qp + 8 * d, AF_DIM);
                }
                for (uint f = 0; f < BC / 8; f++) {
                    simdgroup_half8x8 kf;
                    simdgroup_load(kf, kb + ulong(j0 + 8 * f) * stride + 8 * d, stride, ulong2(0, 0), true);
                    simdgroup_multiply_accumulate(sp[u][f], qf, kf, sp[u][f]);
                }
            }
        }
        simdgroup_float8x8 sf[BC / 8];
        for (uint f = 0; f < BC / 8; f++) {
            sf[f] = sp[0][f];
            for (uint u = 1; u < SPLIT; u++) {
                thread auto &a = sf[f].thread_elements();
                thread auto &b = sp[u][f].thread_elements();
                a[0] += b[0];
                a[1] += b[1];
            }
        }

        float bm = -INFINITY;
        for (uint f = 0; f < BC / 8; f++) {
            thread auto &e = sf[f].thread_elements();
            for (uint i = 0; i < 2; i++) {
                uint key = j0 + 8 * f + fr.y + i;
                e[i] = key <= limit ? e[i] * sl2 : -INFINITY;
                bm = max(bm, e[i]);
            }
        }
        bm = max(bm, simd_shuffle_xor(bm, 1));
        bm = max(bm, simd_shuffle_xor(bm, 8));
        float mn = max(m, bm);
        float a = exp2(m - mn);
        float rs = 0;
        simdgroup_half8x8 pf[BC / 8];
        for (uint f = 0; f < BC / 8; f++) {
            thread auto &e = sf[f].thread_elements();
            thread auto &h = pf[f].thread_elements();
            float p0 = exp2(e[0] - mn), p1 = exp2(e[1] - mn);
            rs += p0 + p1;
            h[0] = half(p0);
            h[1] = half(p1);
        }
        rs += simd_shuffle_xor(rs, 1);
        rs += simd_shuffle_xor(rs, 8);
        l = l * a + rs;
        m = mn;

        for (uint j = 0; j < AF_DIM / 8; j++) {
            thread auto &e = o[j].thread_elements();
            e[0] *= a;
            e[1] *= a;
            for (uint f = 0; f < BC / 8; f++) {
                simdgroup_half8x8 vf;
                simdgroup_load(vf, vb + ulong(j0 + 8 * f) * stride + 8 * j, stride);
                simdgroup_multiply_accumulate(o[j], pf[f], vf, o[j]);
            }
        }
    }

    uint h = kv * 8 + fr.x;
    ulong row = ulong(tok) * p.heads + h;
    float inv = 1.0f / l;
    for (uint j = 0; j < AF_DIM / 8; j++) {
        uint d = 8 * j + fr.y;
        device const float *g = q_and_gate + row * 2 * AF_DIM + AF_DIM + d;
        thread auto &e = o[j].thread_elements();
        float2 v = float2(e[0], e[1]) * inv;
        *(device float2 *)(out + row * AF_DIM + d) = v * float2(sigmoid(g[0]), sigmoid(g[1]));
    }
}

#define FLASH_REG(name, qreg, bc, split) \
    template [[host_name(name)]] kernel decltype(attention_flash_reg<qreg, bc, split>) attention_flash_reg<qreg, bc, split>;
FLASH_REG("attention_flash_reg_64", false, 64, 1)
FLASH_REG("attention_flash_reg_64s2", false, 64, 2)
FLASH_REG("attention_flash_reg_64s4", false, 64, 4)
FLASH_REG("attention_flash_reg_32s4", false, 32, 4)
FLASH_REG("attention_flash_reg_q64s2", true, 64, 2)

// attention_g: attention_rows2 from prefill.metal (simdgroup = 8 tokens of
// one head, scores and P through threadgroup memory), half only, with the
// scale folded into exp2 and the block length and heads per threadgroup as
// parameters, for tuning.
// GQA: a simdgroup takes the 8 q heads of one kv head for one token instead
// of 8 tokens of one head; the threadgroup takes SIMDS tokens of a kv head.
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
ATT_G("attention_g4_64", 4, 64, false)
ATT_G("attention_g8_64", 8, 64, false)
ATT_G("attention_g4_32", 4, 32, false)
ATT_G("attention_g4_128", 4, 128, false)
ATT_G("attention_gq4_64", 4, 64, true)
ATT_G("attention_gq8_64", 8, 64, true)
ATT_G("attention_gq4_128", 4, 128, true)
ATT_G("attention_g2_128", 2, 128, false)
ATT_G("attention_g2_256", 2, 256, false)
ATT_G("attention_gq2_128", 2, 128, true)
ATT_G("attention_gq2_256", 2, 256, true)
ATT_G("attention_gq1_256", 1, 256, true)

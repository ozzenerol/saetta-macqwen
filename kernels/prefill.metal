// Prefill: a chunk of T tokens per dispatch. src/prefill.rs runs the model
// layer by layer over the whole prompt, so each layer's weights are read
// from memory once per chunk instead of once per token. Uses unpack() from
// quant.metal and sigmoid(), silu() from layers.metal.

#include <metal_simdgroup_matrix>

// ---------------------------------------------------------------------------
// Quantized matrix times matrix: y[m, n] = sum over k of x[m, k] * W[n, k].
// A threadgroup of 4 simdgroups computes a 64 x 64 tile of y; each simdgroup
// computes 32 x 32 of it with 8 x 8 simdgroup matrices. Per step of 32 along
// k, the x tile is copied and the W tile dequantized to f32 in threadgroup
// memory.
//
// In gather mode the tile rows come from `tiles` (expert, first row, rows),
// made by moe_group, so one dispatch runs every expert on its own tokens.
// Output row i of a tile is y[first + i]; its input row is x[rows[first + i]]
// with gather 1, or x[first + i] with gather 2.

constant uint MM_BM = 64;
constant uint MM_BN = 64;
constant uint MM_BK = 32;
// Row stride of the x and W tiles; the padding avoids bank conflicts.
constant uint MM_LD = MM_BK + 4;

struct MmShape {
    uint m;
    uint n;
    uint k;
    uint gather;
};

// Dequantizes values OFF .. OFF + 15 of the 32 whose BITS packed words
// start at wk. OFF is a template argument so `packed` stays in registers.
template <uint BITS, uint OFF>
inline void dequant16(device const uint *wk, float scale, float bias, threadgroup float *out) {
    uint packed[BITS];
    for (uint j = 0; j < BITS; j++) {
        packed[j] = wk[j];
    }
#pragma clang loop unroll(full)
    for (uint i = 0; i < 16; i++) {
        out[i] = scale * float(unpack<BITS>(packed, OFF + i)) + bias;
    }
}

template <uint BITS>
kernel void qmm(device const uint *words [[buffer(0)]],
                device const ushort *scales [[buffer(1)]],
                device const ushort *biases [[buffer(2)]],
                device const float *x [[buffer(3)]],
                device float *y [[buffer(4)]],
                constant MmShape &s [[buffer(5)]],
                device const uint4 *tiles [[buffer(6)]],
                device const uint *rows [[buffer(7)]],
                device const uint *tile_count [[buffer(8)]],
                uint2 tg [[threadgroup_position_in_grid]],
                uint tid [[thread_index_in_threadgroup]],
                uint simd [[simdgroup_index_in_threadgroup]]) {
    threadgroup float smem[MM_BM * MM_LD + MM_BN * MM_LD];
    threadgroup float *xs = smem;
    threadgroup float *ws = smem + MM_BM * MM_LD;

    uint expert = 0, first, count;
    if (s.gather != 0) {
        if (tg.y >= tile_count[0]) {
            return;
        }
        uint4 t = tiles[tg.y];
        expert = t.x;
        first = t.y;
        count = t.z;
    } else {
        first = tg.y * MM_BM;
        count = min(MM_BM, s.m - first);
    }
    uint n0 = tg.x * MM_BN;

    // Each thread copies 16 values of one x row and dequantizes 16 values
    // of one W row per step. Simdgroups 0 and 1 take the first 16 of the 32
    // columns, so the branch in dequant16 is the same across a simdgroup.
    uint lr = tid % 64;
    uint lc = (tid / 64) * 16;
    bool x_ok = lr < count;
    device const float *xrow = x;
    if (x_ok) {
        uint r = s.gather == 1 ? rows[first + lr] : first + lr;
        xrow = x + ulong(r) * s.k + lc;
    }
    uint wr = n0 + lr;
    bool w_ok = wr < s.n;
    ulong wrow = ulong(expert) * s.n + (w_ok ? wr : 0);
    uint row_words = s.k * BITS / 32;
    uint groups = s.k / 64;
    device const uint *wp = words + wrow * row_words;
    device const ushort *sp = scales + wrow * groups;
    device const ushort *bp = biases + wrow * groups;

    uint sm = (simd / 2) * 32;
    uint sn = (simd % 2) * 32;
    simdgroup_float8x8 acc[4][4];
    for (uint i = 0; i < 4; i++) {
        for (uint j = 0; j < 4; j++) {
            acc[i][j] = simdgroup_float8x8(0);
        }
    }

    for (uint k0 = 0; k0 < s.k; k0 += MM_BK) {
        threadgroup float *xd = xs + lr * MM_LD + lc;
        if (x_ok) {
            device const float4 *p = (device const float4 *)(xrow + k0);
            for (uint i = 0; i < 4; i++) {
                float4 v = p[i];
                xd[4 * i] = v.x;
                xd[4 * i + 1] = v.y;
                xd[4 * i + 2] = v.z;
                xd[4 * i + 3] = v.w;
            }
        } else {
            for (uint i = 0; i < 16; i++) {
                xd[i] = 0;
            }
        }

        threadgroup float *wd = ws + lr * MM_LD + lc;
        if (w_ok) {
            device const uint *wk = wp + k0 / 32 * BITS;
            uint g = (k0 + lc) / 64;
            float scale = bf16_to_float(sp[g]);
            float bias = bf16_to_float(bp[g]);
            if (lc == 0) {
                dequant16<BITS, 0>(wk, scale, bias, wd);
            } else {
                dequant16<BITS, 16>(wk, scale, bias, wd);
            }
        } else {
            for (uint i = 0; i < 16; i++) {
                wd[i] = 0;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint kk = 0; kk < MM_BK; kk += 8) {
            simdgroup_float8x8 a[4];
            simdgroup_float8x8 b[4];
            for (uint i = 0; i < 4; i++) {
                simdgroup_load(a[i], xs + (sm + 8 * i) * MM_LD + kk, MM_LD);
                simdgroup_load(b[i], ws + (sn + 8 * i) * MM_LD + kk, MM_LD, ulong2(0, 0), true);
            }
            for (uint i = 0; i < 4; i++) {
                for (uint j = 0; j < 4; j++) {
                    simdgroup_multiply_accumulate(acc[i][j], a[i], b[j], acc[i][j]);
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // The 64 x 64 result goes through threadgroup memory so partial tiles
    // can be written with bounds checks.
    threadgroup float *c = smem;
    for (uint i = 0; i < 4; i++) {
        for (uint j = 0; j < 4; j++) {
            simdgroup_store(acc[i][j], c + (sm + 8 * i) * MM_BN + sn + 8 * j, MM_BN);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint oc = (tid / 64) * 32;
    if (lr < count) {
        device float *yrow = y + ulong(first + lr) * s.n;
        for (uint i = 0; i < 32; i++) {
            uint col = n0 + oc + i;
            if (col < s.n) {
                yrow[col] = c[lr * MM_BN + oc + i];
            }
        }
    }
}

template [[host_name("qmm_5bit")]] kernel decltype(qmm<5>) qmm<5>;
template [[host_name("qmm_8bit")]] kernel decltype(qmm<8>) qmm<8>;

// ---------------------------------------------------------------------------
// Small batched ops.

struct EmbedParams {
    uint cols;
};

// out[t] = dequantized row tokens[t] of the embedding, one thread per value.
template <uint BITS>
kernel void embed_rows(device const uint *words [[buffer(0)]],
                       device const ushort *scales [[buffer(1)]],
                       device const ushort *biases [[buffer(2)]],
                       device const uint *tokens [[buffer(3)]],
                       device float *out [[buffer(4)]],
                       constant EmbedParams &p [[buffer(5)]],
                       uint2 id [[thread_position_in_grid]]) {
    uint i = id.x;
    if (i >= p.cols) {
        return;
    }
    ulong row = tokens[id.y];
    device const uint *w = words + row * (p.cols * BITS / 32) + i / 32 * BITS;
    uint packed[BITS];
    for (uint j = 0; j < BITS; j++) {
        packed[j] = w[j];
    }
    ulong g = row * (p.cols / 64) + i / 64;
    float q = float(unpack<BITS>(packed, i % 32));
    out[ulong(id.y) * p.cols + i] = bf16_to_float(scales[g]) * q + bf16_to_float(biases[g]);
}

template [[host_name("embed_rows_5bit")]] kernel decltype(embed_rows<5>) embed_rows<5>;
template [[host_name("embed_rows_8bit")]] kernel decltype(embed_rows<8>) embed_rows<8>;

kernel void add_in_place(device float *x [[buffer(0)]],
                         device const float *y [[buffer(1)]],
                         uint i [[thread_position_in_grid]]) {
    x[i] += y[i];
}

// g = silu(g) * u
kernel void silu_mul(device float *g [[buffer(0)]],
                     device const float *u [[buffer(1)]],
                     uint i [[thread_position_in_grid]]) {
    g[i] = silu(g[i]) * u[i];
}

struct RopeRowsParams {
    uint stride;     // floats between heads
    uint half_dims;  // rotary dims / 2
    uint pos;        // position of the first token
    float log2_theta;
    uint heads;      // heads per token
};

// As rope in layers.metal, for row id.y = token * heads + head.
kernel void rope_rows(device float *x [[buffer(0)]],
                      constant RopeRowsParams &p [[buffer(1)]],
                      uint2 id [[thread_position_in_grid]]) {
    uint i = id.x;
    if (i >= p.half_dims) {
        return;
    }
    device float *h = x + ulong(id.y) * p.stride;
    float freq = exp2(-float(i) / float(p.half_dims) * p.log2_theta);
    float angle = float(p.pos + id.y / p.heads) * freq;
    float s = precise::sin(angle);
    float c = precise::cos(angle);
    float a = h[i];
    float b = h[i + p.half_dims];
    h[i] = a * c - b * s;
    h[i + p.half_dims] = a * s + b * c;
}

struct ConvRowsParams {
    uint channels;
    uint kernel_size;
    uint tokens;
};

// deltanet_conv over `tokens` tokens: one thread per channel walks the
// tokens in order. x and out are [tokens, channels].
kernel void conv_rows(device const float *x [[buffer(0)]],
                      device const float *w [[buffer(1)]],
                      device float *state [[buffer(2)]],
                      device float *out [[buffer(3)]],
                      constant ConvRowsParams &p [[buffer(4)]],
                      uint c [[thread_position_in_grid]]) {
    if (c >= p.channels) {
        return;
    }
    uint past = p.kernel_size - 1;
    device const float *wc = w + c * p.kernel_size;
    float hist[8];
    for (uint t = 0; t < past; t++) {
        hist[t] = state[t * p.channels + c];
    }
    for (uint t = 0; t < p.tokens; t++) {
        float xt = x[ulong(t) * p.channels + c];
        float acc = 0;
        for (uint j = 0; j < past; j++) {
            acc += wc[j] * hist[j];
        }
        acc += wc[past] * xt;
        for (uint j = 0; j + 1 < past; j++) {
            hist[j] = hist[j + 1];
        }
        hist[past - 1] = xt;
        out[ulong(t) * p.channels + c] = silu(acc);
    }
    for (uint t = 0; t < past; t++) {
        state[t * p.channels + c] = hist[t];
    }
}

// ---------------------------------------------------------------------------
// Gated delta rule over T tokens, as the packed kernel in MLX gated_delta.py:
// four lanes own one row of a value head's [v_dim, k_dim] state and keep 32
// values of it in registers, so a simdgroup holds 8 rows. The q and k L2
// norm and the gates g and beta are computed here from the conv output.
// Needs k_dim 128.

struct DeltaRowsParams {
    uint k_heads;
    uint v_heads;
    uint v_dim;
    uint tokens;
    uint channels;  // floats per token in `mixed`: q, k and v
    float eps;
};

kernel void delta_rows(device const float *mixed [[buffer(0)]],
                       device const float *a [[buffer(1)]],
                       device const float *b [[buffer(2)]],
                       device const float *a_log [[buffer(3)]],
                       device const float *dt_bias [[buffer(4)]],
                       device float *state [[buffer(5)]],
                       device float *y [[buffer(6)]],
                       constant DeltaRowsParams &p [[buffer(7)]],
                       uint2 tg [[threadgroup_position_in_grid]],
                       uint simd [[simdgroup_index_in_threadgroup]],
                       uint simds [[simdgroups_per_threadgroup]],
                       uint lane [[thread_index_in_simdgroup]]) {
    constexpr uint K_DIM = 128;
    constexpr uint PER_LANE = 32;
    uint h = tg.y;
    uint row = (tg.x * simds + simd) * 8 + lane / 4;
    uint part = lane % 4;
    if (row >= p.v_dim) {
        return;
    }
    uint kh = h / (p.v_heads / p.k_heads);
    uint key_size = p.k_heads * K_DIM;
    uint value_size = p.v_heads * p.v_dim;

    float exp_a = exp(a_log[h]);
    float bias = dt_bias[h];
    float q_scale = 1.0f / sqrt(float(K_DIM));

    device float *sp = state + (ulong(h) * p.v_dim + row) * K_DIM + part * PER_LANE;
    float st[PER_LANE];
    for (uint i = 0; i < PER_LANE; i++) {
        st[i] = sp[i];
    }

    for (uint t = 0; t < p.tokens; t++) {
        device const float *tok = mixed + ulong(t) * p.channels;
        device const float *qp = tok + kh * K_DIM + part * PER_LANE;
        device const float *kp = tok + key_size + kh * K_DIM + part * PER_LANE;
        float qv[PER_LANE];
        float kv[PER_LANE];
        float qq = 0, kk = 0;
        for (uint i = 0; i < PER_LANE; i++) {
            qv[i] = qp[i];
            kv[i] = kp[i];
            qq += qv[i] * qv[i];
            kk += kv[i] * kv[i];
        }
        qq += simd_shuffle_xor(qq, 1);
        qq += simd_shuffle_xor(qq, 2);
        kk += simd_shuffle_xor(kk, 1);
        kk += simd_shuffle_xor(kk, 2);
        float q_inv = q_scale / sqrt(qq + p.eps);
        float k_inv = 1.0f / sqrt(kk + p.eps);

        float x = a[t * p.v_heads + h] + bias;
        float softplus = max(x, 0.0f) + log(1.0f + exp(-abs(x)));
        float g = exp(-exp_a * softplus);
        float beta = sigmoid(b[t * p.v_heads + h]);

        float mem = 0;
        for (uint i = 0; i < PER_LANE; i++) {
            kv[i] *= k_inv;
            st[i] *= g;
            mem += st[i] * kv[i];
        }
        mem += simd_shuffle_xor(mem, 1);
        mem += simd_shuffle_xor(mem, 2);
        float v = tok[2 * key_size + h * p.v_dim + row];
        float delta = (v - mem) * beta;
        float out = 0;
        for (uint i = 0; i < PER_LANE; i++) {
            st[i] += kv[i] * delta;
            out += st[i] * qv[i];
        }
        out += simd_shuffle_xor(out, 1);
        out += simd_shuffle_xor(out, 2);
        if (part == 0) {
            y[ulong(t) * value_size + h * p.v_dim + row] = out * q_inv;
        }
    }
    for (uint i = 0; i < PER_LANE; i++) {
        sp[i] = st[i];
    }
}

// ---------------------------------------------------------------------------
// Causal attention of T new tokens over the KV cache, which already holds
// them at positions pos .. pos + T - 1. A simdgroup takes 8 query tokens of
// one head and walks the keys in blocks of 32 with an online softmax, using
// 8 x 8 simdgroup matrices for q K^T and P V. The 4 simdgroups of a
// threadgroup are 4 heads of the same kv head, so they share K and V loads.
// Output is softmax(q K^T * scale) V * sigmoid(gate), [T, heads * dim].
// Needs dim 256. q has room for T rounded up to 8 rows, the cache for
// pos + T rounded up to 32 rows, and cache rows past pos + T must be finite.

struct PrefillAttentionParams {
    uint heads;
    uint kv_heads;
    uint pos;
    uint tokens;
    float scale;
};

constant uint ATT_DIM = 256;
constant uint ATT_KB = 32;
constant uint ATT_SIMDS = 4;

kernel void attention_rows(device const float *q [[buffer(0)]],
                           device const float *keys [[buffer(1)]],
                           device const float *values [[buffer(2)]],
                           device const float *q_and_gate [[buffer(3)]],
                           device float *out [[buffer(4)]],
                           constant PrefillAttentionParams &p [[buffer(5)]],
                           uint2 tg [[threadgroup_position_in_grid]],
                           uint simd [[simdgroup_index_in_threadgroup]],
                           uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float s_all[ATT_SIMDS][8 * ATT_KB];
    threadgroup float d_all[ATT_SIMDS][64];
    threadgroup float *sc = s_all[simd];
    threadgroup float *dg = d_all[simd];

    uint h = tg.y * ATT_SIMDS + simd;
    uint kv = h / (p.heads / p.kv_heads);
    uint q0 = tg.x * 8;
    uint q_stride = p.heads * ATT_DIM;
    uint kv_stride = p.kv_heads * ATT_DIM;
    device const float *qh = q + ulong(q0) * q_stride + h * ATT_DIM;
    device const float *kh = keys + kv * ATT_DIM;
    device const float *vh = values + kv * ATT_DIM;
    uint end = p.pos + min(q0 + 8, p.tokens);

    // Lane l handles row l / 4, columns (l % 4) * 8 .. + 8 of the scores.
    uint r = lane / 4;
    uint c0 = (lane % 4) * 8;
    uint last = p.pos + q0 + r;  // the newest key this row may see
    float m = -INFINITY;
    float l = 0;

    simdgroup_float8x8 o[ATT_DIM / 8];
    for (uint i = 0; i < ATT_DIM / 8; i++) {
        o[i] = simdgroup_float8x8(0);
    }
    for (uint i = lane; i < 64; i += 32) {
        dg[i] = 0;
    }

    for (uint k0 = 0; k0 < end; k0 += ATT_KB) {
        simdgroup_float8x8 s[ATT_KB / 8];
        for (uint j = 0; j < ATT_KB / 8; j++) {
            s[j] = simdgroup_float8x8(0);
        }
        for (uint d = 0; d < ATT_DIM; d += 8) {
            simdgroup_float8x8 qm;
            simdgroup_load(qm, qh + d, q_stride);
            for (uint j = 0; j < ATT_KB / 8; j++) {
                simdgroup_float8x8 km;
                simdgroup_load(km, kh + ulong(k0 + 8 * j) * kv_stride + d, kv_stride, ulong2(0, 0),
                               true);
                simdgroup_multiply_accumulate(s[j], qm, km, s[j]);
            }
        }
        for (uint j = 0; j < ATT_KB / 8; j++) {
            simdgroup_store(s[j], sc + 8 * j, ATT_KB);
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);

        float v[8];
        float bm = -INFINITY;
        for (uint i = 0; i < 8; i++) {
            uint key = k0 + c0 + i;
            v[i] = key <= last && key < end ? sc[r * ATT_KB + c0 + i] * p.scale : -INFINITY;
            bm = max(bm, v[i]);
        }
        bm = max(bm, simd_shuffle_xor(bm, 1));
        bm = max(bm, simd_shuffle_xor(bm, 2));
        float m_new = max(m, bm);
        float alpha = exp(m - m_new);
        float sum = 0;
        for (uint i = 0; i < 8; i++) {
            float e = exp(v[i] - m_new);
            sc[r * ATT_KB + c0 + i] = e;
            sum += e;
        }
        sum += simd_shuffle_xor(sum, 1);
        sum += simd_shuffle_xor(sum, 2);
        l = l * alpha + sum;
        m = m_new;
        if (lane % 4 == 0) {
            dg[r * 8 + r] = alpha;
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);

        simdgroup_float8x8 scale_rows;
        simdgroup_load(scale_rows, dg, 8);
        simdgroup_float8x8 pm[ATT_KB / 8];
        for (uint j = 0; j < ATT_KB / 8; j++) {
            simdgroup_load(pm[j], sc + 8 * j, ATT_KB);
        }
        for (uint d = 0; d < ATT_DIM / 8; d++) {
            simdgroup_multiply(o[d], scale_rows, o[d]);
            for (uint j = 0; j < ATT_KB / 8; j++) {
                simdgroup_float8x8 vm;
                simdgroup_load(vm, vh + ulong(k0 + 8 * j) * kv_stride + 8 * d, kv_stride);
                simdgroup_multiply_accumulate(o[d], pm[j], vm, o[d]);
            }
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Lane l writes row l / 4, two values per 8 x 8 block.
    float inv = 1.0f / l;
    uint row_t = q0 + r;
    for (uint d = 0; d < ATT_DIM / 8; d++) {
        simdgroup_store(o[d], sc, 8);
        simdgroup_barrier(mem_flags::mem_threadgroup);
        if (row_t < p.tokens) {
            for (uint i = (lane % 4) * 2; i < (lane % 4) * 2 + 2; i++) {
                uint col = 8 * d + i;
                float gate = q_and_gate[ulong(row_t) * 2 * q_stride + h * 2 * ATT_DIM + ATT_DIM + col];
                out[ulong(row_t) * q_stride + h * ATT_DIM + col] = sc[r * 8 + i] * inv * sigmoid(gate);
            }
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
    }
}

// ---------------------------------------------------------------------------
// MoE for T tokens.

struct RouteParams {
    uint experts;  // 256: 8 per lane
    uint top_k;
};

// One simdgroup per token: softmax over the router logits, the top_k best
// (lower index first on ties, as the stable sort in src/cpu/moe.rs) and
// their weights rescaled to sum to 1.
kernel void moe_route(device const float *logits [[buffer(0)]],
                      device uint *ids [[buffer(1)]],
                      device float *weights [[buffer(2)]],
                      constant RouteParams &p [[buffer(3)]],
                      uint t [[threadgroup_position_in_grid]],
                      uint lane [[thread_index_in_simdgroup]]) {
    constexpr uint PER_LANE = 8;
    device const float *lt = logits + ulong(t) * p.experts;
    float v[PER_LANE];
    float mx = -INFINITY;
    for (uint i = 0; i < PER_LANE; i++) {
        v[i] = lt[lane * PER_LANE + i];
        mx = max(mx, v[i]);
    }
    mx = simd_max(mx);
    float sum = 0;
    for (uint i = 0; i < PER_LANE; i++) {
        v[i] = exp(v[i] - mx);
        sum += v[i];
    }
    sum = simd_sum(sum);
    for (uint i = 0; i < PER_LANE; i++) {
        v[i] /= sum;
    }

    // Lane k keeps the k-th choice.
    float chosen = 0;
    float mine = 0;
    uint mine_id = 0;
    for (uint k = 0; k < p.top_k; k++) {
        float best = -1;
        uint best_i = 0;
        for (uint i = 0; i < PER_LANE; i++) {
            if (v[i] > best) {
                best = v[i];
                best_i = i;
            }
        }
        float top = simd_max(best);
        uint index = simd_min(best == top ? lane * PER_LANE + best_i : 0xffffffffu);
        if (lane == index / PER_LANE) {
            v[index % PER_LANE] = -2;
        }
        if (lane == k) {
            mine = top;
            mine_id = index;
        }
        chosen += top;
    }
    if (lane < p.top_k) {
        ids[t * p.top_k + lane] = mine_id;
        weights[t * p.top_k + lane] = mine / chosen;
    }
}

struct GroupParams {
    uint pairs;  // tokens * top_k
    uint top_k;
    uint experts;
};

// One threadgroup sorts the (token, slot) pairs by expert. For each pair i:
// order[i] is its row in expert order, and rows[order[i]] its token. Tiles of
// up to MM_BM rows of one expert go to `tiles` for qmm in gather mode, with
// their number in tile_count[0]. Pairs of one expert end up in any order,
// which does not change any result.
kernel void moe_group(device const uint *ids [[buffer(0)]],
                      device uint *order [[buffer(1)]],
                      device uint *rows [[buffer(2)]],
                      device uint4 *tiles [[buffer(3)]],
                      device uint *tile_count [[buffer(4)]],
                      constant GroupParams &p [[buffer(5)]],
                      uint tid [[thread_index_in_threadgroup]],
                      uint threads [[threads_per_threadgroup]]) {
    threadgroup atomic_uint counts[256];
    threadgroup uint offsets[256];
    for (uint e = tid; e < p.experts; e += threads) {
        atomic_store_explicit(&counts[e], 0, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = tid; i < p.pairs; i += threads) {
        atomic_fetch_add_explicit(&counts[ids[i]], 1, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        uint offset = 0;
        uint n = 0;
        for (uint e = 0; e < p.experts; e++) {
            uint count = atomic_load_explicit(&counts[e], memory_order_relaxed);
            offsets[e] = offset;
            for (uint s = 0; s < count; s += MM_BM) {
                tiles[n++] = uint4(e, offset + s, min(MM_BM, count - s), 0);
            }
            offset += count;
            atomic_store_explicit(&counts[e], 0, memory_order_relaxed);
        }
        tile_count[0] = n;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = tid; i < p.pairs; i += threads) {
        uint e = ids[i];
        uint at = offsets[e] + atomic_fetch_add_explicit(&counts[e], 1, memory_order_relaxed);
        order[i] = at;
        rows[at] = i / p.top_k;
    }
}

struct CombineParams {
    uint dim;
    uint top_k;
};

// x[t] += sigmoid(shared_gate[t]) * shared[t] + sum over slots of
// weight * expert_out[order[t, slot]], one thread per value.
kernel void moe_combine(device float *x [[buffer(0)]],
                        device const float *shared [[buffer(1)]],
                        device const float *shared_gate [[buffer(2)]],
                        device const float *expert_out [[buffer(3)]],
                        device const uint *order [[buffer(4)]],
                        device const float *weights [[buffer(5)]],
                        constant CombineParams &p [[buffer(6)]],
                        uint2 id [[thread_position_in_grid]]) {
    uint i = id.x;
    uint t = id.y;
    if (i >= p.dim) {
        return;
    }
    ulong at = ulong(t) * p.dim + i;
    float v = sigmoid(shared_gate[t]) * shared[at];
    for (uint s = 0; s < p.top_k; s++) {
        uint slot = t * p.top_k + s;
        v += weights[slot] * expert_out[ulong(order[slot]) * p.dim + i];
    }
    x[at] += v;
}

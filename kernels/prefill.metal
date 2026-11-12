// Prefill: a chunk of T tokens per dispatch. src/prefill.rs runs the model
// layer by layer over the whole prompt, so each layer's weights are read
// from memory once per chunk instead of once per token. Uses unpack() from
// quant.metal and sigmoid(), silu() from layers.metal.

#include <metal_simdgroup_matrix>

// ---------------------------------------------------------------------------
// Quantized matrix times matrix: y[m, n] = sum over k of x[m, k] * W[n, k].
// A threadgroup of 4 simdgroups computes a 64 x 64 tile of y; each simdgroup
// computes 32 x 32 of it with 8 x 8 simdgroup matrices. f32 matrix MACs are
// about 4 times slower than half ones on the M3 Max, so the products run in
// half with f32 sums: the integer q of each weight is exact in half, and per
// group of 64 the f32 sum of q * x is scaled and the bias added in f32:
//   sum(w * x) = scale * sum(q * x) + bias * sum(x).
// x is rounded to half (SPLIT 1), or split into a half and the half of its
// rest (SPLIT 2) for close to f32 results at twice the MACs.
//
// In gather mode the tile rows come from `tiles` (expert, first row, rows),
// made by moe_group, so one dispatch runs every expert on its own tokens.
// Output row i of a tile is y[first + i]; its input row is x[rows[first + i]]
// with gather 1, or x[first + i] with gather 2.

constant uint PREFILL_MM_BM = 64;
// Tiles of this many rows or fewer go to qmm_few.
constant uint PREFILL_FEW = 8;
constant uint PREFILL_MM_BN = 64;
constant uint PREFILL_MM_BK = 64;  // one quantization group per step
// Row stride of the tiles in halves; the padding avoids bank conflicts.
constant uint PREFILL_MM_LD = PREFILL_MM_BK + 8;

struct MmShape {
    uint m;
    uint n;
    uint k;
    uint gather;
};

struct MmTiles {
    half x_hi[PREFILL_MM_BM * PREFILL_MM_LD];
    half x_lo[PREFILL_MM_BM * PREFILL_MM_LD];
    half q[PREFILL_MM_BN * PREFILL_MM_LD];
    float x_sum[2][PREFILL_MM_BM];
    float scale[PREFILL_MM_BN];
    float bias[PREFILL_MM_BN];
};

// Row and first column of the two values a lane holds in an 8 x 8
// simdgroup matrix, as MLX steel uses them.
inline uint2 frag_pos(uint lane) {
    uint qid = lane / 4;
    return uint2((qid & 4) + (lane / 2) % 4, (qid & 2) * 2 + (lane % 2) * 2);
}

template <uint BITS, uint SPLIT>
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
                uint simd [[simdgroup_index_in_threadgroup]],
                uint lane [[thread_index_in_simdgroup]]) {
    threadgroup MmTiles t;

    uint expert = 0, first, count;
    if (s.gather != 0) {
        if (tg.y >= tile_count[0]) {
            return;
        }
        uint4 tile = tiles[tg.y];
        if (tile.z <= PREFILL_FEW) {
            return;  // qmm_few takes it
        }
        expert = tile.x;
        first = tile.y;
        count = tile.z;
    } else {
        first = tg.y * PREFILL_MM_BM;
        count = min(PREFILL_MM_BM, s.m - first);
    }
    uint n0 = tg.x * PREFILL_MM_BN;

    // Each thread loads 32 values of one x row and unpacks 32 values of
    // one W row per step.
    uint lr = tid % 64;
    uint half_k = tid / 64;
    uint lc = half_k * 32;
    bool x_ok = lr < count;
    device const float *xrow = x;
    if (x_ok) {
        uint r = s.gather == 1 ? rows[first + lr] : first + lr;
        xrow = x + ulong(r) * s.k + lc;
    }
    uint wr = n0 + lr;
    bool w_ok = wr < s.n;
    ulong wrow = ulong(expert) * s.n + (w_ok ? wr : 0);
    uint groups = s.k / 64;
    device const uint *wp = words + wrow * (s.k * BITS / 32) + half_k * BITS;
    device const ushort *sp = scales + wrow * groups;
    device const ushort *bp = biases + wrow * groups;

    uint sm = (simd / 2) * 32;
    uint sn = (simd % 2) * 32;
    uint2 fp = frag_pos(lane);
    simdgroup_float8x8 acc[4][4];
    for (uint i = 0; i < 4; i++) {
        for (uint j = 0; j < 4; j++) {
            acc[i][j] = simdgroup_float8x8(0);
        }
    }

    for (uint k0 = 0; k0 < s.k; k0 += PREFILL_MM_BK) {
        threadgroup half *xh = t.x_hi + lr * PREFILL_MM_LD + lc;
        threadgroup half *xl = t.x_lo + lr * PREFILL_MM_LD + lc;
        float sum = 0;
        if (x_ok) {
            device const float4 *p = (device const float4 *)(xrow + k0);
            for (uint i = 0; i < 8; i++) {
                float4 v = p[i];
                half4 hi = half4(v);
                // The sum must be of the values the MACs use: scale * sum(q x)
                // and bias * sum(x) mostly cancel, so a mismatch is amplified.
                float4 used = float4(hi);
                *(threadgroup half4 *)(xh + 4 * i) = hi;
                if (SPLIT == 2) {
                    half4 lo = half4(v - float4(hi));
                    *(threadgroup half4 *)(xl + 4 * i) = lo;
                    used += float4(lo);
                }
                sum += used.x + used.y + used.z + used.w;
            }
        } else {
            for (uint i = 0; i < 8; i++) {
                *(threadgroup half4 *)(xh + 4 * i) = half4(0);
                if (SPLIT == 2) {
                    *(threadgroup half4 *)(xl + 4 * i) = half4(0);
                }
            }
        }
        t.x_sum[half_k][lr] = sum;

        threadgroup half *qd = t.q + lr * PREFILL_MM_LD + lc;
        if (w_ok) {
            uint packed[BITS];
            device const uint *wk = wp + k0 / 32 * BITS;
            for (uint j = 0; j < BITS; j++) {
                packed[j] = wk[j];
            }
#pragma clang loop unroll(full)
            for (uint i = 0; i < 32; i++) {
                qd[i] = half(unpack<BITS>(packed, i));
            }
            if (half_k == 0) {
                t.scale[lr] = bf16_to_float(sp[k0 / 64]);
                t.bias[lr] = bf16_to_float(bp[k0 / 64]);
            }
        } else {
            for (uint i = 0; i < 32; i++) {
                qd[i] = 0;
            }
            if (half_k == 0) {
                t.scale[lr] = 0;
                t.bias[lr] = 0;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        simdgroup_float8x8 part[4][4];
        for (uint i = 0; i < 4; i++) {
            for (uint j = 0; j < 4; j++) {
                part[i][j] = simdgroup_float8x8(0);
            }
        }
        for (uint kk = 0; kk < PREFILL_MM_BK; kk += 8) {
            simdgroup_half8x8 a[4];
            simdgroup_half8x8 b[4];
            for (uint i = 0; i < 4; i++) {
                simdgroup_load(a[i], t.x_hi + (sm + 8 * i) * PREFILL_MM_LD + kk, PREFILL_MM_LD);
                simdgroup_load(b[i], t.q + (sn + 8 * i) * PREFILL_MM_LD + kk, PREFILL_MM_LD, ulong2(0, 0), true);
            }
            for (uint i = 0; i < 4; i++) {
                for (uint j = 0; j < 4; j++) {
                    simdgroup_multiply_accumulate(part[i][j], a[i], b[j], part[i][j]);
                }
            }
            if (SPLIT == 2) {
                for (uint i = 0; i < 4; i++) {
                    simdgroup_load(a[i], t.x_lo + (sm + 8 * i) * PREFILL_MM_LD + kk, PREFILL_MM_LD);
                }
                for (uint i = 0; i < 4; i++) {
                    for (uint j = 0; j < 4; j++) {
                        simdgroup_multiply_accumulate(part[i][j], a[i], b[j], part[i][j]);
                    }
                }
            }
        }

        // acc += scale[col] * part + bias[col] * x_sum[row]
        for (uint i = 0; i < 4; i++) {
            uint row = sm + 8 * i + fp.x;
            float xs = t.x_sum[0][row] + t.x_sum[1][row];
            for (uint j = 0; j < 4; j++) {
                uint col = sn + 8 * j + fp.y;
                thread auto &pe = part[i][j].thread_elements();
                thread auto &ae = acc[i][j].thread_elements();
                ae[0] += t.scale[col] * pe[0] + t.bias[col] * xs;
                ae[1] += t.scale[col + 1] * pe[1] + t.bias[col + 1] * xs;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // The result goes through threadgroup memory so partial tiles can be
    // written with bounds checks.
    threadgroup float *c = (threadgroup float *)t.x_hi;
    for (uint i = 0; i < 4; i++) {
        for (uint j = 0; j < 4; j++) {
            simdgroup_store(acc[i][j], c + (sm + 8 * i) * PREFILL_MM_BN + sn + 8 * j, PREFILL_MM_BN);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint oc = half_k * 32;
    if (lr < count) {
        device float *yrow = y + ulong(first + lr) * s.n;
        for (uint i = 0; i < 32; i++) {
            uint col = n0 + oc + i;
            if (col < s.n) {
                yrow[col] = c[lr * PREFILL_MM_BN + oc + i];
            }
        }
    }
}

template [[host_name("qmm_5bit")]] kernel decltype(qmm<5, 1>) qmm<5, 1>;
template [[host_name("qmm_8bit")]] kernel decltype(qmm<8, 1>) qmm<8, 1>;
template [[host_name("qmm_5bit_split")]] kernel decltype(qmm<5, 2>) qmm<5, 2>;
template [[host_name("qmm_8bit_split")]] kernel decltype(qmm<8, 2>) qmm<8, 2>;

// The same product with the weights dequantized to half in threadgroup
// memory (w = scale q + bias), as MLX does: one set of accumulators and no
// per-group fix-up, at the cost of w rounded to half. Same grid and tiles.
// Simdgroups 0 and 1 dequantize 32 weights of one W row per step, 2 and 3
// load 32 values of one x row.
constant uint PREFILL_DQ_BK = 32;
constant uint PREFILL_DQ_LD = PREFILL_DQ_BK + 8;

template <uint BITS>
kernel void qmm_dq(device const uint *words [[buffer(0)]],
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
                   uint simd [[simdgroup_index_in_threadgroup]],
                   uint lane [[thread_index_in_simdgroup]]) {
    threadgroup half xt[PREFILL_MM_BM * PREFILL_DQ_LD];
    threadgroup half wt[PREFILL_MM_BN * PREFILL_DQ_LD];
    threadgroup float edge[4][64];

    uint expert = 0, first, count;
    if (s.gather != 0) {
        if (tg.y >= tile_count[0]) {
            return;
        }
        uint4 tile = tiles[tg.y];
        if (tile.z <= PREFILL_FEW) {
            return;  // qmm_few takes it
        }
        expert = tile.x;
        first = tile.y;
        count = tile.z;
    } else {
        first = tg.y * PREFILL_MM_BM;
        count = min(PREFILL_MM_BM, s.m - first);
    }
    uint n0 = tg.x * PREFILL_MM_BN;

    bool loads_w = tid < 64;
    uint lr = tid % 64;
    bool ok;
    device const float *xrow = x;
    device const uint *wp = words;
    device const ushort *sp = scales;
    device const ushort *bp = biases;
    if (loads_w) {
        uint wr = n0 + lr;
        ok = wr < s.n;
        ulong wrow = ulong(expert) * s.n + (ok ? wr : 0);
        wp = words + wrow * (s.k * BITS / 32);
        sp = scales + wrow * (s.k / 64);
        bp = biases + wrow * (s.k / 64);
    } else {
        ok = lr < count;
        if (ok) {
            uint r = s.gather == 1 ? rows[first + lr] : first + lr;
            xrow = x + ulong(r) * s.k;
        }
    }
    threadgroup half4 *dst = (threadgroup half4 *)((loads_w ? wt : xt) + lr * PREFILL_DQ_LD);

    uint sm = (simd / 2) * 32;
    uint sn = (simd % 2) * 32;
    simdgroup_float8x8 acc[4][4];
    for (uint i = 0; i < 4; i++) {
        for (uint j = 0; j < 4; j++) {
            acc[i][j] = simdgroup_float8x8(0);
        }
    }

    for (uint k0 = 0; k0 < s.k; k0 += PREFILL_DQ_BK) {
        if (!ok) {
            for (uint i = 0; i < 8; i++) {
                dst[i] = half4(0);
            }
        } else if (loads_w) {
            uint packed[BITS];
            device const uint *wk = wp + k0 / 32 * BITS;
            for (uint j = 0; j < BITS; j++) {
                packed[j] = wk[j];
            }
            float scale = bf16_to_float(sp[k0 / 64]);
            float bias = bf16_to_float(bp[k0 / 64]);
#pragma clang loop unroll(full)
            for (uint i = 0; i < 8; i++) {
                float4 q = float4(unpack<BITS>(packed, 4 * i), unpack<BITS>(packed, 4 * i + 1),
                                  unpack<BITS>(packed, 4 * i + 2), unpack<BITS>(packed, 4 * i + 3));
                dst[i] = half4(q * scale + bias);
            }
        } else {
            device const float4 *p = (device const float4 *)(xrow + k0);
            for (uint i = 0; i < 8; i++) {
                dst[i] = half4(p[i]);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint kk = 0; kk < PREFILL_DQ_BK; kk += 8) {
            simdgroup_half8x8 a[4];
            simdgroup_half8x8 b[4];
            for (uint i = 0; i < 4; i++) {
                simdgroup_load(a[i], xt + (sm + 8 * i) * PREFILL_DQ_LD + kk, PREFILL_DQ_LD);
                simdgroup_load(b[i], wt + (sn + 8 * i) * PREFILL_DQ_LD + kk, PREFILL_DQ_LD, ulong2(0, 0), true);
            }
            for (uint i = 0; i < 4; i++) {
                for (uint j = 0; j < 4; j++) {
                    simdgroup_multiply_accumulate(acc[i][j], a[i], b[j], acc[i][j]);
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    device float *y0 = y + ulong(first) * s.n + n0;
    if (count == PREFILL_MM_BM && n0 + PREFILL_MM_BN <= s.n) {
        for (uint i = 0; i < 4; i++) {
            for (uint j = 0; j < 4; j++) {
                simdgroup_store(acc[i][j], y0 + ulong(sm + 8 * i) * s.n + sn + 8 * j, s.n);
            }
        }
        return;
    }
    // A partial tile goes out 8 x 8 at a time with bounds checks. Lane l
    // writes row l / 4, columns 2 (l % 4) and + 1.
    threadgroup float *e = edge[simd];
    uint er = lane / 4, ec = (lane % 4) * 2;
    for (uint i = 0; i < 4; i++) {
        for (uint j = 0; j < 4; j++) {
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

template [[host_name("qmm_dq_5bit")]] kernel decltype(qmm_dq<5>) qmm_dq<5>;
template [[host_name("qmm_dq_8bit")]] kernel decltype(qmm_dq<8>) qmm_dq<8>;

// The same product for few rows: up to PREFILL_FEW rows of x per
// threadgroup, in f32 as qmv.metal does, so weights are read once for all
// of them instead of computing a 64-row tile. A simdgroup computes 4 rows
// of y for every x row. In gather mode it takes the tiles of PREFILL_FEW
// rows or fewer, and qmm the others.

constant uint PREFILL_FEW_ROWS = 4;
constant uint PREFILL_FEW_SIMDS = 2;

template <uint BITS>
kernel void qmm_few(device const uint *words [[buffer(0)]],
                    device const ushort *scales [[buffer(1)]],
                    device const ushort *biases [[buffer(2)]],
                    device const float *x [[buffer(3)]],
                    device float *y [[buffer(4)]],
                    constant MmShape &s [[buffer(5)]],
                    device const uint4 *tiles [[buffer(6)]],
                    device const uint *rows [[buffer(7)]],
                    device const uint *tile_count [[buffer(8)]],
                    uint2 tg [[threadgroup_position_in_grid]],
                    uint simd [[simdgroup_index_in_threadgroup]],
                    uint lane [[thread_index_in_simdgroup]]) {
    uint expert = 0, first, count;
    if (s.gather != 0) {
        if (tg.y >= tile_count[0]) {
            return;
        }
        uint4 tile = tiles[tg.y];
        if (tile.z > PREFILL_FEW) {
            return;
        }
        expert = tile.x;
        first = tile.y;
        count = tile.z;
    } else {
        first = tg.y * PREFILL_FEW;
        count = min(PREFILL_FEW, s.m - first);
    }
    uint n0 = (tg.x * PREFILL_FEW_SIMDS + simd) * PREFILL_FEW_ROWS;
    uint row_words = s.k * BITS / 32;
    uint groups = s.k / 64;
    device const float *xr[PREFILL_FEW];
    for (uint i = 0; i < PREFILL_FEW; i++) {
        uint r = i < count ? (s.gather == 1 ? rows[first + i] : first + i) : 0;
        xr[i] = x + ulong(r) * s.k;
    }
    float acc[PREFILL_FEW][PREFILL_FEW_ROWS] = {};

    for (uint k = lane * 32; k < s.k; k += 32 * 32) {
        for (uint i = 0; i < PREFILL_FEW; i++) {
            if (i >= count) {
                break;
            }
            float xv[32];
            float xsum = load_x32(xr[i] + k, xv);
#pragma clang loop unroll(full)
            for (uint r = 0; r < PREFILL_FEW_ROWS; r++) {
                uint n = n0 + r;
                if (n < s.n) {
                    ulong row = ulong(expert) * s.n + n;
                    acc[i][r] += dot32<BITS>(words + row * row_words + k * BITS / 32, scales,
                                             biases, row * groups + k / 64, xv, xsum);
                }
            }
        }
    }

    for (uint i = 0; i < PREFILL_FEW; i++) {
        if (i >= count) {
            break;
        }
        for (uint r = 0; r < PREFILL_FEW_ROWS; r++) {
            float sum = simd_sum(acc[i][r]);
            uint n = n0 + r;
            if (lane == 0 && n < s.n) {
                y[ulong(first + i) * s.n + n] = sum;
            }
        }
    }
}

template [[host_name("qmm_few_5bit")]] kernel decltype(qmm_few<5>) qmm_few<5>;
template [[host_name("qmm_few_8bit")]] kernel decltype(qmm_few<8>) qmm_few<8>;

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
kernel void silu_mul_rows(device float *g [[buffer(0)]],
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
    uint snapshots;  // write the state after each token to `snaps`
};

// deltanet_conv over `tokens` tokens: one thread per channel walks the
// tokens in order. x and out are [tokens, channels]. With snapshots, snaps
// gets the state after token t at t * (kernel_size - 1) * channels.
kernel void conv_rows(device const float *x [[buffer(0)]],
                      device const float *w [[buffer(1)]],
                      device float *state [[buffer(2)]],
                      device float *out [[buffer(3)]],
                      constant ConvRowsParams &p [[buffer(4)]],
                      device float *snaps [[buffer(5)]],
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
        if (p.snapshots) {
            device float *snap = snaps + ulong(t) * past * p.channels;
            for (uint j = 0; j < past; j++) {
                snap[j * p.channels + c] = hist[j];
            }
        }
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
    uint snapshots;  // write the state after each token to `snaps`
};

kernel void delta_rows(device const float *mixed [[buffer(0)]],
                       device const float *a [[buffer(1)]],
                       device const float *b [[buffer(2)]],
                       device const float *a_log [[buffer(3)]],
                       device const float *dt_bias [[buffer(4)]],
                       device float *state [[buffer(5)]],
                       device float *y [[buffer(6)]],
                       constant DeltaRowsParams &p [[buffer(7)]],
                       device float *snaps [[buffer(8)]],
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
        if (p.snapshots) {
            // Same layout as `state`, one copy per token.
            device float *snap = snaps + ulong(t) * p.v_heads * p.v_dim * K_DIM +
                                 (ulong(h) * p.v_dim + row) * K_DIM + part * PER_LANE;
            for (uint i = 0; i < PER_LANE; i++) {
                snap[i] = st[i];
            }
        }
    }
    for (uint i = 0; i < PER_LANE; i++) {
        sp[i] = st[i];
    }
}

// conv_rows with one thread per (channel, token), all tokens at once, then
// conv_state moves the last kernel_size - 1 inputs into the state. Input i
// of the window is x[i] for i >= 0, else state row i + kernel_size - 1.
inline float prefill_conv_in(device const float *x, device const float *state, int i, uint c,
                             constant ConvRowsParams &p) {
    return i >= 0 ? x[ulong(i) * p.channels + c] : state[uint(i + int(p.kernel_size) - 1) * p.channels + c];
}

kernel void conv_rows2(device const float *x [[buffer(0)]],
                       device const float *w [[buffer(1)]],
                       device const float *state [[buffer(2)]],
                       device float *out [[buffer(3)]],
                       constant ConvRowsParams &p [[buffer(4)]],
                       device float *snaps [[buffer(5)]],
                       uint2 gid [[thread_position_in_grid]]) {
    uint c = gid.x, t = gid.y;
    if (c >= p.channels || t >= p.tokens) {
        return;
    }
    int past = int(p.kernel_size) - 1;
    device const float *wc = w + c * p.kernel_size;
    float acc = 0;
    for (int j = 0; j <= past; j++) {
        acc += wc[j] * prefill_conv_in(x, state, int(t) - past + j, c, p);
    }
    out[ulong(t) * p.channels + c] = silu(acc);
    if (p.snapshots) {
        device float *snap = snaps + ulong(t) * uint(past) * p.channels;
        for (int j = 0; j < past; j++) {
            snap[uint(j) * p.channels + c] = prefill_conv_in(x, state, int(t) - past + 1 + j, c, p);
        }
    }
}

kernel void conv_state(device const float *x [[buffer(0)]],
                       device float *state [[buffer(2)]],
                       constant ConvRowsParams &p [[buffer(4)]],
                       uint c [[thread_position_in_grid]]) {
    if (c >= p.channels) {
        return;
    }
    int past = int(p.kernel_size) - 1;
    float last[8];
    for (int j = 0; j < past; j++) {
        last[j] = prefill_conv_in(x, state, int(p.tokens) - past + j, c, p);
    }
    for (int j = 0; j < past; j++) {
        state[uint(j) * p.channels + c] = last[j];
    }
}

// delta_rows in two steps. delta_prep runs over all tokens at once: q and
// k of each k head are normalized in place in `mixed` (q also scaled by
// 1/sqrt(K_DIM)), and a and b become the decay g and the rate beta. Then
// delta_rows2 walks the tokens with only the state update left, 8 lanes per
// state row of 16 values each, loading token t + 1 while it computes t.
kernel void delta_prep(device float *mixed [[buffer(0)]],
                       device float *a [[buffer(1)]],
                       device float *b [[buffer(2)]],
                       device const float *a_log [[buffer(3)]],
                       device const float *dt_bias [[buffer(4)]],
                       constant DeltaRowsParams &p [[buffer(5)]],
                       uint2 tg [[threadgroup_position_in_grid]],
                       uint lane [[thread_index_in_simdgroup]]) {
    constexpr uint K_DIM = 128;
    uint t = tg.x;
    uint head = tg.y;  // q heads, then k heads, then v heads for g, beta
    if (head < 2 * p.k_heads) {
        device float4 *v = (device float4 *)(mixed + ulong(t) * p.channels + head * K_DIM) + lane;
        float4 x = *v;
        float ss = simd_sum(dot(x, x));
        float scale = head < p.k_heads ? 1.0f / sqrt(float(K_DIM)) : 1.0f;
        *v = x * (scale / sqrt(ss + p.eps));
    } else {
        uint h = head - 2 * p.k_heads + lane * (p.v_heads / 32);
        for (uint i = 0; i < p.v_heads / 32; i++, h++) {
            ulong at = ulong(t) * p.v_heads + h;
            float x = a[at] + dt_bias[h];
            float softplus = max(x, 0.0f) + log(1.0f + exp(-abs(x)));
            a[at] = exp(-exp(a_log[h]) * softplus);
            b[at] = sigmoid(b[at]);
        }
    }
}

kernel void delta_rows2(device const float *mixed [[buffer(0)]],
                        device const float *g [[buffer(1)]],
                        device const float *beta [[buffer(2)]],
                        device float *state [[buffer(5)]],
                        device float *y [[buffer(6)]],
                        constant DeltaRowsParams &p [[buffer(7)]],
                        device float *snaps [[buffer(8)]],
                        uint2 tg [[threadgroup_position_in_grid]],
                        uint simd [[simdgroup_index_in_threadgroup]],
                        uint simds [[simdgroups_per_threadgroup]],
                        uint lane [[thread_index_in_simdgroup]]) {
    constexpr uint K_DIM = 128;
    constexpr uint PER = 16;
    uint h = tg.y;
    uint row = (tg.x * simds + simd) * 4 + lane / 8;
    uint part = lane % 8;
    if (row >= p.v_dim) {
        return;
    }
    uint kh = h / (p.v_heads / p.k_heads);
    uint key_size = p.k_heads * K_DIM;
    uint value_size = p.v_heads * p.v_dim;

    device float *sp = state + (ulong(h) * p.v_dim + row) * K_DIM + part * PER;
    float st[PER];
    for (uint i = 0; i < PER; i++) {
        st[i] = sp[i];
    }

    device const float *qp = mixed + kh * K_DIM + part * PER;
    device const float *kp = qp + key_size;
    device const float *vp = mixed + 2 * key_size + h * p.v_dim + row;
    float4 nq[PER / 4], nk[PER / 4];
    float nv = 0, ng = 0, nb = 0;
    if (p.tokens > 0) {
        for (uint i = 0; i < PER / 4; i++) {
            nq[i] = ((device const float4 *)qp)[i];
            nk[i] = ((device const float4 *)kp)[i];
        }
        nv = vp[0];
        ng = g[h];
        nb = beta[h];
    }
    for (uint t = 0; t < p.tokens; t++) {
        float4 q[PER / 4], k[PER / 4];
        for (uint i = 0; i < PER / 4; i++) {
            q[i] = nq[i];
            k[i] = nk[i];
        }
        float v = nv, gt = ng, bt = nb;
        if (t + 1 < p.tokens) {
            ulong o = ulong(t + 1) * p.channels;
            for (uint i = 0; i < PER / 4; i++) {
                nq[i] = ((device const float4 *)(qp + o))[i];
                nk[i] = ((device const float4 *)(kp + o))[i];
            }
            nv = vp[o];
            ng = g[(t + 1) * p.v_heads + h];
            nb = beta[(t + 1) * p.v_heads + h];
        }

        float mem = 0;
        for (uint i = 0; i < PER / 4; i++) {
            float4 s4 = float4(st[4 * i], st[4 * i + 1], st[4 * i + 2], st[4 * i + 3]) * gt;
            st[4 * i] = s4.x;
            st[4 * i + 1] = s4.y;
            st[4 * i + 2] = s4.z;
            st[4 * i + 3] = s4.w;
            mem += dot(s4, k[i]);
        }
        mem += simd_shuffle_xor(mem, 1);
        mem += simd_shuffle_xor(mem, 2);
        mem += simd_shuffle_xor(mem, 4);
        float delta = (v - mem) * bt;
        float out = 0;
        for (uint i = 0; i < PER / 4; i++) {
            float4 s4 = float4(st[4 * i], st[4 * i + 1], st[4 * i + 2], st[4 * i + 3]) + k[i] * delta;
            st[4 * i] = s4.x;
            st[4 * i + 1] = s4.y;
            st[4 * i + 2] = s4.z;
            st[4 * i + 3] = s4.w;
            out += dot(s4, q[i]);
        }
        out += simd_shuffle_xor(out, 1);
        out += simd_shuffle_xor(out, 2);
        out += simd_shuffle_xor(out, 4);
        if (part == 0) {
            y[ulong(t) * value_size + h * p.v_dim + row] = out;
        }
        if (p.snapshots) {
            device float *snap = snaps + ulong(t) * p.v_heads * p.v_dim * K_DIM +
                                 (ulong(h) * p.v_dim + row) * K_DIM + part * PER;
            for (uint i = 0; i < PER; i++) {
                snap[i] = st[i];
            }
        }
    }
    for (uint i = 0; i < PER; i++) {
        sp[i] = st[i];
    }
}

// ---------------------------------------------------------------------------
// Causal attention of T new tokens over the KV cache, which already holds
// them at positions pos .. pos + T - 1. A simdgroup takes 8 query tokens of
// one head and walks the keys in blocks of 32 with an online softmax, using
// 8 x 8 simdgroup matrices for q K^T and P V with f32 sums. T is float, or
// half for twice the speed; with half, q, K and V are half copies (to_half).
// The 4 simdgroups of a threadgroup are 4 heads of the same kv head, so they
// share K and V loads.
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

constant uint PREFILL_ATT_DIM = 256;
constant uint PREFILL_ATT_KB = 32;
constant uint PREFILL_ATT_SIMDS = 4;

// The KV cache is half; T is the type of q, the probabilities and the
// matrix products.
template <typename T>
inline simdgroup_matrix<T, 8, 8> prefill_from_half(simdgroup_half8x8 m) {
    simdgroup_matrix<T, 8, 8> r;
    thread auto &re = r.thread_elements();
    thread auto &me = m.thread_elements();
    re[0] = T(me[0]);
    re[1] = T(me[1]);
    return r;
}

template <typename T>
kernel void attention_rows(device const T *q [[buffer(0)]],
                           device const half *keys [[buffer(1)]],
                           device const half *values [[buffer(2)]],
                           device const float *q_and_gate [[buffer(3)]],
                           device float *out [[buffer(4)]],
                           constant PrefillAttentionParams &p [[buffer(5)]],
                           uint2 tg [[threadgroup_position_in_grid]],
                           uint simd [[simdgroup_index_in_threadgroup]],
                           uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float s_all[PREFILL_ATT_SIMDS][8 * PREFILL_ATT_KB];
    threadgroup T p_all[PREFILL_ATT_SIMDS][8 * PREFILL_ATT_KB];
    threadgroup float a_all[PREFILL_ATT_SIMDS][8];
    threadgroup float *sc = s_all[simd];
    threadgroup T *pt = p_all[simd];
    threadgroup float *al = a_all[simd];

    uint h = tg.y * PREFILL_ATT_SIMDS + simd;
    uint kv = h / (p.heads / p.kv_heads);
    uint q0 = tg.x * 8;
    uint q_stride = p.heads * PREFILL_ATT_DIM;
    uint kv_stride = p.kv_heads * PREFILL_ATT_DIM;
    device const T *qh = q + ulong(q0) * q_stride + h * PREFILL_ATT_DIM;
    device const half *kh = keys + kv * PREFILL_ATT_DIM;
    device const half *vh = values + kv * PREFILL_ATT_DIM;
    uint end = p.pos + min(q0 + 8, p.tokens);

    // For the softmax, lane l handles row l / 4, columns (l % 4) * 8 .. + 8
    // of the scores. In the matrices it holds row fp.x.
    uint r = lane / 4;
    uint c0 = (lane % 4) * 8;
    uint last = p.pos + q0 + r;  // the newest key this row may see
    uint2 fp = frag_pos(lane);
    float m = -INFINITY;
    float l = 0;

    simdgroup_float8x8 o[PREFILL_ATT_DIM / 8];
    for (uint i = 0; i < PREFILL_ATT_DIM / 8; i++) {
        o[i] = simdgroup_float8x8(0);
    }

    for (uint k0 = 0; k0 < end; k0 += PREFILL_ATT_KB) {
        simdgroup_float8x8 s[PREFILL_ATT_KB / 8];
        for (uint j = 0; j < PREFILL_ATT_KB / 8; j++) {
            s[j] = simdgroup_float8x8(0);
        }
        for (uint d = 0; d < PREFILL_ATT_DIM; d += 8) {
            simdgroup_matrix<T, 8, 8> qm;
            simdgroup_load(qm, qh + d, q_stride);
            for (uint j = 0; j < PREFILL_ATT_KB / 8; j++) {
                simdgroup_half8x8 km;
                simdgroup_load(km, kh + ulong(k0 + 8 * j) * kv_stride + d, kv_stride, ulong2(0, 0),
                               true);
                simdgroup_multiply_accumulate(s[j], qm, prefill_from_half<T>(km), s[j]);
            }
        }
        for (uint j = 0; j < PREFILL_ATT_KB / 8; j++) {
            simdgroup_store(s[j], sc + 8 * j, PREFILL_ATT_KB);
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);

        float v[8];
        float bm = -INFINITY;
        for (uint i = 0; i < 8; i++) {
            uint key = k0 + c0 + i;
            v[i] = key <= last && key < end ? sc[r * PREFILL_ATT_KB + c0 + i] * p.scale : -INFINITY;
            bm = max(bm, v[i]);
        }
        bm = max(bm, simd_shuffle_xor(bm, 1));
        bm = max(bm, simd_shuffle_xor(bm, 2));
        float m_new = max(m, bm);
        float sum = 0;
        for (uint i = 0; i < 8; i++) {
            float e = exp(v[i] - m_new);
            pt[r * PREFILL_ATT_KB + c0 + i] = T(e);
            sum += e;
        }
        sum += simd_shuffle_xor(sum, 1);
        sum += simd_shuffle_xor(sum, 2);
        float alpha = exp(m - m_new);
        l = l * alpha + sum;
        m = m_new;
        if (lane % 4 == 0) {
            al[r] = alpha;
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);

        float a = al[fp.x];
        simdgroup_matrix<T, 8, 8> pm[PREFILL_ATT_KB / 8];
        for (uint j = 0; j < PREFILL_ATT_KB / 8; j++) {
            simdgroup_load(pm[j], pt + 8 * j, PREFILL_ATT_KB);
        }
        for (uint d = 0; d < PREFILL_ATT_DIM / 8; d++) {
            thread auto &oe = o[d].thread_elements();
            oe[0] *= a;
            oe[1] *= a;
            for (uint j = 0; j < PREFILL_ATT_KB / 8; j++) {
                simdgroup_half8x8 vm;
                simdgroup_load(vm, vh + ulong(k0 + 8 * j) * kv_stride + 8 * d, kv_stride);
                simdgroup_multiply_accumulate(o[d], pm[j], prefill_from_half<T>(vm), o[d]);
            }
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Lane l writes row l / 4, two values per 8 x 8 block.
    float inv = 1.0f / l;
    uint row_t = q0 + r;
    for (uint d = 0; d < PREFILL_ATT_DIM / 8; d++) {
        simdgroup_store(o[d], sc, 8);
        simdgroup_barrier(mem_flags::mem_threadgroup);
        if (row_t < p.tokens) {
            for (uint i = (lane % 4) * 2; i < (lane % 4) * 2 + 2; i++) {
                uint col = 8 * d + i;
                float gate = q_and_gate[ulong(row_t) * 2 * q_stride + h * 2 * PREFILL_ATT_DIM + PREFILL_ATT_DIM + col];
                out[ulong(row_t) * q_stride + h * PREFILL_ATT_DIM + col] = sc[r * 8 + i] * inv * sigmoid(gate);
            }
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
    }
}

template [[host_name("attention_rows")]] kernel decltype(attention_rows<float>) attention_rows<float>;
template [[host_name("attention_rows_half")]] kernel decltype(attention_rows<half>) attention_rows<half>;

// attention_rows over a half cache, with q held in registers for the whole
// key loop and 64 keys per block, SIMDS heads of one kv head per
// threadgroup. T is the type of q and of the
// probabilities; the products accumulate in f32.
constant uint PREFILL_ATT2_KB = 64;

template <typename T>
inline simdgroup_matrix<T, 8, 8> prefill_att2_widen(simdgroup_half8x8 m) {
    simdgroup_matrix<T, 8, 8> r;
    thread auto &re = r.thread_elements();
    thread auto &me = m.thread_elements();
    re[0] = T(me[0]);
    re[1] = T(me[1]);
    return r;
}

template <typename T, uint SIMDS>
kernel void attention_rows2(device const T *q [[buffer(0)]],
                            device const half *keys [[buffer(1)]],
                            device const half *values [[buffer(2)]],
                            device const float *q_and_gate [[buffer(3)]],
                            device float *out [[buffer(4)]],
                            constant PrefillAttentionParams &p [[buffer(5)]],
                            uint2 tg [[threadgroup_position_in_grid]],
                            uint simd [[simdgroup_index_in_threadgroup]],
                            uint lane [[thread_index_in_simdgroup]]) {
    constexpr uint KB = PREFILL_ATT2_KB;
    constexpr uint D = PREFILL_ATT_DIM;
    threadgroup float s_all[SIMDS][8 * KB];
    threadgroup T p_all[SIMDS][8 * KB];
    threadgroup float a_all[SIMDS][8];
    threadgroup float *sc = s_all[simd];
    threadgroup T *pt = p_all[simd];
    threadgroup float *al = a_all[simd];

    uint h = tg.y * SIMDS + simd;
    uint kv = h / (p.heads / p.kv_heads);
    uint q0 = tg.x * 8;
    uint q_stride = p.heads * D;
    uint kv_stride = p.kv_heads * D;
    device const half *kh = keys + kv * D;
    device const half *vh = values + kv * D;
    uint end = p.pos + min(q0 + 8, p.tokens);

    simdgroup_matrix<T, 8, 8> qm[D / 8];
    {
        device const T *qh = q + ulong(q0) * q_stride + h * D;
        for (uint d = 0; d < D / 8; d++) {
            simdgroup_load(qm[d], qh + 8 * d, q_stride);
        }
    }

    // For the softmax, lane l handles row l / 4, columns (l % 4) * 16 .. + 16
    // of the scores. In the matrices it holds row fp.x.
    constexpr uint PER = KB / 4;
    uint r = lane / 4;
    uint c0 = (lane % 4) * PER;
    uint last = p.pos + q0 + r;  // the newest key this row may see
    uint2 fp = frag_pos(lane);
    float m = -INFINITY;
    float l = 0;

    simdgroup_float8x8 o[D / 8];
    for (uint i = 0; i < D / 8; i++) {
        o[i] = simdgroup_float8x8(0);
    }

    for (uint k0 = 0; k0 < end; k0 += KB) {
        simdgroup_float8x8 s[KB / 8];
        for (uint j = 0; j < KB / 8; j++) {
            s[j] = simdgroup_float8x8(0);
        }
        for (uint d = 0; d < D / 8; d++) {
            for (uint j = 0; j < KB / 8; j++) {
                simdgroup_half8x8 km;
                simdgroup_load(km, kh + ulong(k0 + 8 * j) * kv_stride + 8 * d, kv_stride, ulong2(0, 0), true);
                simdgroup_multiply_accumulate(s[j], qm[d], prefill_att2_widen<T>(km), s[j]);
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
            v[i] = key <= last && key < end ? sc[r * KB + c0 + i] * p.scale : -INFINITY;
            bm = max(bm, v[i]);
        }
        bm = max(bm, simd_shuffle_xor(bm, 1));
        bm = max(bm, simd_shuffle_xor(bm, 2));
        float m_new = max(m, bm);
        float sum = 0;
        for (uint i = 0; i < PER; i++) {
            float e = exp(v[i] - m_new);
            pt[r * KB + c0 + i] = T(e);
            sum += e;
        }
        sum += simd_shuffle_xor(sum, 1);
        sum += simd_shuffle_xor(sum, 2);
        float alpha = exp(m - m_new);
        l = l * alpha + sum;
        m = m_new;
        if (lane % 4 == 0) {
            al[r] = alpha;
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);

        float a = al[fp.x];
        simdgroup_matrix<T, 8, 8> pm[KB / 8];
        for (uint j = 0; j < KB / 8; j++) {
            simdgroup_load(pm[j], pt + 8 * j, KB);
        }
        for (uint d = 0; d < D / 8; d++) {
            thread auto &oe = o[d].thread_elements();
            oe[0] *= a;
            oe[1] *= a;
            for (uint j = 0; j < KB / 8; j++) {
                simdgroup_half8x8 vm;
                simdgroup_load(vm, vh + ulong(k0 + 8 * j) * kv_stride + 8 * d, kv_stride);
                simdgroup_multiply_accumulate(o[d], pm[j], prefill_att2_widen<T>(vm), o[d]);
            }
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Lane l writes row l / 4, two values per 8 x 8 block.
    float inv = 1.0f / l;
    uint row_t = q0 + r;
    for (uint d = 0; d < D / 8; d++) {
        simdgroup_store(o[d], sc, 8);
        simdgroup_barrier(mem_flags::mem_threadgroup);
        if (row_t < p.tokens) {
            for (uint i = (lane % 4) * 2; i < (lane % 4) * 2 + 2; i++) {
                uint col = 8 * d + i;
                float gate = q_and_gate[ulong(row_t) * 2 * q_stride + h * 2 * D + D + col];
                out[ulong(row_t) * q_stride + h * D + col] = sc[r * 8 + i] * inv * sigmoid(gate);
            }
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
    }
}

template [[host_name("attention_rows2")]] kernel decltype(attention_rows2<float, 4>) attention_rows2<float, 4>;
template [[host_name("attention_rows2_half")]] kernel decltype(attention_rows2<half, 4>) attention_rows2<half, 4>;
template [[host_name("attention_rows2_half8")]] kernel decltype(attention_rows2<half, 8>) attention_rows2<half, 8>;

kernel void to_half(device const float *x [[buffer(0)]],
                    device half *y [[buffer(1)]],
                    uint i [[thread_position_in_grid]]) {
    y[i] = half(x[i]);
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
    uint tile_rows;  // rows per tile at most
};

// One threadgroup sorts the (token, slot) pairs by expert. For each pair i:
// order[i] is its row in expert order, and rows[order[i]] its token. Tiles of
// up to PREFILL_MM_BM rows of one expert go to `tiles` for qmm in gather mode, with
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
            for (uint s = 0; s < count; s += p.tile_rows) {
                tiles[n++] = uint4(e, offset + s, min(p.tile_rows, count - s), 0);
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

// out[slot] = max(out[slot], max |x[i]|), on the bits of the float: for
// values >= 0 they order the same as the floats. For range checks only.
kernel void max_abs(device const float *x [[buffer(0)]],
                    device atomic_uint *out [[buffer(1)]],
                    constant uint &slot [[buffer(2)]],
                    uint i [[thread_position_in_grid]]) {
    float v = simd_max(abs(x[i]));
    if (simd_is_first()) {
        atomic_fetch_max_explicit(&out[slot], as_type<uint>(v), memory_order_relaxed);
    }
}

// out[row] = the index of the largest of the `n` values of row `row`, the
// lowest index on ties. One threadgroup per row.
kernel void argmax_rows(device const float *x [[buffer(0)]],
                        device uint *out [[buffer(1)]],
                        constant uint &n [[buffer(2)]],
                        uint row [[threadgroup_position_in_grid]],
                        uint tid [[thread_position_in_threadgroup]],
                        uint threads [[threads_per_threadgroup]],
                        uint simd [[simdgroup_index_in_threadgroup]],
                        uint lane [[thread_index_in_simdgroup]],
                        uint simds [[simdgroups_per_threadgroup]]) {
    threadgroup float best_v[32];
    threadgroup uint best_i[32];
    device const float *xr = x + ulong(row) * n;
    float v = -INFINITY;
    uint index = 0;
    for (uint i = tid; i < n; i += threads) {
        if (xr[i] > v) {
            v = xr[i];
            index = i;
        }
    }
    float top = simd_max(v);
    index = simd_min(v == top ? index : 0xffffffffu);
    if (lane == 0) {
        best_v[simd] = top;
        best_i[simd] = index;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        float m = best_v[0];
        uint mi = best_i[0];
        for (uint s = 1; s < simds; s++) {
            if (best_v[s] > m || (best_v[s] == m && best_i[s] < mi)) {
                m = best_v[s];
                mi = best_i[s];
            }
        }
        out[row] = mi;
    }
}

// The experts' gate and up in one pass, for the tiles of more than
// PREFILL_FEW rows, as qmm2 32 x 128 in qmm.metal: a threadgroup computes
// 32 rows by 64 columns of both, with W rows 0-63 of the tile from gate and
// 64-127 from up so x is loaded once, then writes silu(gate) * up to g in
// f32 (for qmm_few in down) and to g_half (for qmm2 in down). x is half.
constant uint PREFILL_GU_BM = 32;
constant uint PREFILL_GU_BN = 64;   // columns of each of gate and up
constant uint PREFILL_GU_BK = 32;
constant uint PREFILL_GU_LD = PREFILL_GU_BK + 8;

template <uint BITS>
kernel void prefill_gate_up(device const uint *gate_words [[buffer(0)]],
                            device const ushort *gate_scales [[buffer(1)]],
                            device const ushort *gate_biases [[buffer(2)]],
                            device const uint *up_words [[buffer(3)]],
                            device const ushort *up_scales [[buffer(4)]],
                            device const ushort *up_biases [[buffer(5)]],
                            device const half *x [[buffer(6)]],
                            device float *g [[buffer(7)]],
                            device half *g_half [[buffer(8)]],
                            constant MmShape &s [[buffer(9)]],
                            device const uint4 *tiles [[buffer(10)]],
                            device const uint *rows [[buffer(11)]],
                            device const uint *tile_count [[buffer(12)]],
                            uint2 tg [[threadgroup_position_in_grid]],
                            uint tid [[thread_index_in_threadgroup]],
                            uint simd [[simdgroup_index_in_threadgroup]]) {
    threadgroup half xt[PREFILL_GU_BM * PREFILL_GU_LD];
    threadgroup half wt[2 * PREFILL_GU_BN * PREFILL_GU_LD];
    threadgroup float c[PREFILL_GU_BM * 2 * PREFILL_GU_BN];

    if (tg.y >= tile_count[0]) {
        return;
    }
    uint4 tile = tiles[tg.y];
    if (tile.z <= PREFILL_FEW) {
        return;  // qmm_few takes it
    }
    uint expert = tile.x, first = tile.y, count = tile.z;
    uint n0 = tg.x * PREFILL_GU_BN;

    // Thread t loads 8 x values of row t / 4 and dequantizes the 32
    // weights of W row t of the tile.
    uint xr = tid / 4, xc = (tid % 4) * 8;
    bool x_ok = xr < count;
    device const half *xrow = x;
    if (x_ok) {
        xrow = x + ulong(rows[first + xr]) * s.k;
    }
    bool is_up = tid >= PREFILL_GU_BN;
    uint wn = n0 + tid % PREFILL_GU_BN;
    bool w_ok = wn < s.n;
    ulong wrow = ulong(expert) * s.n + (w_ok ? wn : 0);
    device const uint *wp = (is_up ? up_words : gate_words) + wrow * (s.k * BITS / 32);
    device const ushort *sp = (is_up ? up_scales : gate_scales) + wrow * (s.k / 64);
    device const ushort *bp = (is_up ? up_biases : gate_biases) + wrow * (s.k / 64);

    // Simdgroups 0 and 1 compute gate columns, 2 and 3 up columns, 32 each.
    uint sn = simd * 32;
    simdgroup_float8x8 acc[4][4];
    for (uint i = 0; i < 4; i++) {
        for (uint j = 0; j < 4; j++) {
            acc[i][j] = simdgroup_float8x8(0);
        }
    }

    threadgroup half4 *xd = (threadgroup half4 *)(xt + xr * PREFILL_GU_LD + xc);
    threadgroup half4 *wd = (threadgroup half4 *)(wt + tid * PREFILL_GU_LD);
    for (uint k0 = 0; k0 < s.k; k0 += PREFILL_GU_BK) {
        if (x_ok) {
            device const half4 *p = (device const half4 *)(xrow + k0 + xc);
            xd[0] = p[0];
            xd[1] = p[1];
        } else {
            xd[0] = half4(0);
            xd[1] = half4(0);
        }
        if (w_ok) {
            uint packed[BITS];
            device const uint *wk = wp + k0 / 32 * BITS;
            for (uint j = 0; j < BITS; j++) {
                packed[j] = wk[j];
            }
            float scale = bf16_to_float(sp[k0 / 64]);
            float bias = bf16_to_float(bp[k0 / 64]);
#pragma clang loop unroll(full)
            for (uint i = 0; i < 8; i++) {
                float4 q = float4(unpack<BITS>(packed, 4 * i), unpack<BITS>(packed, 4 * i + 1),
                                  unpack<BITS>(packed, 4 * i + 2), unpack<BITS>(packed, 4 * i + 3));
                wd[i] = half4(q * scale + bias);
            }
        } else {
            for (uint i = 0; i < 8; i++) {
                wd[i] = half4(0);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

#pragma clang loop unroll(full)
        for (uint kk = 0; kk < PREFILL_GU_BK; kk += 8) {
            simdgroup_half8x8 a[4];
            simdgroup_half8x8 b[4];
            for (uint i = 0; i < 4; i++) {
                simdgroup_load(a[i], xt + (8 * i) * PREFILL_GU_LD + kk, PREFILL_GU_LD);
            }
            for (uint j = 0; j < 4; j++) {
                simdgroup_load(b[j], wt + (sn + 8 * j) * PREFILL_GU_LD + kk, PREFILL_GU_LD, ulong2(0, 0), true);
            }
            for (uint i = 0; i < 4; i++) {
                for (uint j = 0; j < 4; j++) {
                    simdgroup_multiply_accumulate(acc[i][j], a[i], b[j], acc[i][j]);
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    constexpr uint CW = 2 * PREFILL_GU_BN;
    for (uint i = 0; i < 4; i++) {
        for (uint j = 0; j < 4; j++) {
            simdgroup_store(acc[i][j], c + (8 * i) * CW + sn + 8 * j, CW);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Thread t writes 16 columns of row t / 4.
    uint r = tid / 4, c0 = (tid % 4) * 16;
    if (r < count) {
        ulong out = ulong(first + r) * s.n + n0;
        for (uint j = c0; j < c0 + 16; j++) {
            if (n0 + j < s.n) {
                float v = silu(c[r * CW + j]) * c[r * CW + PREFILL_GU_BN + j];
                g[out + j] = v;
                g_half[out + j] = half(v);
            }
        }
    }
}

template [[host_name("prefill_gate_up_5bit")]] kernel decltype(prefill_gate_up<5>) prefill_gate_up<5>;
template [[host_name("prefill_gate_up_8bit")]] kernel decltype(prefill_gate_up<8>) prefill_gate_up<8>;

// prefill_gate_up on tiles of up to 64 rows: 8 simdgroups, each 32 rows by
// 16 columns of gate and the same 16 of up, so silu(gate) * up is taken in
// registers. Each step dequantizes the weights once for up to 64 rows (32
// above), and the 8-row blocks past the tile's end skip their MACs.
// In plain mode (gather 0, the shared expert) tile y is rows 64 y of x.
constant uint PREFILL_GU64_BM = 64;
constant uint PREFILL_GU64_BN = 64;  // columns of each of gate and up

// Dequantizes 16 weights, values OFF to OFF + 15 of a 32-value block.
template <uint BITS, uint OFF>
inline void dequant16(thread const uint *packed, float scale, float bias,
                      threadgroup half4 *dst) {
#pragma clang loop unroll(full)
    for (uint i = 0; i < 4; i++) {
        uint c = OFF + 4 * i;
        float4 q = float4(unpack<BITS>(packed, c), unpack<BITS>(packed, c + 1),
                          unpack<BITS>(packed, c + 2), unpack<BITS>(packed, c + 3));
        dst[i] = half4(q * scale + bias);
    }
}

template <uint BITS>
kernel void prefill_gate_up64(device const uint *gate_words [[buffer(0)]],
                              device const ushort *gate_scales [[buffer(1)]],
                              device const ushort *gate_biases [[buffer(2)]],
                              device const uint *up_words [[buffer(3)]],
                              device const ushort *up_scales [[buffer(4)]],
                              device const ushort *up_biases [[buffer(5)]],
                              device const half *x [[buffer(6)]],
                              device float *g [[buffer(7)]],
                              device half *g_half [[buffer(8)]],
                              constant MmShape &s [[buffer(9)]],
                              device const uint4 *tiles [[buffer(10)]],
                              device const uint *rows [[buffer(11)]],
                              device const uint *tile_count [[buffer(12)]],
                              uint2 tg [[threadgroup_position_in_grid]],
                              uint tid [[thread_index_in_threadgroup]],
                              uint simd [[simdgroup_index_in_threadgroup]],
                              uint lane [[thread_index_in_simdgroup]]) {
    threadgroup half xt[PREFILL_GU64_BM * PREFILL_GU_LD];
    threadgroup half wt[2 * PREFILL_GU64_BN * PREFILL_GU_LD];

    uint expert = 0, first, count;
    if (s.gather != 0) {
        if (tg.y >= tile_count[0]) {
            return;
        }
        uint4 tile = tiles[tg.y];
        if (tile.z <= PREFILL_FEW) {
            return;  // qmm_few takes it
        }
        expert = tile.x;
        first = tile.y;
        count = tile.z;
    } else {
        first = tg.y * PREFILL_GU64_BM;
        count = min(PREFILL_GU64_BM, s.m - first);
    }
    uint n0 = tg.x * PREFILL_GU64_BN;

    // Thread t loads 8 x values of row t / 4 and dequantizes 16 weights of
    // W row t % 128 of the tile (0-63 gate, 64-127 up): values 0-15 of the
    // step for t < 128, 16-31 above, so the branch is uniform per simdgroup.
    uint xr = tid / 4, xc = (tid % 4) * 8;
    bool x_ok = xr < count;
    device const half *xrow = x;
    if (x_ok) {
        uint r = s.gather == 1 ? rows[first + xr] : first + xr;
        xrow = x + ulong(r) * s.k;
    }
    uint wr = tid % (2 * PREFILL_GU64_BN);
    bool low = tid < 2 * PREFILL_GU64_BN;
    bool is_up = wr >= PREFILL_GU64_BN;
    uint wn = n0 + wr % PREFILL_GU64_BN;
    bool w_ok = wn < s.n;
    ulong wrow = ulong(expert) * s.n + (w_ok ? wn : 0);
    device const uint *wp = (is_up ? up_words : gate_words) + wrow * (s.k * BITS / 32);
    device const ushort *sp = (is_up ? up_scales : gate_scales) + wrow * (s.k / 64);
    device const ushort *bp = (is_up ? up_biases : gate_biases) + wrow * (s.k / 64);

    // Simdgroup s: rows 32 (s / 4), gate and up columns 16 (s % 4).
    uint sm = (simd / 4) * 32;
    uint sn = (simd % 4) * 16;
    bool active = sm < count;
    // 8-row blocks of this simdgroup that hold rows; the rest skip their MACs.
    uint live = active ? min(4u, (count - sm + 7) / 8) : 0;
    simdgroup_float8x8 ag[4][2], au[4][2];
    for (uint i = 0; i < 4; i++) {
        for (uint j = 0; j < 2; j++) {
            ag[i][j] = simdgroup_float8x8(0);
            au[i][j] = simdgroup_float8x8(0);
        }
    }

    threadgroup half4 *xd = (threadgroup half4 *)(xt + xr * PREFILL_GU_LD + xc);
    threadgroup half4 *wd = (threadgroup half4 *)(wt + wr * PREFILL_GU_LD + (low ? 0 : 16));
    for (uint k0 = 0; k0 < s.k; k0 += PREFILL_GU_BK) {
        if (x_ok) {
            device const half4 *p = (device const half4 *)(xrow + k0 + xc);
            xd[0] = p[0];
            xd[1] = p[1];
        } else {
            xd[0] = half4(0);
            xd[1] = half4(0);
        }
        if (w_ok) {
            uint packed[BITS];
            device const uint *wk = wp + k0 / 32 * BITS;
            for (uint j = 0; j < BITS; j++) {
                packed[j] = wk[j];
            }
            float scale = bf16_to_float(sp[k0 / 64]);
            float bias = bf16_to_float(bp[k0 / 64]);
            if (low) {
                dequant16<BITS, 0>(packed, scale, bias, wd);
            } else {
                dequant16<BITS, 16>(packed, scale, bias, wd);
            }
        } else {
            for (uint i = 0; i < 4; i++) {
                wd[i] = half4(0);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (active) {
#pragma clang loop unroll(full)
            for (uint kk = 0; kk < PREFILL_GU_BK; kk += 8) {
                simdgroup_half8x8 bg[2], bu[2];
                for (uint j = 0; j < 2; j++) {
                    simdgroup_load(bg[j], wt + (sn + 8 * j) * PREFILL_GU_LD + kk, PREFILL_GU_LD,
                                   ulong2(0, 0), true);
                    simdgroup_load(bu[j], wt + (PREFILL_GU64_BN + sn + 8 * j) * PREFILL_GU_LD + kk,
                                   PREFILL_GU_LD, ulong2(0, 0), true);
                }
                // Unrolled with a test, not a break, so i stays a constant and
                // the accumulators stay in registers.
#pragma clang loop unroll(full)
                for (uint i = 0; i < 4; i++) {
                    if (i >= live) {
                        continue;
                    }
                    simdgroup_half8x8 a;
                    simdgroup_load(a, xt + (sm + 8 * i) * PREFILL_GU_LD + kk, PREFILL_GU_LD);
                    for (uint j = 0; j < 2; j++) {
                        simdgroup_multiply_accumulate(ag[i][j], a, bg[j], ag[i][j]);
                        simdgroup_multiply_accumulate(au[i][j], a, bu[j], au[i][j]);
                    }
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (!active) {
        return;
    }

    // A lane holds 2 neighbouring columns of each 8 x 8 matrix.
    uint2 fp = frag_pos(lane);
    for (uint i = 0; i < 4; i++) {
        uint r = sm + 8 * i + fp.x;
        if (r >= count) {
            continue;
        }
        ulong out = ulong(first + r) * s.n + n0;
        for (uint j = 0; j < 2; j++) {
            uint c = sn + 8 * j + fp.y;
            thread auto &ge = ag[i][j].thread_elements();
            thread auto &ue = au[i][j].thread_elements();
            for (uint e = 0; e < 2; e++) {
                if (n0 + c + e < s.n) {
                    float v = silu(ge[e]) * ue[e];
                    g[out + c + e] = v;
                    g_half[out + c + e] = half(v);
                }
            }
        }
    }
}

template [[host_name("prefill_gate_up64_5bit")]] kernel decltype(prefill_gate_up64<5>) prefill_gate_up64<5>;
template [[host_name("prefill_gate_up64_8bit")]] kernel decltype(prefill_gate_up64<8>) prefill_gate_up64<8>;

// g = silu(g) * u and g_half = g for the rows of the tiles of PREFILL_FEW
// rows or fewer, which prefill_gate_up skips. One threadgroup per tile.
kernel void prefill_silu_few(device float *g [[buffer(0)]],
                             device const float *u [[buffer(1)]],
                             device half *g_half [[buffer(2)]],
                             constant uint &n [[buffer(3)]],
                             device const uint4 *tiles [[buffer(4)]],
                             device const uint *tile_count [[buffer(5)]],
                             uint tg [[threadgroup_position_in_grid]],
                             uint tid [[thread_index_in_threadgroup]],
                             uint threads [[threads_per_threadgroup]]) {
    if (tg >= tile_count[0]) {
        return;
    }
    uint4 tile = tiles[tg];
    if (tile.z > PREFILL_FEW) {
        return;
    }
    for (uint i = tid; i < tile.z * n; i += threads) {
        ulong at = ulong(tile.y) * n + i;
        float v = silu(g[at]) * u[at];
        g[at] = v;
        g_half[at] = half(v);
    }
}

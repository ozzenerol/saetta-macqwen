// Split-K attention for one decode token (flash-decoding).
//
// attention_partial_mma: grid (splits, kv_heads), 256 threads. Threadgroup
// (s, kv) takes the 128-token chunks s, s + splits, s + 2 splits, ... of kv
// head kv for the 8 query heads that share it, so each key and value is read
// once for all 8. Over its chunks it keeps a running max and sum (online
// softmax) and writes, per q head, the unnormalised output sum(exp(x - m) v),
// the max score m and l = sum(exp(x - m)). The number of splits is fixed, so
// the partial results stay small at any length: with 128 splits they are
// 2 MB per layer, where one result per chunk took 32 MB at 262k tokens and
// was written and read again on every token.
// attention_reduce: one threadgroup per q head merges the splits and applies
// the sigmoid output gate.
//
// The cache is half, token-major, kv_heads * 256 values per token, the
// layout attention_prep writes, with room for whole chunks (GpuKvCache
// rounds its capacity up to 128), so a chunk never reads past the buffer.
// Rows past kv_len in the last chunk get weight 0. The math is f32. Head dim
// 256 and 8 q heads per kv head are fixed here; src/attention.rs checks them.

constant uint DEC_CHUNK = 128;
constant uint DEC_GROUP = 8;
constant uint DEC_DIM = 256;
// Row stride of the score tile: 4 floats of padding put the 8 rows of a
// transposed 8x8 tile store in different banks.
constant uint DEC_SC = DEC_CHUNK + 4;

struct DecodeParams {
    uint heads;
    uint kv_heads;
    uint capacity;
    float scale;
    uint splits;  // threadgroups per kv head, and partial results per q head
};

// The running max before any score. It is finite, as fast math wants: a row
// of the verify with no token yet in a chunk keeps m at this, its factor
// exp(m - m) is 1 and l stays 0, and the first real score sends the factor
// to 0.
constant float DEC_NO_SCORE = -1e30f;

// One chunk's softmax step for query head `g` (simdgroup g): turns the raw
// scores of its first n tokens in `row` into exp(scale * x - m) against the
// new running max m and the rest into 0, updates m and l, and leaves in
// alpha[g] the factor that rescales the output summed so far.
inline void chunk_softmax(threadgroup float *row, uint n, float scale, thread float &m,
                          thread float &l, threadgroup float *alpha, uint g, uint lane) {
    float cm = DEC_NO_SCORE;
    for (uint t = lane; t < n; t += 32) {
        cm = max(cm, row[t] * scale);
    }
    cm = simd_max(cm);
    float nm = max(m, cm);
    float a = exp(m - nm);
    float sum = 0;
    for (uint t = lane; t < DEC_CHUNK; t += 32) {
        float e = t < n ? exp(row[t] * scale - nm) : 0;
        row[t] = e;
        sum += e;
    }
    l = l * a + simd_sum(sum);
    m = nm;
    if (lane == 0) {
        alpha[g] = a;
    }
}

// Multiplies this lane's two elements of an 8x8 tile, both in row f.x, by a.
inline void scale_tile(thread simdgroup_float8x8 &t, float a) {
    t.thread_elements()[0] *= a;
    t.thread_elements()[1] *= a;
}

inline void load_half_tile(thread simdgroup_float8x8 &m, device const half *src, uint stride) {
    simdgroup_half8x8 h;
    simdgroup_load(h, src, stride);
    m.thread_elements()[0] = float(h.thread_elements()[0]);
    m.thread_elements()[1] = float(h.thread_elements()[1]);
}

// The end of both partial kernels: m and l of head `simd`. The 8 x 32
// output slice of the simdgroup goes to partial_out(...) + 8 i, tile i.
inline void write_ml(device float *part_ml, constant DecodeParams &p, uint split, uint kv,
                     uint simd, uint lane, float m, float l) {
    if (lane == 0) {
        uint i = split * p.heads + kv * DEC_GROUP + simd;
        part_ml[2 * i] = m;
        part_ml[2 * i + 1] = l;
    }
}

inline device float *partial_out(device float *part_o, constant DecodeParams &p, uint split,
                                 uint kv, uint simd) {
    return part_o + (split * p.heads + kv * DEC_GROUP) * DEC_DIM + 32 * simd;
}

// Scores as S^T = K Q^T in 8x8 tiles, so key rows load as they lie in the
// cache, and the values as O = P V. q, times the scale, sits in threadgroup
// memory, as in the one-threadgroup-per-chunk kernel this replaced; with q
// read from device memory on every tile, decode at 262k ran at 39.8 tok/s
// against that kernel's 42.6.
kernel void attention_partial_mma(device const float *q [[buffer(0)]],
                                  device const half *keys [[buffer(1)]],
                                  device const half *values [[buffer(2)]],
                                  device float *part_o [[buffer(3)]],
                                  device float *part_ml [[buffer(4)]],
                                  constant DecodeParams &p [[buffer(5)]],
                                  constant TokenParams &tp [[buffer(6)]],
                                  uint2 tg [[threadgroup_position_in_grid]],
                                  uint tid [[thread_index_in_threadgroup]],
                                  uint simd [[simdgroup_index_in_threadgroup]],
                                  uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float qs[DEC_GROUP * DEC_DIM];
    threadgroup float sc[DEC_GROUP][DEC_SC];
    threadgroup float alpha[DEC_GROUP];
    uint len = min(tp.kv_len, p.capacity);
    if (tg.x * DEC_CHUNK >= len) {
        return;
    }
    uint kv = tg.y;
    uint stride = p.kv_heads * DEC_DIM;
    uint2 f = frag(lane);
    for (uint i = tid; i < DEC_GROUP * DEC_DIM; i += 256) {
        qs[i] = q[kv * DEC_GROUP * DEC_DIM + i] * p.scale;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    threadgroup const float *qh = qs;

    float m = DEC_NO_SCORE, l = 0;
    simdgroup_float8x8 o[4];
    for (uint i = 0; i < 4; i++) {
        o[i] = simdgroup_float8x8(0);
    }

    for (uint start = tg.x * DEC_CHUNK; start < len; start += p.splits * DEC_CHUNK) {
        uint n = min(DEC_CHUNK, len - start);

        // Scores: simdgroup s takes tokens 16s .. 16s + 15, two tiles of 8.
        {
            device const half *k = keys + (start + 16 * simd) * stride + kv * DEC_DIM;
            simdgroup_float8x8 s0 = simdgroup_float8x8(0), s1 = simdgroup_float8x8(0);
            for (uint d = 0; d < DEC_DIM; d += 8) {
                simdgroup_float8x8 qt, k0, k1;
                simdgroup_load(qt, qh + d, DEC_DIM, ulong2(0, 0), true);
                load_half_tile(k0, k + d, stride);
                load_half_tile(k1, k + 8 * stride + d, stride);
                simdgroup_multiply_accumulate(s0, k0, qt, s0);
                simdgroup_multiply_accumulate(s1, k1, qt, s1);
            }
            simdgroup_store(s0, &sc[0][16 * simd], DEC_SC, ulong2(0, 0), true);
            simdgroup_store(s1, &sc[0][16 * simd + 8], DEC_SC, ulong2(0, 0), true);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        chunk_softmax(sc[simd], n, 1.0f, m, l, alpha, simd, lane);
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Values: simdgroup s takes dims 32s .. 32s + 31, four tiles of 8.
        {
            float a = alpha[f.x];
            for (uint i = 0; i < 4; i++) {
                scale_tile(o[i], a);
            }
            device const half *v = values + start * stride + kv * DEC_DIM + 32 * simd;
            for (uint t = 0; t < DEC_CHUNK; t += 8) {
                simdgroup_float8x8 pt, vt[4];
                simdgroup_load(pt, &sc[0][t], DEC_SC);
                for (uint i = 0; i < 4; i++) {
                    load_half_tile(vt[i], v + t * stride + 8 * i, stride);
                }
                for (uint i = 0; i < 4; i++) {
                    simdgroup_multiply_accumulate(o[i], pt, vt[i], o[i]);
                }
            }
        }
        // The next chunk's scores overwrite sc and alpha.
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    write_ml(part_ml, p, tg.x, kv, simd, lane, m, l);
    device float *out = partial_out(part_o, p, tg.x, kv, simd);
    for (uint i = 0; i < 4; i++) {
        simdgroup_store(o[i], out + 8 * i, DEC_DIM);
    }
}

// attention_partial_mma over an 8- or 4-bit cache (kernels/kv_quant.metal
// layout): per token and kv head, 4 groups of 64 values, x = s q + z with
// half s and z; group G = token * per_token + kv * 4 + dim / 64 holds its
// codes in words 2 BITS G .., lowest bits first. Each tile is dequantized
// as it is loaded: a lane builds its two values, row f.x and columns f.y and
// f.y + 1. A tile row never leaves its group, so a lane loads the scale and
// bias of a row once per group, not once per tile.
template <uint BITS>
inline void set_quant_pair(thread simdgroup_float8x8 &m, device const uint *words, uint i,
                           float s, float z) {
    constexpr uint PER_WORD = 32 / BITS;
    constexpr uint MASK = (1u << BITS) - 1;
    uint word = words[i / PER_WORD];
    uint shift = (i % PER_WORD) * BITS;
    m.thread_elements()[0] = s * float((word >> shift) & MASK) + z;
    m.thread_elements()[1] = s * float((word >> (shift + BITS)) & MASK) + z;
}

template <uint BITS>
kernel void attention_partial_mma_q(device const float *q [[buffer(0)]],
                                    device const uint *keys [[buffer(1)]],
                                    device const uint *values [[buffer(2)]],
                                    device const half *key_scales [[buffer(3)]],
                                    device const half *key_biases [[buffer(4)]],
                                    device const half *value_scales [[buffer(5)]],
                                    device const half *value_biases [[buffer(6)]],
                                    device float *part_o [[buffer(7)]],
                                    device float *part_ml [[buffer(8)]],
                                    constant DecodeParams &p [[buffer(9)]],
                                    constant TokenParams &tp [[buffer(10)]],
                                    uint2 tg [[threadgroup_position_in_grid]],
                                    uint tid [[thread_index_in_threadgroup]],
                                    uint simd [[simdgroup_index_in_threadgroup]],
                                    uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float qs[DEC_GROUP * DEC_DIM];
    threadgroup float sc[DEC_GROUP][DEC_SC];
    threadgroup float alpha[DEC_GROUP];
    uint len = min(tp.kv_len, p.capacity);
    if (tg.x * DEC_CHUNK >= len) {
        return;
    }
    uint kv = tg.y;
    uint per_token = p.kv_heads * 4;
    constexpr uint WORDS = 2 * BITS;  // words per group
    uint2 f = frag(lane);
    for (uint i = tid; i < DEC_GROUP * DEC_DIM; i += 256) {
        qs[i] = q[kv * DEC_GROUP * DEC_DIM + i] * p.scale;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    threadgroup const float *qh = qs;

    float m = DEC_NO_SCORE, l = 0;
    simdgroup_float8x8 o[4];
    for (uint i = 0; i < 4; i++) {
        o[i] = simdgroup_float8x8(0);
    }

    for (uint start = tg.x * DEC_CHUNK; start < len; start += p.splits * DEC_CHUNK) {
        uint n = min(DEC_CHUNK, len - start);

        // Scores: simdgroup s takes tokens 16s .. 16s + 15, two tiles of 8;
        // this lane's rows are tokens ta and ta + 8.
        {
            uint ga = (start + 16 * simd + f.x) * per_token + kv * 4;
            uint gb = ga + 8 * per_token;
            simdgroup_float8x8 s0 = simdgroup_float8x8(0), s1 = simdgroup_float8x8(0);
            for (uint grp = 0; grp < 4; grp++) {
                float sa = float(key_scales[ga + grp]), za = float(key_biases[ga + grp]);
                float sb = float(key_scales[gb + grp]), zb = float(key_biases[gb + grp]);
                device const uint *wa = keys + WORDS * (ga + grp);
                device const uint *wb = keys + WORDS * (gb + grp);
                for (uint d8 = 0; d8 < 64; d8 += 8) {
                    simdgroup_float8x8 qt, k0, k1;
                    simdgroup_load(qt, qh + 64 * grp + d8, DEC_DIM, ulong2(0, 0), true);
                    set_quant_pair<BITS>(k0, wa, d8 + f.y, sa, za);
                    set_quant_pair<BITS>(k1, wb, d8 + f.y, sb, zb);
                    simdgroup_multiply_accumulate(s0, k0, qt, s0);
                    simdgroup_multiply_accumulate(s1, k1, qt, s1);
                }
            }
            simdgroup_store(s0, &sc[0][16 * simd], DEC_SC, ulong2(0, 0), true);
            simdgroup_store(s1, &sc[0][16 * simd + 8], DEC_SC, ulong2(0, 0), true);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        chunk_softmax(sc[simd], n, 1.0f, m, l, alpha, simd, lane);
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Values: simdgroup s takes dims 32s .. 32s + 31, four tiles of 8,
        // all in group s / 2 of the head, from value 32 (s % 2) of it.
        {
            float a = alpha[f.x];
            for (uint i = 0; i < 4; i++) {
                scale_tile(o[i], a);
            }
            uint first = 32 * (simd % 2) + f.y;
            for (uint t = 0; t < DEC_CHUNK; t += 8) {
                uint g = (start + t + f.x) * per_token + kv * 4 + simd / 2;
                float s = float(value_scales[g]), z = float(value_biases[g]);
                device const uint *w = values + WORDS * g;
                simdgroup_float8x8 pt, vt[4];
                simdgroup_load(pt, &sc[0][t], DEC_SC);
                for (uint i = 0; i < 4; i++) {
                    set_quant_pair<BITS>(vt[i], w, first + 8 * i, s, z);
                }
                for (uint i = 0; i < 4; i++) {
                    simdgroup_multiply_accumulate(o[i], pt, vt[i], o[i]);
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    write_ml(part_ml, p, tg.x, kv, simd, lane, m, l);
    device float *out = partial_out(part_o, p, tg.x, kv, simd);
    for (uint i = 0; i < 4; i++) {
        simdgroup_store(o[i], out + 8 * i, DEC_DIM);
    }
}

template [[host_name("attention_partial_mma_q8")]] kernel decltype(attention_partial_mma_q<8>) attention_partial_mma_q<8>;
template [[host_name("attention_partial_mma_q4")]] kernel decltype(attention_partial_mma_q<4>) attention_partial_mma_q<4>;

// attention_reduce: grid (heads, DEC_DIM / 64), 1024 threads. Thread
// (phase, d) merges splits phase, phase + 16, ... for dim d of its 64-dim
// slice with a running max, then the 16 phases are merged. Splits that got
// no chunk (short contexts) wrote nothing and are not read.
constant uint RED_PHASES = 16;

kernel void attention_reduce(device const float *part_o [[buffer(0)]],
                             device const float *part_ml [[buffer(1)]],
                             device const float *q_and_gate [[buffer(2)]],
                             device float *out [[buffer(3)]],
                             constant DecodeParams &p [[buffer(4)]],
                             constant TokenParams &tp [[buffer(5)]],
                             uint2 tg [[threadgroup_position_in_grid]],
                             uint tid [[thread_index_in_threadgroup]]) {
    threadgroup float sm[RED_PHASES][64], sl[RED_PHASES][64], sa[RED_PHASES][64];
    uint h = tg.x;
    uint dl = tid % 64, phase = tid / 64;
    uint d = tg.y * 64 + dl;
    uint len = min(tp.kv_len, p.capacity);
    uint parts = min(p.splits, (len + DEC_CHUNK - 1) / DEC_CHUNK);
    float m = -INFINITY, l = 0, acc = 0;
    for (uint c = phase; c < parts; c += RED_PHASES) {
        uint i = c * p.heads + h;
        float cm = part_ml[2 * i];
        float nm = max(m, cm);
        float a = exp(m - nm), b = exp(cm - nm);
        l = l * a + part_ml[2 * i + 1] * b;
        acc = acc * a + part_o[i * DEC_DIM + d] * b;
        m = nm;
    }
    sm[phase][dl] = m;
    sl[phase][dl] = l;
    sa[phase][dl] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (phase != 0) {
        return;
    }
    float mm = -INFINITY;
    for (uint f = 0; f < RED_PHASES; f++) {
        mm = max(mm, sm[f][dl]);
    }
    float tl = 0, ta = 0;
    for (uint f = 0; f < RED_PHASES; f++) {
        float w = sm[f][dl] == -INFINITY ? 0 : exp(sm[f][dl] - mm);
        tl += w * sl[f][dl];
        ta += w * sa[f][dl];
    }
    out[h * DEC_DIM + d] = ta / tl * sigmoid(q_and_gate[h * 2 * DEC_DIM + DEC_DIM + d]);
}

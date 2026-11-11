// Prefill attention over an 8- or 4-bit KV cache (src/kv_quant.rs): the
// kernels of prefill.metal with the keys and values read in place. The
// SIMDS heads of a threadgroup share one kv head, so each block of keys is
// dequantized once into threadgroup memory, 64 dims (one group) at a time,
// and the matrix products read it from there as they read a half cache.
//
// Buffers: 0 q, 1 key codes, 2 value codes, 3 key scales, 4 key biases,
// 5 value scales, 6 value biases, 7 q_and_gate, 8 out, 9 params.

constant uint KVQ_DIMS = 64;

// Rows row0 .. row0 + rows of one group column `col` (0 to 3) of kv head
// `kv`, dequantized to half into dst[row][64]. Rows at or past `limit`
// are written as 0, so masked scores times a value stay 0. All `threads`
// threads of the threadgroup take part.
template <uint BITS>
inline void kv_stage(device const uint *data, device const half *scales,
                     device const half *biases, uint kv_heads, uint kv, uint col,
                     uint row0, uint rows, uint limit, threadgroup half *dst,
                     uint tid, uint threads) {
    uint per_token = kv_heads * PREFILL_ATT_DIM / KVQ_DIMS;
    uint group = kv * (PREFILL_ATT_DIM / KVQ_DIMS) + col;
    for (uint i = tid; i < rows * 16; i += threads) {
        uint row = i / 16, quad = i % 16;
        uint token = row0 + row;
        float4 x = token < limit
                       ? kv_load4<BITS>(data, scales, biases, token * per_token + group, 4 * quad)
                       : float4(0);
        *(threadgroup half4 *)(dst + row * KVQ_DIMS + 4 * quad) = half4(x);
    }
}

// As kv_stage, all 256 dims of each row: dst[row * stride + dim], for
// kernels that keep a whole block of keys (stride 256 + padding).
template <uint BITS>
inline void kv_stage_rows(device const uint *data, device const half *scales,
                          device const half *biases, uint kv_heads, uint kv, uint row0,
                          uint rows, uint limit, threadgroup half *dst, uint stride, uint tid,
                          uint threads) {
    uint per_token = kv_heads * PREFILL_ATT_DIM / KVQ_DIMS;
    uint first = kv * (PREFILL_ATT_DIM / KVQ_DIMS);
    constexpr uint QUADS = PREFILL_ATT_DIM / 4;
    for (uint i = tid; i < rows * QUADS; i += threads) {
        uint row = i / QUADS, dim = 4 * (i % QUADS);
        uint token = row0 + row;
        float4 x = token < limit ? kv_load4<BITS>(data, scales, biases,
                                                  token * per_token + first + dim / KVQ_DIMS,
                                                  dim % KVQ_DIMS)
                                 : float4(0);
        *(threadgroup half4 *)(dst + row * stride + dim) = half4(x);
    }
}

// attention_rows2 (prefill.metal), keys and values staged as above.
template <typename T, uint SIMDS, uint BITS>
kernel void attention_rows2_q(device const T *q [[buffer(0)]],
                              device const uint *keys [[buffer(1)]],
                              device const uint *values [[buffer(2)]],
                              device const half *key_scales [[buffer(3)]],
                              device const half *key_biases [[buffer(4)]],
                              device const half *value_scales [[buffer(5)]],
                              device const half *value_biases [[buffer(6)]],
                              device const float *q_and_gate [[buffer(7)]],
                              device float *out [[buffer(8)]],
                              constant PrefillAttentionParams &p [[buffer(9)]],
                              uint2 tg [[threadgroup_position_in_grid]],
                              uint tid [[thread_index_in_threadgroup]],
                              uint simd [[simdgroup_index_in_threadgroup]],
                              uint lane [[thread_index_in_simdgroup]]) {
    constexpr uint KB = PREFILL_ATT2_KB;
    constexpr uint D = PREFILL_ATT_DIM;
    constexpr uint THREADS = SIMDS * 32;
    threadgroup float s_all[SIMDS][8 * KB];
    threadgroup T p_all[SIMDS][8 * KB];
    threadgroup float a_all[SIMDS][8];
    threadgroup half staged[KB * KVQ_DIMS];
    threadgroup float *sc = s_all[simd];
    threadgroup T *pt = p_all[simd];
    threadgroup float *al = a_all[simd];

    uint h = tg.y * SIMDS + simd;
    uint kv = h / (p.heads / p.kv_heads);
    uint q0 = tg.x * 8;
    uint q_stride = p.heads * D;
    uint end = p.pos + min(q0 + 8, p.tokens);

    simdgroup_matrix<T, 8, 8> qm[D / 8];
    {
        device const T *qh = q + ulong(q0) * q_stride + h * D;
        for (uint d = 0; d < D / 8; d++) {
            simdgroup_load(qm[d], qh + 8 * d, q_stride);
        }
    }

    constexpr uint PER = KB / 4;
    uint r = lane / 4;
    uint c0 = (lane % 4) * PER;
    uint last = p.pos + q0 + r;
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
        for (uint col = 0; col < D / KVQ_DIMS; col++) {
            threadgroup_barrier(mem_flags::mem_threadgroup);
            kv_stage<BITS>(keys, key_scales, key_biases, p.kv_heads, kv, col, k0, KB, end,
                           staged, tid, THREADS);
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint d = 0; d < KVQ_DIMS / 8; d++) {
                for (uint j = 0; j < KB / 8; j++) {
                    simdgroup_half8x8 km;
                    simdgroup_load(km, staged + (8 * j) * KVQ_DIMS + 8 * d, KVQ_DIMS,
                                   ulong2(0, 0), true);
                    simdgroup_multiply_accumulate(s[j], qm[col * (KVQ_DIMS / 8) + d],
                                                  prefill_att2_widen<T>(km), s[j]);
                }
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
        for (uint col = 0; col < D / KVQ_DIMS; col++) {
            threadgroup_barrier(mem_flags::mem_threadgroup);
            kv_stage<BITS>(values, value_scales, value_biases, p.kv_heads, kv, col, k0, KB, end,
                           staged, tid, THREADS);
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint d = 0; d < KVQ_DIMS / 8; d++) {
                uint dd = col * (KVQ_DIMS / 8) + d;
                thread auto &oe = o[dd].thread_elements();
                oe[0] *= a;
                oe[1] *= a;
                for (uint j = 0; j < KB / 8; j++) {
                    simdgroup_half8x8 vm;
                    simdgroup_load(vm, staged + (8 * j) * KVQ_DIMS + 8 * d, KVQ_DIMS);
                    simdgroup_multiply_accumulate(o[dd], pm[j], prefill_att2_widen<T>(vm), o[dd]);
                }
            }
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
    }

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

template [[host_name("attention_rows2_half_q8")]] kernel decltype(attention_rows2_q<half, 4, 8>) attention_rows2_q<half, 4, 8>;
template [[host_name("attention_rows2_half_q4")]] kernel decltype(attention_rows2_q<half, 4, 4>) attention_rows2_q<half, 4, 4>;

// attention_rows (prefill.metal, the exact path), keys and values staged
// as above.
template <typename T, uint BITS>
kernel void attention_rows_q(device const T *q [[buffer(0)]],
                             device const uint *keys [[buffer(1)]],
                             device const uint *values [[buffer(2)]],
                             device const half *key_scales [[buffer(3)]],
                             device const half *key_biases [[buffer(4)]],
                             device const half *value_scales [[buffer(5)]],
                             device const half *value_biases [[buffer(6)]],
                             device const float *q_and_gate [[buffer(7)]],
                             device float *out [[buffer(8)]],
                             constant PrefillAttentionParams &p [[buffer(9)]],
                             uint2 tg [[threadgroup_position_in_grid]],
                             uint tid [[thread_index_in_threadgroup]],
                             uint simd [[simdgroup_index_in_threadgroup]],
                             uint lane [[thread_index_in_simdgroup]]) {
    constexpr uint KB = PREFILL_ATT_KB;
    constexpr uint D = PREFILL_ATT_DIM;
    constexpr uint THREADS = PREFILL_ATT_SIMDS * 32;
    threadgroup float s_all[PREFILL_ATT_SIMDS][8 * KB];
    threadgroup T p_all[PREFILL_ATT_SIMDS][8 * KB];
    threadgroup float a_all[PREFILL_ATT_SIMDS][8];
    threadgroup half staged[KB * KVQ_DIMS];
    threadgroup float *sc = s_all[simd];
    threadgroup T *pt = p_all[simd];
    threadgroup float *al = a_all[simd];

    uint h = tg.y * PREFILL_ATT_SIMDS + simd;
    uint kv = h / (p.heads / p.kv_heads);
    uint q0 = tg.x * 8;
    uint q_stride = p.heads * D;
    device const T *qh = q + ulong(q0) * q_stride + h * D;
    uint end = p.pos + min(q0 + 8, p.tokens);

    uint r = lane / 4;
    uint c0 = (lane % 4) * 8;
    uint last = p.pos + q0 + r;
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
        for (uint col = 0; col < D / KVQ_DIMS; col++) {
            threadgroup_barrier(mem_flags::mem_threadgroup);
            kv_stage<BITS>(keys, key_scales, key_biases, p.kv_heads, kv, col, k0, KB, end,
                           staged, tid, THREADS);
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint d = 0; d < KVQ_DIMS; d += 8) {
                simdgroup_matrix<T, 8, 8> qm;
                simdgroup_load(qm, qh + col * KVQ_DIMS + d, q_stride);
                for (uint j = 0; j < KB / 8; j++) {
                    simdgroup_half8x8 km;
                    simdgroup_load(km, staged + (8 * j) * KVQ_DIMS + d, KVQ_DIMS, ulong2(0, 0),
                                   true);
                    simdgroup_multiply_accumulate(s[j], qm, prefill_from_half<T>(km), s[j]);
                }
            }
        }
        for (uint j = 0; j < KB / 8; j++) {
            simdgroup_store(s[j], sc + 8 * j, KB);
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);

        float v[8];
        float bm = -INFINITY;
        for (uint i = 0; i < 8; i++) {
            uint key = k0 + c0 + i;
            v[i] = key <= last && key < end ? sc[r * KB + c0 + i] * p.scale : -INFINITY;
            bm = max(bm, v[i]);
        }
        bm = max(bm, simd_shuffle_xor(bm, 1));
        bm = max(bm, simd_shuffle_xor(bm, 2));
        float m_new = max(m, bm);
        float sum = 0;
        for (uint i = 0; i < 8; i++) {
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
        for (uint col = 0; col < D / KVQ_DIMS; col++) {
            threadgroup_barrier(mem_flags::mem_threadgroup);
            kv_stage<BITS>(values, value_scales, value_biases, p.kv_heads, kv, col, k0, KB, end,
                           staged, tid, THREADS);
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint d = 0; d < KVQ_DIMS / 8; d++) {
                uint dd = col * (KVQ_DIMS / 8) + d;
                thread auto &oe = o[dd].thread_elements();
                oe[0] *= a;
                oe[1] *= a;
                for (uint j = 0; j < KB / 8; j++) {
                    simdgroup_half8x8 vm;
                    simdgroup_load(vm, staged + (8 * j) * KVQ_DIMS + 8 * d, KVQ_DIMS);
                    simdgroup_multiply_accumulate(o[dd], pm[j], prefill_from_half<T>(vm), o[dd]);
                }
            }
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
    }

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

template [[host_name("attention_rows_q8")]] kernel decltype(attention_rows_q<float, 8>) attention_rows_q<float, 8>;
template [[host_name("attention_rows_q4")]] kernel decltype(attention_rows_q<float, 4>) attention_rows_q<float, 4>;

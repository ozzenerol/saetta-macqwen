// Split-K attention for one decode token (flash-decoding).
//
// attention_partial: one threadgroup per (128-token chunk, kv head). The 8
// query heads that share the kv head are done together, so each key and
// value is read once for all 8. Writes per (chunk, q head) the unnormalised
// output sum(exp(s - m) v), its max score m and sum(exp(s - m)) l.
// attention_reduce: one threadgroup per q head merges the chunks and applies
// the sigmoid output gate.
//
// The cache is f32, token-major, kv_heads * 256 floats per token, the layout
// kv_store writes. Head dim 256 and 8 q heads per kv head are fixed here;
// src/attention.rs checks them.

constant uint ATT_CHUNK = 128;
constant uint ATT_GROUP = 8;
constant uint ATT_DIM = 256;

struct DecodeParams {
    uint heads;
    uint kv_heads;
    uint capacity;
    float scale;
};

kernel void attention_partial(device const float *q [[buffer(0)]],
                              device const float *keys [[buffer(1)]],
                              device const float *values [[buffer(2)]],
                              device float *part_o [[buffer(3)]],
                              device float *part_ml [[buffer(4)]],
                              constant DecodeParams &p [[buffer(5)]],
                              constant TokenParams &tp [[buffer(6)]],
                              uint2 tg [[threadgroup_position_in_grid]],
                              uint tid [[thread_index_in_threadgroup]],
                              uint simd [[simdgroup_index_in_threadgroup]],
                              uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float sc[ATT_GROUP][ATT_CHUNK];
    uint len = min(tp.kv_len, p.capacity);
    uint start = tg.x * ATT_CHUNK;
    if (start >= len) {
        return;
    }
    uint n = min(ATT_CHUNK, len - start);
    uint kv = tg.y;
    uint stride = p.kv_heads * ATT_DIM;

    // Lane l holds dims 8l..8l+7 of each of the 8 queries, pre-scaled.
    float qv[ATT_GROUP][8];
    for (uint g = 0; g < ATT_GROUP; g++) {
        device const float4 *q4 = (device const float4 *)(q + (kv * ATT_GROUP + g) * ATT_DIM + lane * 8);
        float4 a = q4[0] * p.scale, b = q4[1] * p.scale;
        qv[g][0] = a.x; qv[g][1] = a.y; qv[g][2] = a.z; qv[g][3] = a.w;
        qv[g][4] = b.x; qv[g][5] = b.y; qv[g][6] = b.z; qv[g][7] = b.w;
    }

    // Scores: simdgroup s takes tokens s, s + 8, ...
    for (uint t = simd; t < n; t += 8) {
        device const float4 *k4 =
            (device const float4 *)(keys + (start + t) * stride + kv * ATT_DIM + lane * 8);
        float4 a = k4[0], b = k4[1];
        for (uint g = 0; g < ATT_GROUP; g++) {
            float d = qv[g][0] * a.x + qv[g][1] * a.y + qv[g][2] * a.z + qv[g][3] * a.w +
                      qv[g][4] * b.x + qv[g][5] * b.y + qv[g][6] * b.z + qv[g][7] * b.w;
            d = simd_sum(d);
            if (lane == 0) {
                sc[g][t] = d;
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Softmax pieces: simdgroup g takes query head g.
    {
        uint g = simd;
        float m = -INFINITY;
        for (uint t = lane; t < n; t += 32) {
            m = max(m, sc[g][t]);
        }
        m = simd_max(m);
        float l = 0;
        for (uint t = lane; t < n; t += 32) {
            float e = exp(sc[g][t] - m);
            sc[g][t] = e;
            l += e;
        }
        l = simd_sum(l);
        if (lane == 0) {
            uint i = tg.x * p.heads + kv * ATT_GROUP + g;
            part_ml[2 * i] = m;
            part_ml[2 * i + 1] = l;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Weighted values: thread tid owns dim tid for all 8 heads.
    float acc[ATT_GROUP] = {0};
    device const float *v = values + start * stride + kv * ATT_DIM + tid;
    for (uint t = 0; t < n; t++) {
        float x = v[t * stride];
        for (uint g = 0; g < ATT_GROUP; g++) {
            acc[g] += sc[g][t] * x;
        }
    }
    for (uint g = 0; g < ATT_GROUP; g++) {
        part_o[(tg.x * p.heads + kv * ATT_GROUP + g) * ATT_DIM + tid] = acc[g];
    }
}

kernel void attention_reduce(device const float *part_o [[buffer(0)]],
                             device const float *part_ml [[buffer(1)]],
                             device const float *q_and_gate [[buffer(2)]],
                             device float *out [[buffer(3)]],
                             constant DecodeParams &p [[buffer(4)]],
                             constant TokenParams &tp [[buffer(5)]],
                             uint h [[threadgroup_position_in_grid]],
                             uint d [[thread_index_in_threadgroup]]) {
    uint len = min(tp.kv_len, p.capacity);
    uint chunks = (len + ATT_CHUNK - 1) / ATT_CHUNK;
    float m = -INFINITY;
    for (uint c = 0; c < chunks; c++) {
        m = max(m, part_ml[2 * (c * p.heads + h)]);
    }
    float l = 0, acc = 0;
    for (uint c = 0; c < chunks; c++) {
        uint i = c * p.heads + h;
        float w = exp(part_ml[2 * i] - m);
        l += w * part_ml[2 * i + 1];
        acc += w * part_o[i * ATT_DIM + d];
    }
    out[h * ATT_DIM + d] = acc / l * sigmoid(q_and_gate[h * 2 * ATT_DIM + ATT_DIM + d]);
}

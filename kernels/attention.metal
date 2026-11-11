// Split-K attention for one decode token (flash-decoding).
//
// attention_partial: one threadgroup per (128-token chunk, kv head). The 8
// query heads that share the kv head are done together, so each key and
// value is read once for all 8. Writes per (chunk, q head) the unnormalised
// output sum(exp(s - m) v), its max score m and sum(exp(s - m)) l.
// attention_reduce: one threadgroup per q head merges the chunks and applies
// the sigmoid output gate.
//
// The cache is half, token-major, kv_heads * 256 values per token, the
// layout kv_store writes. The math is f32. Head dim 256 and 8 q heads per kv head are fixed here;
// src/attention.rs checks them.

constant uint DEC_CHUNK = 128;
constant uint DEC_GROUP = 8;
constant uint DEC_DIM = 256;

struct DecodeParams {
    uint heads;
    uint kv_heads;
    uint capacity;
    float scale;
};

kernel void attention_partial(device const float *q [[buffer(0)]],
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
    threadgroup float sc[DEC_GROUP][DEC_CHUNK];
    uint len = min(tp.kv_len, p.capacity);
    uint start = tg.x * DEC_CHUNK;
    if (start >= len) {
        return;
    }
    uint n = min(DEC_CHUNK, len - start);
    uint kv = tg.y;
    uint stride = p.kv_heads * DEC_DIM;

    // Lane l holds dims 8l..8l+7 of each of the 8 queries, pre-scaled.
    float qv[DEC_GROUP][8];
    for (uint g = 0; g < DEC_GROUP; g++) {
        device const float4 *q4 = (device const float4 *)(q + (kv * DEC_GROUP + g) * DEC_DIM + lane * 8);
        float4 a = q4[0] * p.scale, b = q4[1] * p.scale;
        qv[g][0] = a.x; qv[g][1] = a.y; qv[g][2] = a.z; qv[g][3] = a.w;
        qv[g][4] = b.x; qv[g][5] = b.y; qv[g][6] = b.z; qv[g][7] = b.w;
    }

    // Scores: simdgroup s takes tokens s, s + 8, ...
    for (uint t = simd; t < n; t += 8) {
        device const half4 *k4 =
            (device const half4 *)(keys + (start + t) * stride + kv * DEC_DIM + lane * 8);
        float4 a = float4(k4[0]), b = float4(k4[1]);
        for (uint g = 0; g < DEC_GROUP; g++) {
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
            uint i = tg.x * p.heads + kv * DEC_GROUP + g;
            part_ml[2 * i] = m;
            part_ml[2 * i + 1] = l;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Weighted values: thread tid owns dims 4 * (tid % 64) .. + 3 of all 8
    // heads, for the tokens t with t % 4 == tid / 64. One half4 load per
    // token; the four token phases are summed through `sum` afterwards.
    threadgroup float4 sum[DEC_GROUP][DEC_DIM / 4];
    uint d4 = tid % 64, phase = tid / 64;
    float4 acc[DEC_GROUP] = {0};
    device const half4 *v = (device const half4 *)(values + start * stride + kv * DEC_DIM) + d4;
    for (uint t = phase; t < n; t += 4) {
        float4 x = float4(v[t * stride / 4]);
        for (uint g = 0; g < DEC_GROUP; g++) {
            acc[g] += sc[g][t] * x;
        }
    }
    for (uint step = 3; step > 0; step--) {
        if (phase == step) {
            for (uint g = 0; g < DEC_GROUP; g++) {
                sum[g][d4] = step == 3 ? acc[g] : sum[g][d4] + acc[g];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (phase == 0) {
        for (uint g = 0; g < DEC_GROUP; g++) {
            device float4 *o = (device float4 *)(part_o + (tg.x * p.heads + kv * DEC_GROUP + g) * DEC_DIM);
            o[d4] = sum[g][d4] + acc[g];
        }
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
    uint chunks = (len + DEC_CHUNK - 1) / DEC_CHUNK;
    float m = -INFINITY;
    for (uint c = 0; c < chunks; c++) {
        m = max(m, part_ml[2 * (c * p.heads + h)]);
    }
    float l = 0, acc = 0;
    for (uint c = 0; c < chunks; c++) {
        uint i = c * p.heads + h;
        float w = exp(part_ml[2 * i] - m);
        l += w * part_ml[2 * i + 1];
        acc += w * part_o[i * DEC_DIM + d];
    }
    out[h * DEC_DIM + d] = acc / l * sigmoid(q_and_gate[h * 2 * DEC_DIM + DEC_DIM + d]);
}

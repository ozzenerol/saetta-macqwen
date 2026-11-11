// The two 8-bit gate matrices of an MoE block in one dispatch:
//   out[0..rows)  = mlp.gate x               (router logits, one per expert)
//   out[rows]     = mlp.shared_expert_gate x (the shared expert's gate logit)
// One simdgroup per output row, so 257 rows make 257 simdgroups. qmv packs
// 4 rows per simdgroup, which leaves too few threads for a 257-row matrix.
// Splitting each row over 2 simdgroups measured no faster (7.1 vs 7.3 us).

constant uint ROUTER_SIMDS = 8;

struct RouterShape {
    uint rows;  // experts
    uint cols;  // hidden size
};

// NORM: x is RMS-normed first with nw (buffer 9) and eps (10), as qmv_norm.
template <bool NORM>
kernel void router_8bit(device const uint *gate_w [[buffer(0)]],
                        device const ushort *gate_s [[buffer(1)]],
                        device const ushort *gate_b [[buffer(2)]],
                        device const uint *shared_w [[buffer(3)]],
                        device const ushort *shared_s [[buffer(4)]],
                        device const ushort *shared_b [[buffer(5)]],
                        device const float *x [[buffer(6)]],
                        device float *out [[buffer(7)]],
                        constant RouterShape &shape [[buffer(8)]],
                        device const float *nw [[buffer(9)]],
                        constant float &eps [[buffer(10)]],
                        uint tg [[threadgroup_position_in_grid]],
                        uint simd [[simdgroup_index_in_threadgroup]],
                        uint lane [[thread_index_in_simdgroup]]) {
    uint row = tg * ROUTER_SIMDS + simd;
    if (row > shape.rows) {
        return;
    }
    bool shared = row == shape.rows;
    device const uint *w = shared ? shared_w : gate_w;
    device const ushort *s = shared ? shared_s : gate_s;
    device const ushort *b = shared ? shared_b : gate_b;
    uint r = shared ? 0 : row;
    uint row_words = shape.cols * 8 / 32;
    uint groups = shape.cols / 64;

    float acc = 0;
    float sq = 0;
    for (uint k = lane * 32; k < shape.cols; k += 32 * 32) {
        float xv[32];
        float xsum = NORM ? load_x32_norm(x + k, nw + k, xv, sq) : load_x32(x + k, xv);
        acc += dot32<8>(w + r * row_words + k / 4, s, b, r * groups + k / 64, xv, xsum);
    }
    float inv = NORM ? norm_inv(sq, shape.cols, eps) : 1.0f;
    float sum = simd_sum(acc) * inv;
    if (lane == 0) {
        out[row] = sum;
    }
}

template [[host_name("router_8bit")]] kernel decltype(router_8bit<false>) router_8bit<false>;
template [[host_name("router_norm_8bit")]] kernel decltype(router_8bit<true>) router_8bit<true>;

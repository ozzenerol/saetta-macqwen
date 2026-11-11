// The routed experts of one MoE block for one token, in two dispatches:
//   moe_gate_up: h[s] = silu(gate[e_s] x) * up[e_s] x for each slot s
//   moe_down:    y = sum over s of weight[s] * down[e_s] h[s]
// Expert ids and weights come from GPU buffers. Stacked expert matrices are
// [experts, rows, cols]. One simdgroup computes MOE_ROWS rows. 8 simdgroups of
// one row each measured best on the M3 Max (1, 2, 4 rows and 2, 4, 8
// simdgroups tried).

constant uint MOE_ROWS = 1;
constant uint MOE_SIMDS = 8;

struct MoeShape {
    uint rows;
    uint cols;
    uint count;
};

// Threadgroup row tg.y is the slot. x has `cols` values, h gets
// count * rows.
// NORM: x is RMS-normed first with nw (buffer 10) and eps (11), as qmv_norm.
template <uint BITS, bool NORM>
kernel void moe_gate_up(device const uint *gate_w [[buffer(0)]],
                        device const ushort *gate_s [[buffer(1)]],
                        device const ushort *gate_b [[buffer(2)]],
                        device const uint *up_w [[buffer(3)]],
                        device const ushort *up_s [[buffer(4)]],
                        device const ushort *up_b [[buffer(5)]],
                        device const float *x [[buffer(6)]],
                        device float *h [[buffer(7)]],
                        constant MoeShape &shape [[buffer(8)]],
                        device const uint *ids [[buffer(9)]],
                        device const float *nw [[buffer(10)]],
                        constant float &eps [[buffer(11)]],
                        uint2 tg [[threadgroup_position_in_grid]],
                        uint simd [[simdgroup_index_in_threadgroup]],
                        uint lane [[thread_index_in_simdgroup]]) {
    uint slot = tg.y;
    uint e = ids[slot];
    uint row0 = (tg.x * MOE_SIMDS + simd) * MOE_ROWS;
    uint row_words = shape.cols * BITS / 32;
    uint groups = shape.cols / 64;
    float g[MOE_ROWS] = {0};
    float u[MOE_ROWS] = {0};
    float sq = 0;

    for (uint k = lane * 32; k < shape.cols; k += 32 * 32) {
        float xv[32];
        float xsum = NORM ? load_x32_norm(x + k, nw + k, xv, sq) : load_x32(x + k, xv);
#pragma clang loop unroll(full)
        for (uint r = 0; r < MOE_ROWS; r++) {
            uint row = e * shape.rows + row0 + r;
            uint w = row * row_words + k * BITS / 32;
            uint grp = row * groups + k / 64;
            g[r] += dot32<BITS>(gate_w + w, gate_s, gate_b, grp, xv, xsum);
            u[r] += dot32<BITS>(up_w + w, up_s, up_b, grp, xv, xsum);
        }
    }

    float inv = NORM ? norm_inv(sq, shape.cols, eps) : 1.0f;
    for (uint r = 0; r < MOE_ROWS; r++) {
        float gs = simd_sum(g[r]) * inv;
        float us = simd_sum(u[r]) * inv;
        if (lane == 0) {
            h[slot * shape.rows + row0 + r] = gs / (1.0f + exp(-gs)) * us;
        }
    }
}

// h has count * cols values. The lanes walk the 32-value chunks of all
// slots together, so a short row (cols 512 is 16 chunks) still keeps all 32
// lanes busy, and the simd_sum adds up the slots as well as the chunks.
template <uint BITS>
kernel void moe_down(device const uint *words [[buffer(0)]],
                     device const ushort *scales [[buffer(1)]],
                     device const ushort *biases [[buffer(2)]],
                     device const float *h [[buffer(3)]],
                     device float *y [[buffer(4)]],
                     constant MoeShape &shape [[buffer(5)]],
                     device const uint *ids [[buffer(6)]],
                     device const float *weights [[buffer(7)]],
                     uint tg [[threadgroup_position_in_grid]],
                     uint simd [[simdgroup_index_in_threadgroup]],
                     uint lane [[thread_index_in_simdgroup]]) {
    uint row0 = (tg * MOE_SIMDS + simd) * MOE_ROWS;
    uint row_words = shape.cols * BITS / 32;
    uint groups = shape.cols / 64;
    uint chunks = shape.cols / 32;
    float acc[MOE_ROWS] = {0};

    for (uint c = lane; c < shape.count * chunks; c += 32) {
        uint slot = c / chunks;
        uint k = (c % chunks) * 32;
        uint e = ids[slot];
        float weight = weights[slot];
        float xv[32];
        float xsum = load_x32(h + slot * shape.cols + k, xv);
#pragma clang loop unroll(full)
        for (uint r = 0; r < MOE_ROWS; r++) {
            uint row = e * shape.rows + row0 + r;
            acc[r] += weight * dot32<BITS>(words + row * row_words + k * BITS / 32, scales,
                                           biases, row * groups + k / 64, xv, xsum);
        }
    }

    for (uint r = 0; r < MOE_ROWS; r++) {
        float sum = simd_sum(acc[r]);
        if (lane == 0) {
            y[row0 + r] = sum;
        }
    }
}

template [[host_name("moe_gate_up_5bit")]] kernel decltype(moe_gate_up<5, false>) moe_gate_up<5, false>;
template [[host_name("moe_gate_up_norm_5bit")]] kernel decltype(moe_gate_up<5, true>) moe_gate_up<5, true>;
template [[host_name("moe_down_5bit")]] kernel decltype(moe_down<5>) moe_down<5>;

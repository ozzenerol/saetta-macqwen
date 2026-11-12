
// ---------------------------------------------------------------------------
// The gated delta rule in chunks of WY_C tokens (the WY form), for prefill
// without snapshots, after delta_prep. Per value head and chunk, with
// gamma_i the product of g over the chunk up to token i, beta_i, and k, q
// and v rows of the chunk:
//   A = strictly lower (beta_i gamma_i / gamma_j k_i . k_j), T = (I + A)^-1
//   U = T diag(beta) V, W = T diag(beta gamma) K
//   P = lower incl. diagonal (gamma_i / gamma_j q_i . k_j)
// and then, from the state S at the chunk's start ([v_dim][k_dim]):
//   delta = U - W S^T, y = diag(gamma) Q S^T + P delta
//   S = gamma_C S + delta^T diag(gamma_C / gamma) K.
// delta_wy_prep makes W, U, P and log gamma for every chunk at once;
// delta_wy_state runs the chunks in order, 16 state rows per threadgroup.
// Needs k_dim = v_dim = 128.
constant uint WY_C = 64;
constant uint WY_D = 128;
// Floats per head and chunk: W [C][D], U [C][D], P [C][C], log gamma [C].
constant uint WY_SIZE = 2 * WY_C * WY_D + WY_C * WY_C + WY_C;

kernel void delta_wy_prep(device const float *mixed [[buffer(0)]],
                          device const float *g [[buffer(1)]],
                          device const float *beta [[buffer(2)]],
                          device float *wy [[buffer(3)]],
                          constant DeltaRowsParams &p [[buffer(4)]],
                          uint2 tg [[threadgroup_position_in_grid]],
                          uint tid [[thread_index_in_threadgroup]],
                          uint simd [[simdgroup_index_in_threadgroup]],
                          uint lane [[thread_index_in_simdgroup]]) {
    constexpr uint C = WY_C, D = WY_D;
    threadgroup float x[C * C];
    threadgroup float lgs[C], bs[C];

    uint c = tg.x, h = tg.y;
    uint chunks = (p.tokens + C - 1) / C;
    uint t0 = c * C;
    uint n = min(C, p.tokens - t0);
    uint kh = h / (p.v_heads / p.k_heads);
    uint key_size = p.k_heads * D;
    device const float *qb = mixed + ulong(t0) * p.channels + kh * D;
    device const float *kb = qb + key_size;
    device const float *vb = mixed + ulong(t0) * p.channels + 2 * key_size + h * D;
    device float *out = wy + ulong(h * chunks + c) * WY_SIZE;

    // log gamma as an inclusive prefix sum of log g; rows past the chunk's
    // end get g = 1 and beta = 0, so they add nothing.
    if (tid < C) {
        bool ok = tid < n;
        float gv = ok ? g[(t0 + tid) * p.v_heads + h] : 1.0f;
        lgs[tid] = simd_prefix_inclusive_sum(log(max(gv, 1e-30f)));
        bs[tid] = ok ? beta[(t0 + tid) * p.v_heads + h] : 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid >= 32 && tid < C) {
        lgs[tid] += lgs[31];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // K K^T, simdgroup s rows 8 s to 8 s + 7.
    {
        simdgroup_float8x8 acc[8];
        for (uint j = 0; j < 8; j++) {
            acc[j] = simdgroup_float8x8(0);
        }
        for (uint kk = 0; kk < D; kk += 8) {
            simdgroup_float8x8 a, b;
            simdgroup_load(a, kb + ulong(8 * simd) * p.channels + kk, p.channels);
            for (uint j = 0; j < 8; j++) {
                simdgroup_load(b, kb + ulong(8 * j) * p.channels + kk, p.channels, ulong2(0, 0), true);
                simdgroup_multiply_accumulate(acc[j], a, b, acc[j]);
            }
        }
        for (uint j = 0; j < 8; j++) {
            simdgroup_store(acc[j], x + 8 * simd * C + 8 * j, C);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint e = tid; e < C * C; e += 256) {
        uint i = e / C, j = e % C;
        x[e] = j < i ? bs[i] * exp(lgs[i] - lgs[j]) * x[e] : 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // T = (I + A)^-1 in place, row by row: T_i = e_i - sum_{j < i} A_ij T_j.
    // Lane l owns columns l and l + 32.
    if (simd == 0) {
        uint c0 = lane, c1 = lane + 32;
        for (uint i = 0; i < C; i++) {
            float r0 = 0, r1 = 0;
            for (uint j = 0; j < i; j++) {
                float aij = x[i * C + j];
                r0 -= aij * x[j * C + c0];
                r1 -= aij * x[j * C + c1];
            }
            simdgroup_barrier(mem_flags::mem_threadgroup);
            x[i * C + c0] = c0 < i ? r0 : (c0 == i ? 1.0f : 0.0f);
            x[i * C + c1] = c1 < i ? r1 : (c1 == i ? 1.0f : 0.0f);
            simdgroup_barrier(mem_flags::mem_threadgroup);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint e = tid; e < C * C; e += 256) {
        x[e] *= bs[e % C];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // U = T diag(beta) V, then W = T diag(beta gamma) K.
    for (uint pass = 0; pass < 2; pass++) {
        device const float *src = pass == 0 ? vb : kb;
        device float *dst = out + (pass == 0 ? C * D : 0);
        simdgroup_float8x8 acc[16];
        for (uint j = 0; j < 16; j++) {
            acc[j] = simdgroup_float8x8(0);
        }
        for (uint kk = 0; kk < C; kk += 8) {
            simdgroup_float8x8 a, b;
            simdgroup_load(a, x + 8 * simd * C + kk, C);
            for (uint j = 0; j < 16; j++) {
                simdgroup_load(b, src + ulong(kk) * p.channels + 8 * j, p.channels);
                simdgroup_multiply_accumulate(acc[j], a, b, acc[j]);
            }
        }
        for (uint j = 0; j < 16; j++) {
            simdgroup_store(acc[j], dst + 8 * simd * D + 8 * j, D);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (pass == 0) {
            for (uint e = tid; e < C * C; e += 256) {
                x[e] *= exp(lgs[e % C]);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }

    // P = Q K^T, masked to j <= i and scaled by gamma_i / gamma_j.
    {
        simdgroup_float8x8 acc[8];
        for (uint j = 0; j < 8; j++) {
            acc[j] = simdgroup_float8x8(0);
        }
        for (uint kk = 0; kk < D; kk += 8) {
            simdgroup_float8x8 a, b;
            simdgroup_load(a, qb + ulong(8 * simd) * p.channels + kk, p.channels);
            for (uint j = 0; j < 8; j++) {
                simdgroup_load(b, kb + ulong(8 * j) * p.channels + kk, p.channels, ulong2(0, 0), true);
                simdgroup_multiply_accumulate(acc[j], a, b, acc[j]);
            }
        }
        uint2 fp = frag_pos(lane);
        uint i = 8 * simd + fp.x;
        for (uint j = 0; j < 8; j++) {
            thread auto &e = acc[j].thread_elements();
            for (uint u = 0; u < 2; u++) {
                uint jj = 8 * j + fp.y + u;
                e[u] = jj <= i ? e[u] * exp(lgs[i] - lgs[jj]) : 0.0f;
            }
            simdgroup_store(acc[j], out + 2 * C * D + 8 * simd * C + 8 * j, C);
        }
    }
    if (tid < C) {
        out[2 * C * D + C * C + tid] = lgs[tid];
    }
}

constant uint WY_R = 16;  // state rows per delta_wy_state threadgroup

kernel void delta_wy_state(device const float *mixed [[buffer(0)]],
                           device const float *wy [[buffer(1)]],
                           device float *state [[buffer(2)]],
                           device float *y [[buffer(3)]],
                           constant DeltaRowsParams &p [[buffer(4)]],
                           uint2 tg [[threadgroup_position_in_grid]],
                           uint tid [[thread_index_in_threadgroup]],
                           uint simd [[simdgroup_index_in_threadgroup]],
                           uint lane [[thread_index_in_simdgroup]]) {
    constexpr uint C = WY_C, D = WY_D, R = WY_R;
    threadgroup float st[D * R];  // S^T of this threadgroup's rows, [k][r]
    threadgroup float dl[C * R];  // delta, [token][r]
    threadgroup float lgs[C];

    uint h = tg.y, r0 = tg.x * R;
    uint chunks = (p.tokens + C - 1) / C;
    uint kh = h / (p.v_heads / p.k_heads);
    uint key_size = p.k_heads * D;
    uint value_size = p.v_heads * D;
    device float *sp = state + (ulong(h) * D + r0) * D;
    for (uint e = tid; e < R * D; e += 128) {
        st[(e % D) * R + e / D] = sp[e];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint2 fp = frag_pos(lane);

    for (uint c = 0; c < chunks; c++) {
        uint t0 = c * C;
        uint n = min(C, p.tokens - t0);
        device const float *w = wy + ulong(h * chunks + c) * WY_SIZE;
        device const float *u = w + C * D;
        device const float *pm = u + C * D;
        device const float *qb = mixed + ulong(t0) * p.channels + kh * D;
        device const float *kb = qb + key_size;
        if (tid < C) {
            lgs[tid] = pm[C * C + tid];
        }

        // delta = U - W S^T; simdgroup s rows 16 s to 16 s + 15.
        simdgroup_float8x8 acc[2][2];
        for (uint i = 0; i < 2; i++) {
            for (uint j = 0; j < 2; j++) {
                acc[i][j] = simdgroup_float8x8(0);
            }
        }
        for (uint kk = 0; kk < D; kk += 8) {
            simdgroup_float8x8 a[2], b[2];
            for (uint i = 0; i < 2; i++) {
                simdgroup_load(a[i], w + (16 * simd + 8 * i) * D + kk, D);
                simdgroup_load(b[i], st + kk * R + 8 * i, R);
            }
            for (uint i = 0; i < 2; i++) {
                for (uint j = 0; j < 2; j++) {
                    simdgroup_multiply_accumulate(acc[i][j], a[i], b[j], acc[i][j]);
                }
            }
        }
        for (uint i = 0; i < 2; i++) {
            for (uint j = 0; j < 2; j++) {
                simdgroup_float8x8 um;
                simdgroup_load(um, u + (16 * simd + 8 * i) * D + r0 + 8 * j, D);
                thread auto &ue = um.thread_elements();
                thread auto &ae = acc[i][j].thread_elements();
                ae[0] = ue[0] - ae[0];
                ae[1] = ue[1] - ae[1];
                simdgroup_store(acc[i][j], dl + (16 * simd + 8 * i) * R + 8 * j, R);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // y = diag(gamma) Q S^T + P delta.
        for (uint i = 0; i < 2; i++) {
            for (uint j = 0; j < 2; j++) {
                acc[i][j] = simdgroup_float8x8(0);
            }
        }
        for (uint kk = 0; kk < D; kk += 8) {
            simdgroup_float8x8 a[2], b[2];
            for (uint i = 0; i < 2; i++) {
                simdgroup_load(a[i], qb + ulong(16 * simd + 8 * i) * p.channels + kk, p.channels);
                simdgroup_load(b[i], st + kk * R + 8 * i, R);
            }
            for (uint i = 0; i < 2; i++) {
                for (uint j = 0; j < 2; j++) {
                    simdgroup_multiply_accumulate(acc[i][j], a[i], b[j], acc[i][j]);
                }
            }
        }
        for (uint i = 0; i < 2; i++) {
            float gm = exp(lgs[16 * simd + 8 * i + fp.x]);
            for (uint j = 0; j < 2; j++) {
                thread auto &ae = acc[i][j].thread_elements();
                ae[0] *= gm;
                ae[1] *= gm;
            }
        }
        for (uint kk = 0; kk < C; kk += 8) {
            simdgroup_float8x8 a[2], b[2];
            for (uint i = 0; i < 2; i++) {
                simdgroup_load(a[i], pm + (16 * simd + 8 * i) * C + kk, C);
                simdgroup_load(b[i], dl + kk * R + 8 * i, R);
            }
            for (uint i = 0; i < 2; i++) {
                for (uint j = 0; j < 2; j++) {
                    simdgroup_multiply_accumulate(acc[i][j], a[i], b[j], acc[i][j]);
                }
            }
        }
        for (uint i = 0; i < 2; i++) {
            uint row = 16 * simd + 8 * i + fp.x;
            if (row >= n) {
                continue;
            }
            device float *yrow = y + ulong(t0 + row) * value_size + h * D + r0;
            for (uint j = 0; j < 2; j++) {
                thread auto &ae = acc[i][j].thread_elements();
                yrow[8 * j + fp.y] = ae[0];
                yrow[8 * j + fp.y + 1] = ae[1];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float lc = lgs[C - 1];
        for (uint e = tid; e < C * R; e += 128) {
            dl[e] *= exp(lc - lgs[e / R]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // S^T = gamma_C S^T + K^T delta; simdgroup s rows 32 s to 32 s + 31.
        float gc = exp(lc);
        for (uint i = 0; i < 4; i++) {
            uint kr = 8 * (4 * simd + i);
            simdgroup_float8x8 s2[2];
            for (uint j = 0; j < 2; j++) {
                simdgroup_load(s2[j], st + kr * R + 8 * j, R);
                thread auto &se = s2[j].thread_elements();
                se[0] *= gc;
                se[1] *= gc;
            }
            for (uint kk = 0; kk < C; kk += 8) {
                simdgroup_float8x8 a, b;
                simdgroup_load(a, kb + ulong(kk) * p.channels + kr, p.channels, ulong2(0, 0), true);
                for (uint j = 0; j < 2; j++) {
                    simdgroup_load(b, dl + kk * R + 8 * j, R);
                    simdgroup_multiply_accumulate(s2[j], a, b, s2[j]);
                }
            }
            for (uint j = 0; j < 2; j++) {
                simdgroup_store(s2[j], st + kr * R + 8 * j, R);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    for (uint e = tid; e < R * D; e += 128) {
        sp[e] = st[(e % D) * R + e / D];
    }
}

// Zeros the rows of `mixed` from p.tokens up to the next multiple of WY_C,
// which the chunk products read: so stale values there can't make a NaN.
kernel void delta_wy_zero_tail(device float *mixed [[buffer(0)]],
                               constant DeltaRowsParams &p [[buffer(4)]],
                               uint i [[thread_position_in_grid]]) {
    uint pad = (WY_C - p.tokens % WY_C) % WY_C;
    if (i < pad * p.channels) {
        mixed[ulong(p.tokens) * p.channels + i] = 0.0f;
    }
}

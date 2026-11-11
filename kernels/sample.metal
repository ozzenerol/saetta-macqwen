// Sampling on the GPU, the same steps as src/sample.rs: top_k and top_p cut
// the distribution at temperature 1, then a token is drawn from what is left
// at the temperature, with the same splitmix64 stream. Two dispatches:
//   sample_partial: SAMPLE_GROUPS threadgroups of 256 threads each take a
//     slice of the logits and write its top_k (best first, lower id on ties)
//     and its max and sum(exp(l - max)) for the top_p total.
//   sample_final: one threadgroup merges the candidates, applies top_p and
//     draws. It writes the token and its prob into the ring and, when asked,
//     advances TokenParams so the next token can run without the CPU.
// Temperature 0 takes the best token without drawing, as Sampler::sample.
// With a draft d in the slot (a greedy draft, so its own prob is 1) the
// speculative rule runs here: keep d with prob p(d), otherwise draw from p
// with d removed. The token is then distributed as p either way.
// top_k must be 1..SAMPLE_MAX_K; src/sample_gpu.rs checks that.

constant uint SAMPLE_GROUPS = 256;
constant uint SAMPLE_SIMDS = 8;
constant uint SAMPLE_PER_THREAD_P = 4;  // 256 groups * 256 threads * 4 = 262144 >= vocab
constant uint SAMPLE_SLICE = SAMPLE_SIMDS * 32 * SAMPLE_PER_THREAD_P;
constant uint SAMPLE_MAX_K = 64;
constant uint SAMPLE_PER_THREAD = SAMPLE_GROUPS * SAMPLE_MAX_K / 1024;
constant uint SAMPLE_KEEP = 1024;  // candidates at or above the bound

struct SampleParams {
    uint n;
    uint top_k;
    float top_p;
    float temperature;
    uint advance;  // 1: write the token into TokenParams and step the position
};

// Shared with the next plans and with speculative decoding. Slot s holds the
// token fed at a position p with p & (SAMPLE_RING - 1) == s.
constant uint SAMPLE_RING = 64;
struct SampleRing {
    uint ids[SAMPLE_RING];
    float probs[SAMPLE_RING];    // prob of ids[s] under the cut distribution
    uint draft[SAMPLE_RING];     // a draft token to verify, or 0xffffffff
    uint accepted[SAMPLE_RING];  // 1 when ids[s] is the draft
};

// (v, i) is better than (w, j) when v > w, or v == w and i < j.
inline bool better(float v, uint i, float w, uint j) {
    return v > w || (v == w && i < j);
}

// The best (v, i) over the simdgroup.
inline void simd_best(thread float &v, thread uint &i) {
    float m = simd_max(v);
    uint id = simd_min(v == m ? i : 0xffffffffu);
    v = m;
    i = id;
}

kernel void sample_partial(device const float *logits [[buffer(0)]],
                           device float *cand_v [[buffer(1)]],
                           device uint *cand_i [[buffer(2)]],
                           device float *part [[buffer(3)]],
                           constant SampleParams &p [[buffer(4)]],
                           uint g [[threadgroup_position_in_grid]],
                           uint t [[thread_position_in_threadgroup]],
                           uint simd [[simdgroup_index_in_threadgroup]],
                           uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float sv[SAMPLE_SIMDS];
    threadgroup uint si[SAMPLE_SIMDS];
    threadgroup uint win;
    float v[SAMPLE_PER_THREAD_P];
    uint id[SAMPLE_PER_THREAD_P];
    uint base = g * SAMPLE_SLICE + t * SAMPLE_PER_THREAD_P;
    float m = -INFINITY;
    for (uint j = 0; j < SAMPLE_PER_THREAD_P; j++) {
        uint i = base + j;
        v[j] = i < p.n ? logits[i] : -INFINITY;
        id[j] = i < p.n ? i : 0xffffffffu;
        m = max(m, v[j]);
    }
    float mm = simd_max(m);
    if (lane == 0) {
        sv[simd] = mm;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float gm = sv[0];
    for (uint s = 1; s < SAMPLE_SIMDS; s++) {
        gm = max(gm, sv[s]);
    }
    float e = 0;
    for (uint j = 0; j < SAMPLE_PER_THREAD_P; j++) {
        e += id[j] != 0xffffffffu ? precise::exp(v[j] - gm) : 0;
    }
    e = simd_sum(e);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lane == 0) {
        sv[simd] = e;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (t == 0) {
        float total = 0;
        for (uint s = 0; s < SAMPLE_SIMDS; s++) {
            total += sv[s];
        }
        part[2 * g] = gm;
        part[2 * g + 1] = total;
    }

    for (uint r = 0; r < p.top_k; r++) {
        float bv = v[0];
        uint bi = id[0];
        for (uint j = 1; j < SAMPLE_PER_THREAD_P; j++) {
            if (better(v[j], id[j], bv, bi)) {
                bv = v[j];
                bi = id[j];
            }
        }
        simd_best(bv, bi);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (lane == 0) {
            sv[simd] = bv;
            si[simd] = bi;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (simd == 0) {
            float x = lane < SAMPLE_SIMDS ? sv[lane] : -INFINITY;
            uint xi = lane < SAMPLE_SIMDS ? si[lane] : 0xffffffffu;
            simd_best(x, xi);
            if (lane == 0) {
                win = xi;
                cand_v[g * SAMPLE_MAX_K + r] = x;
                cand_i[g * SAMPLE_MAX_K + r] = xi;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        uint w = win;
        for (uint j = 0; j < SAMPLE_PER_THREAD_P; j++) {
            if (id[j] == w) {
                v[j] = -INFINITY;
                id[j] = 0xffffffffu;
            }
        }
    }
}

// splitmix64, top 24 bits as a float in [0, 1), as Sampler::next_f32.
inline float next_f32(device ulong *state) {
    ulong z = *state + 0x9e3779b97f4a7c15ul;
    *state = z;
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ul;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebul;
    z ^= z >> 31;
    return float(z >> 40) / float(1u << 24);
}

kernel void sample_final(device const float *cand_v [[buffer(0)]],
                         device const uint *cand_i [[buffer(1)]],
                         device const float *part [[buffer(2)]],
                         constant SampleParams &p [[buffer(3)]],
                         device ulong *rng [[buffer(4)]],
                         device SampleRing *ring [[buffer(5)]],
                         device TokenParams *tp [[buffer(6)]],
                         uint t [[thread_position_in_threadgroup]],
                         uint simd [[simdgroup_index_in_threadgroup]],
                         uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float sv[32];
    threadgroup uint si[32];
    threadgroup float top_v[SAMPLE_MAX_K];
    threadgroup uint top_i[SAMPLE_MAX_K];
    threadgroup float keep_v[SAMPLE_KEEP];
    threadgroup uint keep_i[SAMPLE_KEEP];
    threadgroup atomic_uint kept;
    threadgroup float total_tg;
    threadgroup float gmax[SAMPLE_GROUPS];
    threadgroup float bound_tg;
    uint k = p.top_k;

    // The whole-vocab softmax total for top_p, relative to the best logit.
    {
        float m = t < SAMPLE_GROUPS ? part[2 * t] : -INFINITY;
        float s = t < SAMPLE_GROUPS ? part[2 * t + 1] : 0;
        float mm = simd_max(m);
        if (lane == 0) {
            sv[simd] = mm;
        }
        // The global k-th best is at least each group's k-th best.
        float kth = t < SAMPLE_GROUPS ? cand_v[t * SAMPLE_MAX_K + k - 1] : -INFINITY;
        kth = simd_max(kth);
        if (lane == 0) {
            top_v[simd] = kth;
        }
        if (t == 0) {
            atomic_store_explicit(&kept, 0, memory_order_relaxed);
        }
        if (t < SAMPLE_GROUPS) {
            gmax[t] = m;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float best = simd_max(sv[lane]);
        float bound = simd_max(top_v[lane]);
        float e = s > 0 ? s * precise::exp(m - best) : 0;
        e = simd_sum(e);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (lane == 0) {
            sv[simd] = e;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (simd == 0) {
            float total = simd_sum(sv[lane]);
            if (lane == 0) {
                total_tg = total;
            }
        }
        // So does the k-th best group max.
        if (t < SAMPLE_GROUPS) {
            uint rank = 0;
            for (uint o = 0; o < SAMPLE_GROUPS; o++) {
                rank += better(gmax[o], o, m, t) ? 1 : 0;
            }
            if (rank == k - 1) {
                bound_tg = max(bound, m);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        bound = bound_tg;

        // Keep the candidates at or above the bound; the global top_k is
        // among them.
        for (uint c = t; c < SAMPLE_GROUPS * k; c += 1024) {
            uint at = (c / k) * SAMPLE_MAX_K + c % k;
            float v = cand_v[at];
            if (v >= bound) {
                uint n = atomic_fetch_add_explicit(&kept, 1, memory_order_relaxed);
                if (n < SAMPLE_KEEP) {
                    keep_v[n] = v;
                    keep_i[n] = cand_i[at];
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    uint n = atomic_load_explicit(&kept, memory_order_relaxed);
    if (n <= SAMPLE_KEEP) {
        // Rank each kept candidate; the order is total, so ranks are unique.
        for (uint c = t; c < n; c += 1024) {
            float v = keep_v[c];
            uint i = keep_i[c];
            uint rank = 0;
            for (uint o = 0; o < n && rank < k; o++) {
                rank += better(keep_v[o], keep_i[o], v, i) ? 1 : 0;
            }
            if (rank < k) {
                top_v[rank] = v;
                top_i[rank] = i;
            }
        }
    } else {
        // Many ties at the bound: merge round by round.
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float v[SAMPLE_PER_THREAD];
        uint id[SAMPLE_PER_THREAD];
        for (uint j = 0; j < SAMPLE_PER_THREAD; j++) {
            uint c = t + j * 1024;
            bool ok = c < SAMPLE_GROUPS * k;
            uint at = ok ? (c / k) * SAMPLE_MAX_K + c % k : 0;
            v[j] = ok ? cand_v[at] : -INFINITY;
            id[j] = ok ? cand_i[at] : 0xffffffffu;
        }
        for (uint r = 0; r < k; r++) {
            float bv = -INFINITY;
            uint bi = 0xffffffffu;
            for (uint j = 0; j < SAMPLE_PER_THREAD; j++) {
                if (better(v[j], id[j], bv, bi)) {
                    bv = v[j];
                    bi = id[j];
                }
            }
            float wv = bv;
            uint wi = bi;
            simd_best(wv, wi);
            if (lane == 0) {
                sv[simd] = wv;
                si[simd] = wi;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (simd == 0) {
                float x = sv[lane];
                uint xi = si[lane];
                simd_best(x, xi);
                if (lane == 0) {
                    top_v[r] = x;
                    top_i[r] = xi;
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            uint w = top_i[r];
            for (uint j = 0; j < SAMPLE_PER_THREAD; j++) {
                if (id[j] == w && w != 0xffffffffu) {
                    v[j] = -INFINITY;
                    id[j] = 0xffffffffu;
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (t != 0) {
        return;
    }
    uint slot = (tp->position + 1) & (SAMPLE_RING - 1);
    uint draft = ring->draft[slot];
    uint chosen = top_i[0];
    float prob = 1;
    if (p.temperature > 0) {
        float best = top_v[0];
        uint keep = p.top_k;
        if (p.top_p < 1.0f) {
            float total = total_tg;
            float mass = 0;
            keep = 0;
            while (keep < p.top_k && mass < p.top_p) {
                mass += precise::exp(top_v[keep] - best) / total;
                keep++;
            }
            keep = max(keep, 1u);
        }
        float sum = 0;
        float w_draft = 0;
        for (uint i = 0; i < keep; i++) {
            float w = precise::exp((top_v[i] - best) / p.temperature);
            sum += w;
            if (top_i[i] == draft) {
                w_draft = w;
            }
        }
        if (draft != 0xffffffffu && next_f32(rng) * sum < w_draft) {
            chosen = draft;
            prob = w_draft / sum;
        } else {
            // A draft that missed takes no part in the draw.
            float rest = sum - w_draft;
            float x = next_f32(rng) * rest;
            uint at = keep;
            for (uint i = 0; i < keep; i++) {
                if (top_i[i] == draft) {
                    continue;
                }
                at = i;
                x -= precise::exp((top_v[i] - best) / p.temperature);
                if (x < 0) {
                    break;
                }
            }
            chosen = top_i[at];
            prob = precise::exp((top_v[at] - best) / p.temperature) / sum;
        }
    }
    ring->ids[slot] = chosen;
    ring->probs[slot] = prob;
    ring->accepted[slot] = draft != 0xffffffffu && chosen == draft ? 1 : 0;
    if (p.advance != 0) {
        tp->token = chosen;
        tp->position += 1;
        tp->kv_len = tp->position + 1;
    }
}

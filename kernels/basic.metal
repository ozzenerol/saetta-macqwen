#include <metal_stdlib>
using namespace metal;

kernel void add_one(device float *data [[buffer(0)]],
                    uint i [[thread_position_in_grid]]) {
    data[i] += 1.0f;
}

kernel void copy_u32(device const uint *src [[buffer(0)]],
                     device uint *dst [[buffer(1)]],
                     uint i [[thread_position_in_grid]]) {
    dst[i] = src[i];
}

// For tests of plan.rs. Uses TokenParams from params.metal.
kernel void add_position(device float *data [[buffer(0)]],
                         device const TokenParams &p [[buffer(1)]],
                         uint i [[thread_position_in_grid]]) {
    data[i] += float(p.position);
}

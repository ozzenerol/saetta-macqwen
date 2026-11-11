//! How much does one dependent dispatch cost? A chain of N small kernels,
//! each reading the previous one's output, encoded six ways.

// The objc 0.2 msg_send! macro checks an old `cargo-clippy` cfg.
#![allow(unexpected_cfgs)]

#[cfg(target_os = "macos")]
fn main() {
    use metal::objc::{msg_send, sel, sel_impl};
    use metal::{
        CompileOptions, ComputePipelineDescriptor, IndirectCommandBufferDescriptor,
        MTLDispatchType, MTLIndirectCommandType, MTLResourceOptions, MTLResourceUsage, MTLSize,
        NSRange,
    };
    use saetta::gpu::Gpu;
    use std::time::Instant;

    const N: usize = 500;
    const WIDTH: u64 = 2048;
    let src = "#include <metal_stdlib>\nusing namespace metal;\n\
        kernel void step(device const float *x [[buffer(0)]], device float *y [[buffer(1)]],\n\
                         uint i [[thread_position_in_grid]]) { y[i] = x[i] + 1.0f; }\n";

    let gpu = Gpu::new().unwrap();
    let dev = &gpu.device;
    let lib = dev
        .new_library_with_source(src, &CompileOptions::new())
        .unwrap();
    let desc = ComputePipelineDescriptor::new();
    desc.set_compute_function(Some(&lib.get_function("step", None).unwrap()));
    desc.set_support_indirect_command_buffers(true);
    let pipe = dev.new_compute_pipeline_state(&desc).unwrap();
    let queue = dev.new_command_queue();

    let tracked = MTLResourceOptions::StorageModeShared;
    let untracked = tracked | MTLResourceOptions::HazardTrackingModeUntracked;
    let pair = |opt| {
        [
            dev.new_buffer(WIDTH * 4, opt),
            dev.new_buffer(WIDTH * 4, opt),
        ]
    };
    let bt = pair(tracked);
    let bu = pair(untracked);
    let grid = MTLSize::new(WIDTH / 256, 1, 1);
    let tg = MTLSize::new(256, 1, 1);

    // The ICB is encoded once: N chained commands, each with a barrier.
    let icb = |barriers: bool| {
        let d = IndirectCommandBufferDescriptor::new();
        // MTLIndirectCommandTypeConcurrentDispatch is 1 << 5 in the Metal
        // headers; the metal crate has it as 1 << 4, which crashes.
        d.set_command_types(MTLIndirectCommandType::from_bits_retain(1 << 5));
        d.set_inherit_buffers(false);
        d.set_inherit_pipeline_state(false);
        d.set_max_kernel_buffer_bind_count(2);
        let icb = dev.new_indirect_command_buffer_with_descriptor(
            &d,
            N as u64,
            MTLResourceOptions::empty(),
        );
        for i in 0..N {
            let c = icb.indirect_compute_command_at_index(i as u64);
            c.set_compute_pipeline_state(&pipe);
            c.set_kernel_buffer(0, Some(&bu[i % 2]), 0);
            c.set_kernel_buffer(1, Some(&bu[(i + 1) % 2]), 0);
            if barriers {
                c.set_barrier();
            }
            c.concurrent_dispatch_threadgroups(grid, tg);
        }
        icb
    };
    let icb_dep = icb(true);
    let icb_free = icb(false);

    // Returns (wall seconds, GPU seconds).
    let time = |mode: usize| -> (f64, f64) {
        let start = Instant::now();
        let cb = queue.new_command_buffer();
        let ty = if mode == 0 || mode == 1 {
            MTLDispatchType::Serial
        } else {
            MTLDispatchType::Concurrent
        };
        let enc = cb.compute_command_encoder_with_dispatch_type(ty);
        match mode {
            0..=3 => {
                let b = if mode == 0 { &bt } else { &bu };
                enc.set_compute_pipeline_state(&pipe);
                for i in 0..N {
                    enc.set_buffer(0, Some(&b[i % 2]), 0);
                    enc.set_buffer(1, Some(&b[(i + 1) % 2]), 0);
                    enc.dispatch_thread_groups(grid, tg);
                    if mode == 2 {
                        enc.memory_barrier_with_resources(&[&b[(i + 1) % 2]]);
                    }
                }
            }
            _ => {
                enc.use_resource(&bu[0], MTLResourceUsage::Read | MTLResourceUsage::Write);
                enc.use_resource(&bu[1], MTLResourceUsage::Read | MTLResourceUsage::Write);
                let icb = if mode == 4 { &icb_dep } else { &icb_free };
                // The metal crate binds this only on the render encoder.
                let range = NSRange::new(0, N as u64);
                let icb: &metal::IndirectCommandBufferRef = icb;
                unsafe {
                    let _: () = msg_send![enc, executeCommandsInBuffer: icb withRange: range];
                }
            }
        }
        enc.end_encoding();
        cb.commit();
        cb.wait_until_completed();
        let wall = start.elapsed().as_secs_f64();
        let (t0, t1): (f64, f64) =
            unsafe { (msg_send![cb, GPUStartTime], msg_send![cb, GPUEndTime]) };
        (wall, t1 - t0)
    };
    // Each step adds 1, so a chain that keeps its order ends at N.
    let order_ok = |mode: usize| {
        for b in bt.iter().chain(&bu) {
            unsafe { std::ptr::write_bytes(b.contents() as *mut u8, 0, (WIDTH * 4) as usize) };
        }
        time(mode);
        let b = if mode == 0 { &bt } else { &bu };
        let out = unsafe {
            std::slice::from_raw_parts(b[N % 2].contents() as *const f32, WIDTH as usize)
        };
        out.iter().all(|&v| v == N as f32)
    };
    let empty = || {
        let start = Instant::now();
        let cb = queue.new_command_buffer();
        cb.new_compute_command_encoder().end_encoding();
        cb.commit();
        cb.wait_until_completed();
        start.elapsed().as_secs_f64()
    };

    let names = [
        "1  serial encoder, tracked buffers (today)",
        "1b serial encoder, untracked buffers",
        "2  concurrent encoder, barrier on the written buffer",
        "3  indirect command buffer, barrier per command",
        "2b concurrent encoder, no barriers (not a chain, lower bound)",
        "3b indirect command buffer, no barriers (not a chain)",
    ];
    let modes = [0, 1, 2, 4, 3, 5];
    let mut e: Vec<f64> = (0..20).map(|_| empty()).collect();
    e.sort_by(f64::total_cmp);
    let base = e[10];
    println!(
        "{N} dispatches of {WIDTH} threads per command buffer, empty command buffer {:.0} us wall.",
        base * 1e6
    );
    println!(
        "Per dispatch, median of 5 runs after 1 warm-up, 3 rounds: wall minus the empty buffer, and GPU time:"
    );
    for (name, &m) in names.iter().zip(&modes) {
        let mut line = String::new();
        for _ in 0..3 {
            let mut t: Vec<(f64, f64)> = (0..6).map(|_| time(m)).skip(1).collect();
            t.sort_by(|a, b| a.0.total_cmp(&b.0));
            let wall = (t[2].0 - base) / N as f64 * 1e6;
            t.sort_by(|a, b| a.1.total_cmp(&b.1));
            let gpu = t[2].1 / N as f64 * 1e6;
            line += &format!("  wall {wall:.2} gpu {gpu:.2}");
        }
        println!("{name}:{line} us, order kept: {}", order_ok(m));
    }
}

#[cfg(not(target_os = "macos"))]
fn main() {}

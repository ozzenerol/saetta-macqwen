//! Metal device, kernel library and buffers.
//!
//! Kernels live in `kernels/*.metal` and are compiled at startup from source,
//! so the Mac needs no Xcode, only the system Metal framework.

use std::process::ExitCode;

pub const KERNELS: &str = include_str!("../kernels/basic.metal");

#[cfg(target_os = "macos")]
pub fn check() -> ExitCode {
    use metal::{Device, MTLResourceOptions, MTLSize};

    let device = Device::system_default().expect("no Metal device");
    let library = device
        .new_library_with_source(KERNELS, &metal::CompileOptions::new())
        .expect("kernel compile failed");
    let function = library.get_function("add_one", None).unwrap();
    let pipeline = device
        .new_compute_pipeline_state_with_function(&function)
        .unwrap();

    let n = 1024usize;
    let input: Vec<f32> = (0..n).map(|i| i as f32).collect();
    let buffer = device.new_buffer_with_data(
        input.as_ptr().cast(),
        (n * 4) as u64,
        MTLResourceOptions::StorageModeShared,
    );

    let queue = device.new_command_queue();
    let commands = queue.new_command_buffer();
    let encoder = commands.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&pipeline);
    encoder.set_buffer(0, Some(&buffer), 0);
    encoder.dispatch_threads(MTLSize::new(n as u64, 1, 1), MTLSize::new(256, 1, 1));
    encoder.end_encoding();
    commands.commit();
    commands.wait_until_completed();

    let output = unsafe { std::slice::from_raw_parts(buffer.contents() as *const f32, n) };
    let ok = output.iter().enumerate().all(|(i, &v)| v == i as f32 + 1.0);
    println!("{}: {}", device.name(), if ok { "ok" } else { "WRONG" });
    if ok {
        ExitCode::SUCCESS
    } else {
        ExitCode::FAILURE
    }
}

#[cfg(not(target_os = "macos"))]
pub fn check() -> ExitCode {
    eprintln!("gpu-check needs macOS; run it with scripts/mac.sh");
    ExitCode::FAILURE
}

//! Metal device, kernel library and buffers.
//!
//! Kernels live in `kernels/*.metal` and are compiled at startup from source,
//! so the Mac needs no Xcode, only the system Metal framework.

pub const KERNELS: &str = concat!(
    include_str!("../kernels/basic.metal"),
    include_str!("../kernels/qmv.metal"),
);

#[cfg(target_os = "macos")]
pub use mac::*;

#[cfg(not(target_os = "macos"))]
pub fn check() -> std::process::ExitCode {
    eprintln!("gpu-check needs macOS; run it with scripts/mac.sh");
    std::process::ExitCode::FAILURE
}

#[cfg(target_os = "macos")]
mod mac {
    use super::KERNELS;
    use memmap2::Mmap;
    use metal::{
        Buffer, CommandQueue, CompileOptions, ComputeCommandEncoderRef, ComputePipelineState,
        Device, Library, MTLResourceOptions, MTLSize,
    };
    use std::cell::RefCell;
    use std::collections::HashMap;
    use std::fs::File;
    use std::path::Path;
    use std::process::ExitCode;
    use std::time::{Duration, Instant};

    /// Apple Silicon page size. No-copy buffers must start and end on it.
    pub const PAGE: usize = 16384;

    pub struct Gpu {
        pub device: Device,
        queue: CommandQueue,
        library: Library,
        pipelines: RefCell<HashMap<String, ComputePipelineState>>,
    }

    /// A file mapped into memory and wrapped as a GPU buffer without a copy.
    /// Pages are read from disk only when the CPU or GPU touches them.
    pub struct MappedFile {
        map: Mmap,
        pub buffer: Buffer,
    }

    impl MappedFile {
        pub fn bytes(&self) -> &[u8] {
            &self.map
        }
    }

    impl Gpu {
        pub fn new() -> Result<Gpu, String> {
            let device = Device::system_default().ok_or("no Metal device")?;
            let library = device
                .new_library_with_source(KERNELS, &CompileOptions::new())
                .map_err(|e| format!("kernel compile failed: {e}"))?;
            let queue = device.new_command_queue();
            Ok(Gpu {
                device,
                queue,
                library,
                pipelines: RefCell::new(HashMap::new()),
            })
        }

        /// The compute pipeline for the kernel function `name`, built once.
        pub fn pipeline(&self, name: &str) -> ComputePipelineState {
            if let Some(p) = self.pipelines.borrow().get(name) {
                return p.clone();
            }
            let function = self
                .library
                .get_function(name, None)
                .unwrap_or_else(|e| panic!("kernel {name}: {e}"));
            let pipeline = self
                .device
                .new_compute_pipeline_state_with_function(&function)
                .unwrap_or_else(|e| panic!("pipeline {name}: {e}"));
            self.pipelines
                .borrow_mut()
                .insert(name.to_string(), pipeline.clone());
            pipeline
        }

        /// A zeroed buffer of `len` bytes in shared memory.
        pub fn buffer(&self, len: usize) -> Buffer {
            self.device
                .new_buffer(len as u64, MTLResourceOptions::StorageModeShared)
        }

        /// A buffer holding a copy of `data`.
        pub fn buffer_from<T: Copy>(&self, data: &[T]) -> Buffer {
            self.device.new_buffer_with_data(
                data.as_ptr().cast(),
                std::mem::size_of_val(data) as u64,
                MTLResourceOptions::StorageModeShared,
            )
        }

        /// Wraps any byte range of a memory-mapped file as a buffer, without
        /// a copy. The buffer starts at the page that holds `bytes[0]`; the
        /// returned offset is where `bytes` starts inside it.
        ///
        /// # Safety
        /// `bytes` must lie in an mmap that outlives the buffer, so the whole
        /// pages around it are mapped.
        pub unsafe fn buffer_no_copy(&self, bytes: &[u8]) -> (Buffer, u64) {
            let start = bytes.as_ptr() as usize;
            let page_start = start / PAGE * PAGE;
            let page_end = (start + bytes.len()).div_ceil(PAGE) * PAGE;
            let buffer = self.device.new_buffer_with_bytes_no_copy(
                page_start as *const std::ffi::c_void,
                (page_end - page_start) as u64,
                MTLResourceOptions::StorageModeShared,
                None,
            );
            (buffer, (start - page_start) as u64)
        }

        /// Maps a file read-only and wraps it as a buffer. Kernels must not
        /// write to it.
        pub fn map_file(&self, path: &Path) -> std::io::Result<MappedFile> {
            let file = File::open(path)?;
            let map = unsafe { Mmap::map(&file)? };
            let (buffer, _) = unsafe { self.buffer_no_copy(&map) };
            Ok(MappedFile { map, buffer })
        }

        /// Encodes work with `encode` into one command buffer, runs it and
        /// waits. Returns the wall time from commit to completion.
        pub fn run(&self, encode: impl FnOnce(&ComputeCommandEncoderRef)) -> Duration {
            let commands = self.queue.new_command_buffer();
            let encoder = commands.new_compute_command_encoder();
            encode(encoder);
            encoder.end_encoding();
            let start = Instant::now();
            commands.commit();
            commands.wait_until_completed();
            start.elapsed()
        }
    }

    /// Dispatches `pipeline` over `n` threads in a 1D grid.
    pub fn dispatch_1d(
        encoder: &ComputeCommandEncoderRef,
        pipeline: &ComputePipelineState,
        buffers: &[(&Buffer, u64)],
        n: usize,
    ) {
        encoder.set_compute_pipeline_state(pipeline);
        for (i, (buffer, offset)) in buffers.iter().enumerate() {
            encoder.set_buffer(i as u64, Some(buffer), *offset);
        }
        let width = pipeline.max_total_threads_per_threadgroup().min(256);
        encoder.dispatch_threads(MTLSize::new(n as u64, 1, 1), MTLSize::new(width, 1, 1));
    }

    /// Views the first `n` values of a shared buffer. Do not hold the slice
    /// while the GPU writes to the buffer.
    pub fn contents<T: Copy>(buffer: &Buffer, n: usize) -> &[T] {
        assert!(n * size_of::<T>() <= buffer.length() as usize);
        unsafe { std::slice::from_raw_parts(buffer.contents() as *const T, n) }
    }

    pub fn check() -> ExitCode {
        let gpu = match Gpu::new() {
            Ok(g) => g,
            Err(e) => {
                eprintln!("{e}");
                return ExitCode::FAILURE;
            }
        };
        let n = 1024;
        let input: Vec<f32> = (0..n).map(|i| i as f32).collect();
        let buffer = gpu.buffer_from(&input);
        let pipeline = gpu.pipeline("add_one");
        gpu.run(|enc| dispatch_1d(enc, &pipeline, &[(&buffer, 0)], n));

        let ok = contents::<f32>(&buffer, n)
            .iter()
            .enumerate()
            .all(|(i, &v)| v == i as f32 + 1.0);
        println!(
            "{}: {}, max buffer {} GB, working set {} GB",
            gpu.device.name(),
            if ok { "ok" } else { "WRONG" },
            gpu.device.max_buffer_length() >> 30,
            gpu.device.recommended_max_working_set_size() >> 30,
        );
        if ok {
            ExitCode::SUCCESS
        } else {
            ExitCode::FAILURE
        }
    }
}

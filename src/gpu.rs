//! Metal device, kernel library and buffers.
//!
//! Kernels live in `kernels/*.metal` and are compiled at startup from source,
//! so the Mac needs no Xcode, only the system Metal framework.

pub const KERNELS: &str = concat!(
    include_str!("../kernels/params.metal"),
    include_str!("../kernels/basic.metal"),
    include_str!("../kernels/quant.metal"),
    include_str!("../kernels/qmv.metal"),
    include_str!("../kernels/moe.metal"),
    include_str!("../kernels/router.metal"),
    include_str!("../kernels/rows.metal"),
    include_str!("../kernels/layers.metal"),
    include_str!("../kernels/attention.metal"),
    include_str!("../kernels/kv_quant.metal"),
    include_str!("../kernels/attention_rows.metal"),
    include_str!("../kernels/model.metal"),
    include_str!("../kernels/prefill.metal"),
    include_str!("../kernels/attention_rows_q.metal"),
    include_str!("../kernels/sample.metal"),
    include_str!("../kernels/qmm.metal"),
    include_str!("../kernels/attention_prefill.metal"),
    include_str!("../kernels/attention_gemm.metal"),
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
        BlitCommandEncoderRef, Buffer, CommandBuffer, CommandQueue, CompileOptions,
        ComputeCommandEncoderRef, ComputePipelineState, Device, Library, MTLDispatchType,
        MTLResourceOptions, MTLSize,
    };
    use std::cell::RefCell;
    use std::collections::HashMap;
    use std::fs::File;
    use std::path::Path;
    use std::process::ExitCode;
    use std::time::{Duration, Instant};

    /// All buffers skip Metal's hazard tracking. A serial encoder runs its
    /// dispatches in order anyway, and a Plan replayed from an indirect
    /// command buffer has explicit barriers. Tracking only adds cost.
    const OPTIONS: MTLResourceOptions = MTLResourceOptions::StorageModeShared
        .union(MTLResourceOptions::HazardTrackingModeUntracked);

    /// Apple Silicon page size. No-copy buffers must start and end on it.
    pub const PAGE: usize = 16384;

    pub struct Gpu {
        pub device: Device,
        queue: CommandQueue,
        library: Library,
        pipelines: RefCell<HashMap<String, ComputePipelineState>>,
        /// The buffers `buffer_no_copy` made, by page range. The decode
        /// model and the prefill wrap the same weights; with one buffer per
        /// range, switching between them does not make Metal change which
        /// 20 GB is resident, which took about 250 ms each way.
        no_copy: RefCell<HashMap<(usize, usize), Buffer>>,
        /// Keeps the weights resident between requests (see `Residency`).
        residency: Option<Residency>,
    }

    /// An MTLResidencySet (macOS 15) with every no-copy weight buffer,
    /// attached to the queue. Without it the driver drops the residency of
    /// the 20 GB of weights a few seconds after the GPU goes idle, and the
    /// next prefill pays 10 to 17 ms a layer to make them resident again
    /// (about 0.55 s for a qh turn). `SAETTA_RESIDENCY=0` turns it off.
    struct Residency {
        set: *mut metal::objc::runtime::Object,
        /// Allocations were added since the last commit.
        dirty: std::cell::Cell<bool>,
    }

    // The objc 0.2 msg_send! macro checks an old `cargo-clippy` cfg.
    #[allow(unexpected_cfgs)]
    impl Residency {
        fn new(device: &Device, queue: &CommandQueue) -> Option<Residency> {
            use metal::foreign_types::ForeignType;
            use metal::objc::runtime::{BOOL, Class, NO, Object};
            use metal::objc::{class, msg_send, sel, sel_impl};
            if std::env::var("SAETTA_RESIDENCY").is_ok_and(|v| v == "0") {
                return None;
            }
            unsafe {
                let device = device.as_ptr() as *mut Object;
                let queue = queue.as_ptr() as *mut Object;
                let ok: BOOL = msg_send![device, respondsToSelector: sel!(newResidencySetWithDescriptor:error:)];
                if ok == NO || Class::get("MTLResidencySetDescriptor").is_none() {
                    eprintln!(
                        "no residency sets before macOS 15; weights may go cold between requests"
                    );
                    return None;
                }
                let desc: *mut Object = msg_send![class!(MTLResidencySetDescriptor), new];
                let mut error: *mut Object = std::ptr::null_mut();
                let set: *mut Object =
                    msg_send![device, newResidencySetWithDescriptor: desc error: &mut error];
                let _: () = msg_send![desc, release];
                if set.is_null() {
                    eprintln!("could not make a residency set");
                    return None;
                }
                let _: () = msg_send![queue, addResidencySet: set];
                Some(Residency {
                    set,
                    dirty: std::cell::Cell::new(false),
                })
            }
        }

        fn add(&self, buffer: &Buffer) {
            use metal::foreign_types::ForeignType;
            use metal::objc::runtime::Object;
            use metal::objc::{msg_send, sel, sel_impl};
            unsafe {
                let _: () = msg_send![self.set, addAllocation: buffer.as_ptr() as *mut Object];
            }
            self.dirty.set(true);
        }

        /// Commits added allocations and asks for them to stay resident.
        fn commit(&self) {
            use metal::objc::{msg_send, sel, sel_impl};
            if !self.dirty.replace(false) {
                return;
            }
            unsafe {
                let _: () = msg_send![self.set, commit];
                let _: () = msg_send![self.set, requestResidency];
            }
        }
    }

    // The objc 0.2 msg_send! macro checks an old `cargo-clippy` cfg.
    #[allow(unexpected_cfgs)]
    impl Drop for Residency {
        fn drop(&mut self) {
            use metal::objc::{msg_send, sel, sel_impl};
            unsafe {
                let _: () = msg_send![self.set, endResidency];
                let _: () = msg_send![self.set, release];
            }
        }
    }

    /// Keeps the GPU warm between requests. A few seconds after the GPU goes
    /// idle, the first dispatch of the next request waits about 270 ms, even
    /// a one-thread kernel (batch 55), which is most of the time to first
    /// token of a short prompt. A thread on its own queue runs a one-thread
    /// kernel every `every` until `idle` after the last `touch`.
    pub struct Warm {
        last: std::sync::Arc<std::sync::atomic::AtomicU64>,
        start: Instant,
    }

    impl Warm {
        /// Marks GPU work now.
        pub fn touch(&self) {
            let ms = self.start.elapsed().as_millis() as u64;
            self.last.store(ms, std::sync::atomic::Ordering::Relaxed);
        }
    }

    impl Gpu {
        // The objc 0.2 msg_send! macro checks an old `cargo-clippy` cfg.
        #[allow(unexpected_cfgs)]
        pub fn keep_warm(&self, idle: Duration, every: Duration) -> Warm {
            use metal::foreign_types::ForeignType;
            use metal::objc::runtime::Object;
            use metal::objc::{msg_send, sel, sel_impl};
            use std::sync::atomic::Ordering;
            let queue = self.device.new_command_queue();
            if let Some(r) = &self.residency {
                unsafe {
                    let _: () = msg_send![queue.as_ptr() as *mut Object, addResidencySet: r.set];
                }
            }
            let pipeline = self.pipeline("add_in_place");
            let buffer = self.buffer(4);
            let start = Instant::now();
            let last = std::sync::Arc::new(std::sync::atomic::AtomicU64::new(0));
            let seen = std::sync::Arc::clone(&last);
            std::thread::spawn(move || {
                loop {
                    std::thread::sleep(every);
                    let since = start.elapsed().as_millis() as u64 - seen.load(Ordering::Relaxed);
                    if since > idle.as_millis() as u64 {
                        continue;
                    }
                    metal::objc::rc::autoreleasepool(|| {
                        let commands = queue.new_command_buffer();
                        let enc = commands.new_compute_command_encoder();
                        dispatch_1d(enc, &pipeline, &[(&buffer, 0), (&buffer, 0)], 1);
                        enc.end_encoding();
                        commands.commit();
                        commands.wait_until_completed();
                    });
                }
            });
            Warm { last, start }
        }
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
            let residency = Residency::new(&device, &queue);
            Ok(Gpu {
                device,
                queue,
                library,
                pipelines: RefCell::new(HashMap::new()),
                no_copy: RefCell::new(HashMap::new()),
                residency,
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
            let desc = metal::ComputePipelineDescriptor::new();
            desc.set_compute_function(Some(&function));
            desc.set_support_indirect_command_buffers(true);
            let pipeline = self
                .device
                .new_compute_pipeline_state(&desc)
                .unwrap_or_else(|e| panic!("pipeline {name}: {e}"));
            self.pipelines
                .borrow_mut()
                .insert(name.to_string(), pipeline.clone());
            pipeline
        }

        /// A zeroed buffer of `len` bytes in shared memory.
        pub fn buffer(&self, len: usize) -> Buffer {
            self.device.new_buffer(len as u64, OPTIONS)
        }

        /// Buffers that share memory: group `g` gets one buffer per length
        /// in `groups[g]`, laid out one after another from the start of one
        /// placement heap, so buffer `i` of each group overlaps buffer `i` of
        /// the others when the lengths match. The heap is as big as the
        /// largest group. Only for scratch whose groups are never live at
        /// the same time; the contents are not zeroed. None if the device
        /// cannot make the heap or `SAETTA_ALIAS=0`.
        pub fn aliased(&self, groups: &[&[usize]]) -> Option<Vec<Vec<Buffer>>> {
            if std::env::var("SAETTA_ALIAS").is_ok_and(|v| v == "0") {
                return None;
            }
            let layout = |lengths: &[usize]| -> Vec<(u64, u64)> {
                let mut at = 0u64;
                lengths
                    .iter()
                    .map(|&n| {
                        let n = n.max(4) as u64;
                        let sa = self.device.heap_buffer_size_and_align(n, OPTIONS);
                        at = at.next_multiple_of(sa.align);
                        let place = (at, n);
                        at += sa.size;
                        place
                    })
                    .collect()
            };
            let layouts: Vec<_> = groups.iter().map(|g| layout(g)).collect();
            let size = layouts
                .iter()
                .filter_map(|l| l.last().map(|&(at, n)| at + n))
                .max()?
                .next_multiple_of(PAGE as u64);
            let desc = metal::HeapDescriptor::new();
            desc.set_heap_type(metal::MTLHeapType::Placement);
            desc.set_storage_mode(metal::MTLStorageMode::Shared);
            desc.set_hazard_tracking_mode(metal::MTLHazardTrackingMode::Untracked);
            desc.set_size(size);
            let heap = self.device.new_heap(&desc);
            if metal::foreign_types::ForeignType::as_ptr(&heap).is_null() {
                return None;
            }
            layouts
                .iter()
                .map(|l| {
                    l.iter()
                        .map(|&(at, n)| heap.new_buffer_with_offset(n, OPTIONS, at))
                        .collect::<Option<Vec<_>>>()
                })
                .collect()
        }

        /// A buffer holding a copy of `data`.
        pub fn buffer_from<T: Copy>(&self, data: &[T]) -> Buffer {
            self.device.new_buffer_with_data(
                data.as_ptr().cast(),
                std::mem::size_of_val(data) as u64,
                OPTIONS,
            )
        }

        /// Wraps any byte range of a memory-mapped file as a buffer, without
        /// a copy. The buffer starts at the page that holds `bytes[0]`; the
        /// returned offset is where `bytes` starts inside it.
        ///
        /// The same page range gives the same buffer, which the `Gpu` keeps.
        ///
        /// # Safety
        /// `bytes` must lie in an mmap that outlives the `Gpu`, so the whole
        /// pages around it stay mapped and a later mmap cannot reuse the
        /// range under a kept buffer.
        pub unsafe fn buffer_no_copy(&self, bytes: &[u8]) -> (Buffer, u64) {
            let start = bytes.as_ptr() as usize;
            let page_start = start / PAGE * PAGE;
            let page_end = (start + bytes.len()).div_ceil(PAGE) * PAGE;
            let buffer = self
                .no_copy
                .borrow_mut()
                .entry((page_start, page_end))
                .or_insert_with(|| {
                    let buffer = self.device.new_buffer_with_bytes_no_copy(
                        page_start as *const std::ffi::c_void,
                        (page_end - page_start) as u64,
                        OPTIONS,
                        None,
                    );
                    if let Some(r) = &self.residency {
                        r.add(&buffer);
                    }
                    buffer
                })
                .clone();
            (buffer, (start - page_start) as u64)
        }

        /// Bytes of all the no-copy buffers, which `current_allocated_size`
        /// counts too.
        pub fn no_copy_bytes(&self) -> u64 {
            self.no_copy.borrow().values().map(|b| b.length()).sum()
        }

        /// Maps a file read-only and wraps it as a buffer. Kernels must not
        /// write to it.
        pub fn map_file(&self, path: &Path) -> std::io::Result<MappedFile> {
            let file = File::open(path)?;
            let map = unsafe { Mmap::map(&file)? };
            let (buffer, _) = unsafe { self.buffer_no_copy(&map) };
            Ok(MappedFile { map, buffer })
        }

        /// Encodes work with `encode` into one serial command buffer, runs
        /// it and waits. Returns the wall time from commit to completion.
        pub fn run(&self, encode: impl FnOnce(&ComputeCommandEncoderRef)) -> Duration {
            self.run_with(MTLDispatchType::Serial, encode)
        }

        /// `run` with a chosen dispatch type for the encoder.
        pub fn run_with(
            &self,
            dispatch: MTLDispatchType,
            encode: impl FnOnce(&ComputeCommandEncoderRef),
        ) -> Duration {
            self.run_timed(dispatch, encode).0
        }

        /// `run_with`, also returning the GPU's own time for the command
        /// buffer in seconds (GPUEndTime - GPUStartTime).
        // The objc 0.2 msg_send! macro checks an old `cargo-clippy` cfg.
        #[allow(unexpected_cfgs)]
        pub fn run_timed(
            &self,
            dispatch: MTLDispatchType,
            encode: impl FnOnce(&ComputeCommandEncoderRef),
        ) -> (Duration, f64) {
            // The queue hands out autoreleased command buffers and encoders,
            // and server threads have no pool of their own, so without this
            // every command buffer of a request, with the buffers it retains,
            // stayed alive until its thread ended.
            metal::objc::rc::autoreleasepool(|| self.run_timed_in_pool(dispatch, encode))
        }

        // The objc 0.2 msg_send! macro checks an old `cargo-clippy` cfg.
        #[allow(unexpected_cfgs)]
        fn run_timed_in_pool(
            &self,
            dispatch: MTLDispatchType,
            encode: impl FnOnce(&ComputeCommandEncoderRef),
        ) -> (Duration, f64) {
            use metal::objc::{msg_send, sel, sel_impl};
            self.commit_residency();
            let commands = self.queue.new_command_buffer();
            let encoder = commands.compute_command_encoder_with_dispatch_type(dispatch);
            encode(encoder);
            encoder.end_encoding();
            let start = Instant::now();
            commands.commit();
            commands.wait_until_completed();
            let elapsed = start.elapsed();
            if commands.status() == metal::MTLCommandBufferStatus::Error {
                panic!(
                    "GPU command buffer failed: {}",
                    command_buffer_error(commands)
                );
            }
            let (t0, t1): (f64, f64) = unsafe {
                (
                    msg_send![commands, GPUStartTime],
                    msg_send![commands, GPUEndTime],
                )
            };
            (elapsed, t1 - t0)
        }
    }

    /// A command buffer committed by `Gpu::commit`: `wait` blocks until it
    /// is done and panics if it failed.
    pub struct Pending(metal::CommandBuffer);

    impl Pending {
        pub fn wait(self) {
            self.0.wait_until_completed();
            if self.0.status() == metal::MTLCommandBufferStatus::Error {
                panic!("GPU command buffer failed: {}", command_buffer_error(&self.0));
            }
        }
    }

    impl Gpu {
        /// `run_timed` without the wait: the command buffer is committed and
        /// runs while the CPU goes on; `Pending::wait` checks it.
        pub fn commit_with(
            &self,
            dispatch: MTLDispatchType,
            encode: impl FnOnce(&ComputeCommandEncoderRef),
        ) -> Pending {
            metal::objc::rc::autoreleasepool(|| {
                self.commit_residency();
                let commands = self.queue.new_command_buffer();
                let encoder = commands.compute_command_encoder_with_dispatch_type(dispatch);
                encode(encoder);
                encoder.end_encoding();
                commands.commit();
                Pending(commands.to_owned())
            })
        }
    }

    /// The error description of a failed command buffer.
    // The objc 0.2 msg_send! macro checks an old `cargo-clippy` cfg.
    #[allow(unexpected_cfgs)]
    fn command_buffer_error(commands: &metal::CommandBufferRef) -> String {
        use metal::objc::runtime::Object;
        use metal::objc::{msg_send, sel, sel_impl};
        unsafe {
            let error: *mut Object = msg_send![commands, error];
            if error.is_null() {
                return "no error object".into();
            }
            let text: *mut Object = msg_send![error, localizedDescription];
            let utf8: *const std::ffi::c_char = msg_send![text, UTF8String];
            std::ffi::CStr::from_ptr(utf8)
                .to_string_lossy()
                .into_owned()
        }
    }

    impl Gpu {
        /// Commits weight buffers added since the last command buffer to
        /// the residency set.
        pub fn commit_residency(&self) {
            if let Some(r) = &self.residency {
                r.commit();
            }
        }

        /// Encodes buffer copies and commits them without waiting. Work
        /// committed later on the queue runs after the copies, so callers
        /// only wait (on the returned command buffer) to time them.
        pub fn blit(&self, encode: impl FnOnce(&BlitCommandEncoderRef)) -> CommandBuffer {
            metal::objc::rc::autoreleasepool(|| {
                self.commit_residency();
                let commands = self.queue.new_command_buffer().to_owned();
                let encoder = commands.new_blit_command_encoder();
                encode(encoder);
                encoder.end_encoding();
                commands.commit();
                commands
            })
        }

        /// `blit` for compute work: one serial encoder, committed without
        /// waiting.
        pub fn commit(&self, encode: impl FnOnce(&ComputeCommandEncoderRef)) -> CommandBuffer {
            metal::objc::rc::autoreleasepool(|| {
                self.commit_residency();
                let commands = self.queue.new_command_buffer().to_owned();
                let encoder =
                    commands.compute_command_encoder_with_dispatch_type(MTLDispatchType::Serial);
                encode(encoder);
                encoder.end_encoding();
                commands.commit();
                commands
            })
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

//! Prompt cache snapshots on the GPU (docs/CACHE.md). A snapshot is one
//! buffer from a preallocated pool that holds the conv and recurrent state
//! of every DeltaNet layer, written and read back with blit copies. The KV
//! caches are not copied: going back only sets the model position.
//!
//! The copies are committed without waiting. Work committed after them on
//! the same queue runs after them, so the next token sees the copied state.

#[cfg(target_os = "macos")]
pub use mac::*;

#[cfg(target_os = "macos")]
mod mac {
    use crate::gpu::Gpu;
    use crate::gpu_layers::GpuDeltaNetState;
    use metal::{Buffer, CommandBuffer, Device, MTLResourceOptions};
    use std::sync::{Arc, Mutex};

    /// Snapshot buffers, all the same size. A snapshot gives its buffer back
    /// when it is dropped, for example when the cache evicts it.
    pub struct SnapshotPool {
        device: Device,
        size: u64,
        free: Arc<Mutex<Vec<Buffer>>>,
    }

    pub struct GpuSnapshot {
        buffer: Option<Buffer>,
        free: Arc<Mutex<Vec<Buffer>>>,
    }

    impl Drop for GpuSnapshot {
        fn drop(&mut self) {
            if let Some(buffer) = self.buffer.take() {
                self.free.lock().unwrap().push(buffer);
            }
        }
    }

    impl GpuSnapshot {
        fn buffer(&self) -> &Buffer {
            self.buffer.as_ref().unwrap()
        }
    }

    impl SnapshotPool {
        /// `count` buffers that each fit the states of `layers`. The cache
        /// adds a snapshot before it evicts one, so use its limit + 1.
        pub fn new(gpu: &Gpu, layers: &[&GpuDeltaNetState], count: usize) -> SnapshotPool {
            let size = layers
                .iter()
                .map(|s| s.conv.length() + s.recurrent.length())
                .sum();
            let pool = SnapshotPool {
                device: gpu.device.clone(),
                size,
                free: Arc::new(Mutex::new(Vec::new())),
            };
            let buffers = (0..count).map(|_| pool.new_buffer()).collect();
            *pool.free.lock().unwrap() = buffers;
            pool
        }

        /// Bytes per snapshot.
        pub fn size(&self) -> u64 {
            self.size
        }

        pub fn free_count(&self) -> usize {
            self.free.lock().unwrap().len()
        }

        fn new_buffer(&self) -> Buffer {
            self.device
                .new_buffer(self.size, MTLResourceOptions::StorageModePrivate)
        }

        /// A buffer from the pool, or a new one when the pool is empty.
        fn take(&self) -> GpuSnapshot {
            let buffer = self.free.lock().unwrap().pop();
            GpuSnapshot {
                buffer: Some(buffer.unwrap_or_else(|| self.new_buffer())),
                free: Arc::clone(&self.free),
            }
        }
    }

    /// Each state buffer with its offset in a snapshot.
    fn layout<'a>(layers: &[&'a GpuDeltaNetState]) -> Vec<(&'a Buffer, u64)> {
        let mut offset = 0;
        let mut out = Vec::new();
        for state in layers {
            for buffer in [&state.conv, &state.recurrent] {
                out.push((buffer, offset));
                offset += buffer.length();
            }
        }
        out
    }

    /// Copies the DeltaNet states into a new snapshot. The command buffer
    /// completes when the copy is done; there is no need to wait on it.
    pub fn save(
        gpu: &Gpu,
        pool: &SnapshotPool,
        layers: &[&GpuDeltaNetState],
    ) -> (GpuSnapshot, CommandBuffer) {
        let snapshot = pool.take();
        let commands = gpu.blit(|blit| {
            for (buffer, offset) in layout(layers) {
                blit.copy_from_buffer(buffer, 0, snapshot.buffer(), offset, buffer.length());
            }
        });
        (snapshot, commands)
    }

    /// Copies a snapshot back into the DeltaNet states. The caller sets the
    /// KV length (the model position) to the snapshot length.
    pub fn restore(
        gpu: &Gpu,
        snapshot: &GpuSnapshot,
        layers: &[&GpuDeltaNetState],
    ) -> CommandBuffer {
        gpu.blit(|blit| {
            for (buffer, offset) in layout(layers) {
                blit.copy_from_buffer(snapshot.buffer(), offset, buffer, 0, buffer.length());
            }
        })
    }

    #[cfg(test)]
    mod tests {
        //!   scripts/mac.sh test --release --lib gpu_cache -- --ignored --nocapture

        use super::*;
        use crate::config::Config;
        use crate::cpu::model::deltanet_config;
        use crate::cpu::ops::rms_norm;
        use crate::gpu::contents;
        use crate::gpu_layers::{GpuAttention, GpuDeltaNet, GpuKvCache};
        use crate::qmv::View;
        use crate::weights::Weights;
        use std::path::PathBuf;
        use std::time::Instant;

        const MODEL_DIR: &str = "llm/saetta-models/Qwen3.6-35B-A3B-5bit";

        fn open() -> (Config, Weights) {
            let dir = PathBuf::from(std::env::var("HOME").unwrap()).join(MODEL_DIR);
            (Config::load(&dir).unwrap(), Weights::open(&dir).unwrap())
        }

        /// A normed hidden state for token `t`.
        fn input(t: usize, eps: f32) -> Vec<f32> {
            let mut x: Vec<f32> = (0..2048)
                .map(|i| ((i * 37 + t * 101) as f32 * 0.013).sin())
                .collect();
            rms_norm(&mut x, &vec![1.0; 2048], eps);
            x
        }

        fn same_bits(a: &[f32], b: &[f32]) -> bool {
            a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits())
        }

        /// Layer 0 (DeltaNet) and layer 3 (full attention) on the GPU: going
        /// back to a snapshot and running the same tokens gives the same
        /// outputs, bit for bit, as running them straight.
        #[test]
        #[ignore = "needs the model on the Mac"]
        fn restore_matches_a_straight_run() {
            let (cfg, weights) = open();
            let gpu = Gpu::new().unwrap();
            let p = |i: usize| format!("language_model.model.layers.{i}");
            let delta = GpuDeltaNet::load(&gpu, &cfg, &weights, &p(0)).unwrap();
            let attn = GpuAttention::load(&gpu, &cfg, &weights, &p(3)).unwrap();
            let out_d = View::new(gpu.buffer(cfg.hidden_size * 4));
            let out_a = View::new(gpu.buffer(cfg.hidden_size * 4));
            let eps = cfg.rms_norm_eps;

            let run = |t: usize, state: &GpuDeltaNetState, kv: &mut GpuKvCache| {
                let x = View::new(gpu.buffer_from(&input(t, eps)));
                gpu.run(|enc| {
                    delta.encode(&gpu, enc, &x, &out_d, state);
                    attn.encode(&gpu, enc, &x, &out_a, kv);
                });
                let n = cfg.hidden_size;
                let mut out = contents::<f32>(&out_d.buffer, n).to_vec();
                out.extend_from_slice(contents::<f32>(&out_a.buffer, n));
                out
            };

            let straight_state = GpuDeltaNetState::new(&gpu, &delta.cfg);
            let mut straight_kv = GpuKvCache::new(&gpu, &attn.cfg, 64);
            let mut want = Vec::new();
            for t in 0..12 {
                want = run(t, &straight_state, &mut straight_kv);
            }

            let state = GpuDeltaNetState::new(&gpu, &delta.cfg);
            let mut kv = GpuKvCache::new(&gpu, &attn.cfg, 64);
            let pool = SnapshotPool::new(&gpu, &[&state], 2);
            for t in 0..6 {
                run(t, &state, &mut kv);
            }
            let (snapshot, _) = save(&gpu, &pool, &[&state]);
            let mut detour = Vec::new();
            for t in 100..104 {
                detour = run(t, &state, &mut kv);
            }
            restore(&gpu, &snapshot, &[&state]);
            kv.len = 6;
            let mut got = Vec::new();
            for t in 6..12 {
                got = run(t, &state, &mut kv);
            }
            assert!(!same_bits(&detour, &want));
            assert!(same_bits(&got, &want), "outputs differ after restore");
        }

        /// Times save and restore of the 30 DeltaNet states of the 35B model,
        /// on stand-in buffers of the real sizes.
        #[test]
        #[ignore = "uses the GPU for about a second"]
        fn save_and_restore_time() {
            let (cfg, _) = open();
            let gpu = Gpu::new().unwrap();
            let dcfg = deltanet_config(&cfg);
            let states: Vec<GpuDeltaNetState> = (0..30)
                .map(|_| GpuDeltaNetState::new(&gpu, &dcfg))
                .collect();
            let layers: Vec<&GpuDeltaNetState> = states.iter().collect();
            let pool = SnapshotPool::new(&gpu, &layers, 17);
            println!("snapshot size {:.1} MB", pool.size() as f64 / 1e6);

            // The first use of each pool buffer maps its memory; do that once.
            let warm: Vec<_> = (0..17).map(|_| save(&gpu, &pool, &layers)).collect();
            for (_, commands) in &warm {
                commands.wait_until_completed();
            }
            drop(warm);

            let runs = 20;
            let (mut save_cpu, mut save_all, mut restore_cpu, mut restore_all) =
                (0.0, 0.0, 0.0, 0.0);
            let mut kept = Vec::new();
            for _ in 0..runs {
                let start = Instant::now();
                let (snapshot, commands) = save(&gpu, &pool, &layers);
                save_cpu += start.elapsed().as_secs_f64();
                commands.wait_until_completed();
                save_all += start.elapsed().as_secs_f64();

                let start = Instant::now();
                let commands = restore(&gpu, &snapshot, &layers);
                restore_cpu += start.elapsed().as_secs_f64();
                commands.wait_until_completed();
                restore_all += start.elapsed().as_secs_f64();
                kept.push(snapshot);
                if kept.len() > 16 {
                    kept.remove(0);
                }
            }
            let ms = |s: f64| s * 1e3 / runs as f64;
            println!(
                "save: {:.3} ms to encode and commit, {:.3} ms until done ({runs} runs)",
                ms(save_cpu),
                ms(save_all)
            );
            println!(
                "restore: {:.3} ms to encode and commit, {:.3} ms until done ({runs} runs)",
                ms(restore_cpu),
                ms(restore_all)
            );
            assert_eq!(kept.len() + pool.free_count(), 17);
        }
    }
}

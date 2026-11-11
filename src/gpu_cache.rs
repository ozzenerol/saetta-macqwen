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
    use crate::gpu_layers::{GpuDeltaNetState, GpuKvCache};
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

        /// A buffer from the pool, or a new one when the pool is empty. The
        /// cache never holds more than the pool, so a new one means a bug;
        /// it is logged, as it goes past `--cache-mb`.
        fn take(&self) -> GpuSnapshot {
            let buffer = self.free.lock().unwrap().pop();
            GpuSnapshot {
                buffer: Some(buffer.unwrap_or_else(|| {
                    eprintln!("snapshot pool is empty: allocating past the cache limit");
                    self.new_buffer()
                })),
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

    /// A copy of KV rows `from..from + n` of every attention layer, for a
    /// short detour that writes over a long conversation (cache::Restore).
    /// The buffers are made for each copy at its size and freed when the
    /// rows go back or a new copy replaces them.
    pub struct RowBackup {
        copy: Option<RowCopy>,
    }

    /// Rows `from..from + n` of each buffer of each cache, in the order of
    /// `GpuKvCache::buffers`.
    struct RowCopy {
        parts: Vec<Buffer>,
        from: usize,
        n: usize,
    }

    /// Bytes per row of one cache buffer: its length over the capacity.
    fn row_bytes(buffer: &Buffer, cache: &GpuKvCache) -> u64 {
        buffer.length() / cache.capacity as u64
    }

    /// Each cache buffer with its row size, in a fixed order.
    fn parts<'c>(caches: &[&'c GpuKvCache]) -> Vec<(&'c Buffer, u64)> {
        caches
            .iter()
            .flat_map(|c| c.buffers().into_iter().map(move |b| (b, row_bytes(b, c))))
            .collect()
    }

    impl RowBackup {
        pub fn new() -> RowBackup {
            RowBackup { copy: None }
        }

        /// Bytes held now.
        pub fn bytes(&self) -> u64 {
            self.copy
                .as_ref()
                .map_or(0, |c| c.parts.iter().map(|b| b.length()).sum())
        }

        /// Copies rows `from..from + n` out of the caches.
        pub fn save(&mut self, gpu: &Gpu, caches: &[&GpuKvCache], from: usize, n: usize) {
            assert!(n <= crate::cache::BACKUP_ROWS);
            self.copy = None;
            if n == 0 || caches.is_empty() {
                return;
            }
            let parts = parts(caches);
            let copies: Vec<Buffer> = parts
                .iter()
                .map(|(_, row)| {
                    gpu.device
                        .new_buffer(n as u64 * row, MTLResourceOptions::StorageModePrivate)
                })
                .collect();
            gpu.blit(|blit| {
                for ((buffer, row), copy) in parts.iter().zip(&copies) {
                    blit.copy_from_buffer(buffer, from as u64 * row, copy, 0, n as u64 * row);
                }
            });
            self.copy = Some(RowCopy {
                parts: copies,
                from,
                n,
            });
        }

        /// Copies the rows back where they came from and frees the copy.
        /// The buffers stay alive until the GPU is done with them, since the
        /// command buffer retains them.
        pub fn restore(&mut self, gpu: &Gpu, caches: &[&GpuKvCache]) {
            let Some(copy) = self.copy.take() else {
                return;
            };
            let parts = parts(caches);
            gpu.blit(|blit| {
                for ((buffer, row), saved) in parts.iter().zip(&copy.parts) {
                    let (start, len) = (copy.from as u64 * row, copy.n as u64 * row);
                    blit.copy_from_buffer(saved, 0, buffer, start, len);
                }
            });
        }

        /// Frees the copy without putting it back.
        pub fn clear(&mut self) {
            self.copy = None;
        }
    }

    impl Default for RowBackup {
        fn default() -> RowBackup {
            RowBackup::new()
        }
    }

    #[cfg(test)]
    mod tests {
        //!   scripts/mac.sh test --release --lib gpu_cache -- --ignored --nocapture

        use super::*;
        use crate::config::Config;
        use crate::cpu::model::deltanet_config;
        use crate::gpu_model::GpuModel;
        use crate::weights::Weights;
        use std::path::PathBuf;
        use std::time::Instant;

        const MODEL_DIR: &str = "llm/saetta-models/Qwen3.6-35B-A3B-5bit";

        fn open() -> (Config, Weights) {
            let dir = PathBuf::from(std::env::var("HOME").unwrap()).join(MODEL_DIR);
            (Config::load(&dir).unwrap(), Weights::open(&dir).unwrap())
        }

        fn same_bits(a: &[f32], b: &[f32]) -> bool {
            a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits())
        }

        /// The GPU model, 8 layers by default and the logits: going back to a
        /// snapshot and running the same tokens gives the same logits, bit for
        /// bit, as running straight.
        #[test]
        #[ignore = "needs the model on the Mac"]
        fn restore_matches_a_straight_run() {
            let (cfg, weights) = open();
            let gpu = Gpu::new().unwrap();
            // The whole model does not fit next to mlx_lm.server; SAETTA_LAYERS=40
            // runs all of it when MLX is stopped.
            let layers = std::env::var("SAETTA_LAYERS").map_or(8, |n| n.parse().unwrap());
            let mut m = GpuModel::load(&gpu, &cfg, &weights, layers, 64).unwrap();
            // "The capital of France is" and other short text, as token ids.
            let prompt = [
                760, 6511, 314, 9625, 369, 11, 323, 279, 6511, 314, 9419, 369,
            ];
            let detour = [12345, 678, 9012, 3456];
            let straight = |m: &mut GpuModel| {
                m.reset();
                prompt.iter().map(|&t| m.forward(t)).last().unwrap()
            };
            let want = straight(&mut m);
            assert!(
                same_bits(&straight(&mut m), &want),
                "two straight runs differ"
            );

            m.reset();
            for &t in &prompt[..6] {
                m.run(t);
            }
            let pool = SnapshotPool::new(&gpu, &m.deltanet_states(), 1);
            let (snapshot, _) = save(&gpu, &pool, &m.deltanet_states());
            let mut off = Vec::new();
            for &t in &detour {
                off = m.forward(t);
            }
            restore(&gpu, &snapshot, &m.deltanet_states());
            m.set_position(6);
            let mut got = Vec::new();
            for &t in &prompt[6..] {
                got = m.forward(t);
            }
            assert!(!same_bits(&off, &want));
            assert!(same_bits(&got, &want), "logits differ after restore");
        }

        fn caches<'m>(m: &'m GpuModel) -> Vec<&'m GpuKvCache> {
            m.states()
                .filter_map(|s| match s {
                    crate::gpu_model::State::Full(c) => Some(c),
                    crate::gpu_model::State::Linear(_) => None,
                })
                .collect()
        }

        /// A detour from position 2 writes over KV rows of the prompt; with
        /// those rows backed up and put back, the prompt's snapshot gives the
        /// same logits, bit for bit, as running straight.
        #[test]
        #[ignore = "needs the model on the Mac"]
        fn row_backup_undoes_a_detour() {
            let (cfg, weights) = open();
            let gpu = Gpu::new().unwrap();
            let layers = std::env::var("SAETTA_LAYERS").map_or(8, |n| n.parse().unwrap());
            let mut m = GpuModel::load(&gpu, &cfg, &weights, layers, 64).unwrap();
            assert!(!caches(&m).is_empty(), "needs an attention layer");
            let prompt = [
                760, 6511, 314, 9625, 369, 11, 323, 279, 6511, 314, 9419, 369,
            ];
            m.reset();
            let want = prompt.iter().map(|&t| m.forward(t)).last().unwrap();

            m.reset();
            for &t in &prompt[..8] {
                m.run(t);
            }
            let pool = SnapshotPool::new(&gpu, &m.deltanet_states(), 1);
            let (snapshot, copy) = save(&gpu, &pool, &m.deltanet_states());
            let mut rows = RowBackup::new();
            rows.save(&gpu, &caches(&m), 2, 6);
            // reset zeroes the states from the CPU, as GpuServed waits too.
            copy.wait_until_completed();

            // The detour: back to 2 tokens (a reset stands in for a snapshot
            // there) and other tokens into rows 2..6.
            m.reset();
            for &t in &[prompt[0], prompt[1], 12345, 678, 9012, 3456] {
                m.run(t);
            }
            m.forward(prompt[0]);
            // Without the rows put back the detour shows.
            restore(&gpu, &snapshot, &m.deltanet_states());
            m.set_position(8);
            let off = prompt[8..].iter().map(|&t| m.forward(t)).last().unwrap();
            assert!(!same_bits(&off, &want), "the detour changed nothing");
            rows.restore(&gpu, &caches(&m));
            assert_eq!(rows.bytes(), 0);

            restore(&gpu, &snapshot, &m.deltanet_states());
            m.set_position(8);
            let got = prompt[8..].iter().map(|&t| m.forward(t)).last().unwrap();
            assert!(same_bits(&got, &want), "logits differ after the detour");
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

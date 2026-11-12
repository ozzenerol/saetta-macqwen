//! Prompt cache snapshots on the GPU (docs/CACHE.md). A snapshot is one
//! buffer that holds the conv and recurrent state of every DeltaNet layer,
//! written and read back with blit copies. It is made when the snapshot is
//! saved and freed when it is dropped. The KV caches are not copied: going
//! back only sets the model position.
//!
//! With `SnapshotPool::half`, a snapshot holds the states in f16, half the
//! memory, written and read with a conversion kernel instead of a blit.
//!
//! The copies are committed without waiting. Work committed after them on
//! the same queue runs after them, so the next token sees the copied state.

#[cfg(target_os = "macos")]
pub use mac::*;

#[cfg(target_os = "macos")]
mod mac {
    use crate::gpu::{Gpu, dispatch_1d};
    use crate::gpu_layers::{GpuDeltaNetState, GpuKvCache};
    use metal::{Buffer, CommandBuffer, Device, MTLResourceOptions};
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};

    /// Makes the snapshot buffers, all the same size. A buffer is made when
    /// a snapshot is saved and freed when the snapshot is dropped, for
    /// example when the cache evicts it, so the memory follows the snapshots
    /// the cache holds instead of staying at the most it ever held.
    pub struct SnapshotPool {
        device: Device,
        size: u64,
        limit: usize,
        live: Arc<AtomicUsize>,
        /// States are stored in f16.
        half: bool,
    }

    pub struct GpuSnapshot {
        buffer: Buffer,
        live: Arc<AtomicUsize>,
        half: bool,
    }

    impl Drop for GpuSnapshot {
        fn drop(&mut self) {
            // A blit still in flight retains the buffer until it is done.
            self.live.fetch_sub(1, Ordering::Relaxed);
        }
    }

    impl GpuSnapshot {
        fn buffer(&self) -> &Buffer {
            &self.buffer
        }
    }

    impl SnapshotPool {
        /// A pool of at most `count` snapshots that each fit the states of
        /// `layers`. The cache adds a snapshot before it evicts one, so use
        /// its limit + 1.
        pub fn new(gpu: &Gpu, layers: &[&GpuDeltaNetState], count: usize) -> SnapshotPool {
            SnapshotPool {
                device: gpu.device.clone(),
                size: layers
                    .iter()
                    .map(|s| s.conv.length() + s.recurrent.length())
                    .sum(),
                limit: count,
                live: Arc::new(AtomicUsize::new(0)),
                half: false,
            }
        }

        /// Stores the states in f16: half the bytes per snapshot. The states
        /// are f32, so a restore starts from rounded states.
        pub fn half(mut self, half: bool) -> SnapshotPool {
            if half != self.half {
                self.size = if half { self.size / 2 } else { self.size * 2 };
                self.half = half;
            }
            self
        }

        /// Bytes per snapshot.
        pub fn size(&self) -> u64 {
            self.size
        }

        /// Snapshots that can still be made within the limit.
        pub fn free_count(&self) -> usize {
            self.limit.saturating_sub(self.live.load(Ordering::Relaxed))
        }

        /// Snapshots alive now.
        pub fn live(&self) -> usize {
            self.live.load(Ordering::Relaxed)
        }

        /// A new snapshot buffer. The cache never holds more than the
        /// limit, so going past it means a bug; it is logged, as it goes
        /// past `--cache-mb`.
        fn take(&self) -> GpuSnapshot {
            if self.live.fetch_add(1, Ordering::Relaxed) >= self.limit {
                eprintln!("snapshot pool is full: allocating past the cache limit");
            }
            GpuSnapshot {
                buffer: self
                    .device
                    .new_buffer(self.size, MTLResourceOptions::StorageModePrivate),
                live: Arc::clone(&self.live),
                half: self.half,
            }
        }
    }

    /// Each state buffer with its offset in a snapshot, in bytes of the
    /// snapshot (half of the state's bytes with `half`).
    fn layout<'a>(layers: &[&'a GpuDeltaNetState], half: bool) -> Vec<(&'a Buffer, u64)> {
        let mut offset = 0;
        let mut out = Vec::new();
        for state in layers {
            for buffer in [&state.conv, &state.recurrent] {
                out.push((buffer, offset));
                offset += if half {
                    buffer.length() / 2
                } else {
                    buffer.length()
                };
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
        let commands = if snapshot.half {
            gpu.commit(|enc| {
                let pipeline = gpu.pipeline("to_half");
                for (buffer, offset) in layout(layers, true) {
                    let floats = buffer.length() as usize / 4;
                    dispatch_1d(
                        enc,
                        &pipeline,
                        &[(buffer, 0), (snapshot.buffer(), offset)],
                        floats,
                    );
                }
            })
        } else {
            gpu.blit(|blit| {
                for (buffer, offset) in layout(layers, false) {
                    blit.copy_from_buffer(buffer, 0, snapshot.buffer(), offset, buffer.length());
                }
            })
        };
        (snapshot, commands)
    }

    /// Copies a snapshot back into the DeltaNet states. The caller sets the
    /// KV length (the model position) to the snapshot length.
    pub fn restore(
        gpu: &Gpu,
        snapshot: &GpuSnapshot,
        layers: &[&GpuDeltaNetState],
    ) -> CommandBuffer {
        if snapshot.half {
            return gpu.commit(|enc| {
                let pipeline = gpu.pipeline("from_half");
                for (buffer, offset) in layout(layers, true) {
                    let floats = buffer.length() as usize / 4;
                    dispatch_1d(
                        enc,
                        &pipeline,
                        &[(snapshot.buffer(), offset), (buffer, 0)],
                        floats,
                    );
                }
            });
        }
        gpu.blit(|blit| {
            for (buffer, offset) in layout(layers, false) {
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

        fn log_softmax(x: &[f32]) -> Vec<f64> {
            let max = x.iter().copied().fold(f32::MIN, f32::max) as f64;
            let sum: f64 = x.iter().map(|&v| (v as f64 - max).exp()).sum();
            x.iter().map(|&v| v as f64 - max - sum.ln()).collect()
        }

        /// KL(p || q) of the distributions of two logit vectors.
        fn kl(p: &[f32], q: &[f32]) -> f64 {
            let (lp, lq) = (log_softmax(p), log_softmax(q));
            lp.iter().zip(&lq).map(|(a, b)| a.exp() * (a - b)).sum()
        }

        fn argmax(x: &[f32]) -> usize {
            (0..x.len()).max_by(|&a, &b| x[a].total_cmp(&x[b])).unwrap()
        }

        /// This repo's docs and sources, as real text for the model.
        fn repo_text() -> String {
            let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
            let mut paths = Vec::new();
            for dir in ["docs", "src"] {
                for entry in std::fs::read_dir(root.join(dir)).unwrap() {
                    let path = entry.unwrap().path();
                    if path.extension().is_some_and(|e| e == "md" || e == "rs") {
                        paths.push(path);
                    }
                }
            }
            paths.sort();
            paths
                .iter()
                .map(|p| std::fs::read_to_string(p).unwrap())
                .collect()
        }

        /// f16 snapshots (`--half-snapshots`): the logits after restoring
        /// f16 states against a run without restores, over the whole model,
        /// at 4k and 32k of context (SAETTA_HALF_CTX=4096,32768) of this
        /// repo's docs and sources. Each of 4 rounds saves, runs a detour,
        /// restores and reads 16 more tokens of the text, so the rounding
        /// compounds as over the turns of a qh session. It compares f16
        /// against f32 snapshots on the same path, and both against the
        /// straight run, and passes when all three meet the quality gate:
        /// median KL < 1e-3, top-1 >= 90%.
        #[test]
        #[ignore = "needs the whole model on the GPU"]
        fn half_snapshot_error() {
            use crate::prefill::{Prefill, exact_from_env};
            const ROUNDS: usize = 4;
            const PER_ROUND: usize = 16;
            const DETOUR: [u32; 4] = [12345, 678, 9012, 3456];
            let dir = PathBuf::from(std::env::var("HOME").unwrap()).join(MODEL_DIR);
            let (cfg, weights) = open();
            let tokenizer = crate::tokenizer::Tokenizer::load(&dir).unwrap();
            let text = tokenizer.encode(&repo_text());
            let contexts: Vec<usize> = std::env::var("SAETTA_HALF_CTX")
                .unwrap_or_else(|_| "4096,32768".into())
                .split(',')
                .map(|c| c.trim().parse().unwrap())
                .collect();
            let gpu = Gpu::new().unwrap();
            let mut prefill = Prefill::new(&gpu, &cfg, &weights, 2048).unwrap();
            prefill.exact = exact_from_env();
            let mut failed = Vec::new();
            for &ctx in &contexts {
                let n = ctx + ROUNDS * PER_ROUND;
                let tokens: Vec<u32> = text.iter().copied().cycle().take(n).collect();
                let (prompt, forced) = tokens.split_at(ctx);
                let mut m = GpuModel::load(&gpu, &cfg, &weights, cfg.num_layers(), n + 64).unwrap();
                let fill = |m: &mut GpuModel| {
                    m.reset();
                    m.reserve(n + DETOUR.len());
                    let states: Vec<_> = m.states().collect();
                    prefill.run(&gpu, prompt, 0, &states);
                    m.set_position(ctx);
                };
                fill(&mut m);
                let want: Vec<Vec<f32>> = forced.iter().map(|&t| m.forward(t)).collect();
                // Whether a straight run repeats bit for bit, to tell apart
                // differences of the restores from those of the runs.
                fill(&mut m);
                let again = forced
                    .iter()
                    .all(|&t| same_bits(&m.forward(t), &want[m.position() - ctx - 1]));
                println!("{ctx}: a second straight run matches bit for bit: {again}");
                let mut runs = Vec::new();
                for half in [false, true] {
                    fill(&mut m);
                    let pool = SnapshotPool::new(&gpu, &m.deltanet_states(), 2).half(half);
                    let mut got = Vec::new();
                    for round in forced.chunks(PER_ROUND) {
                        let (snapshot, copy) = save(&gpu, &pool, &m.deltanet_states());
                        copy.wait_until_completed();
                        let pos = m.position();
                        for &t in &DETOUR {
                            m.run(t);
                        }
                        restore(&gpu, &snapshot, &m.deltanet_states()).wait_until_completed();
                        m.set_position(pos);
                        got.extend(round.iter().map(|&t| m.forward(t)));
                    }
                    runs.push(got);
                }
                let (f32s, f16s) = (&runs[0], &runs[1]);
                let exact = f32s.iter().zip(&want).all(|(g, w)| same_bits(g, w));
                println!("{ctx}: f32 snapshots match the straight run bit for bit: {exact}");
                // f32 against the straight run checks restores themselves;
                // f16 against f32 is the same path, so only the rounding of
                // the states differs.
                for (name, got, base) in [
                    ("f32 snapshots vs the straight run", f32s, &want),
                    ("f16 vs f32 snapshots", f16s, f32s),
                    ("f16 snapshots vs the straight run", f16s, &want),
                ] {
                    let mut kls: Vec<f64> = got.iter().zip(base).map(|(g, w)| kl(w, g)).collect();
                    let top1 = got
                        .iter()
                        .zip(base)
                        .filter(|(g, w)| argmax(g) == argmax(w))
                        .count() as f64
                        / got.len() as f64;
                    let rounds: Vec<String> = kls
                        .chunks(PER_ROUND)
                        .map(|k| format!("{:.2e}", k.iter().copied().fold(0.0, f64::max)))
                        .collect();
                    kls.sort_by(f64::total_cmp);
                    let (median, max) = (kls[kls.len() / 2], kls[kls.len() - 1]);
                    println!(
                        "{ctx} tokens, {name}: median KL {median:.2e}, max {max:.2e}, top-1 {:.1}% over {} tokens, max KL by round {}",
                        top1 * 100.0,
                        kls.len(),
                        rounds.join(" ")
                    );
                    if median >= 1e-3 || top1 < 0.9 {
                        failed.push(format!(
                            "{ctx}, {name}: median KL {median:.2e}, top-1 {top1:.2}"
                        ));
                    }
                }
            }
            assert!(failed.is_empty(), "{failed:?}");
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
            for half in [false, true] {
                let pool = SnapshotPool::new(&gpu, &layers, 17).half(half);
                println!(
                    "half {half}: snapshot size {:.1} MB",
                    pool.size() as f64 / 1e6
                );

                // Every save makes a new buffer, so the times include mapping
                // its memory on first use, as in the server.
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
}

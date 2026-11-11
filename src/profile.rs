//! Where the time of one decode token goes (`saetta profile-token`).

#[cfg(target_os = "macos")]
pub use mac::*;

#[cfg(target_os = "macos")]
mod mac {
    use crate::config::Config;
    use crate::gpu::Gpu;
    use crate::gpu_model::GpuModel;
    use crate::plan::{Built, TokenParams};
    use crate::weights::Weights;
    use std::path::Path;
    use std::time::Instant;

    fn median(mut v: Vec<f64>) -> f64 {
        v.sort_by(f64::total_cmp);
        v[v.len() / 2]
    }

    /// Median wall and GPU seconds of `runs` replays after one warm-up.
    fn time(gpu: &Gpu, built: &Built, runs: usize) -> (f64, f64) {
        built.run(gpu);
        let t: Vec<(f64, f64)> = (0..runs)
            .map(|_| {
                let (wall, g) = built.run_timed(gpu);
                (wall.as_secs_f64(), g)
            })
            .collect();
        (
            median(t.iter().map(|x| x.0).collect()),
            median(t.iter().map(|x| x.1).collect()),
        )
    }

    /// Profiles one token at `position`: the whole token as the server runs
    /// it, the token replayed without barriers, and the ops of each kernel
    /// alone (all their layers, with barriers).
    pub fn profile_token(
        dir: &Path,
        position: usize,
        runs: usize,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let cfg = Config::load(dir)?;
        let weights = Weights::open(dir)?;
        let gpu = Gpu::new()?;
        let mut model = GpuModel::load(&gpu, &cfg, &weights, cfg.num_layers(), position + 2)?;

        let forward: Vec<f64> = (0..=runs)
            .map(|_| {
                model.set_position(position);
                let start = Instant::now();
                std::hint::black_box(model.forward(220));
                start.elapsed().as_secs_f64()
            })
            .skip(1)
            .collect();
        let run: Vec<f64> = (0..runs)
            .map(|_| {
                model.set_position(position);
                let start = Instant::now();
                model.run(220);
                start.elapsed().as_secs_f64()
            })
            .collect();
        let (forward, run) = (median(forward), median(run));

        let plan = model.plan();
        plan.set_params(TokenParams::new(220, position as u32));
        let (wall, gpu_time) = time(&gpu, &plan.build(), runs);
        let (_, free) = time(&gpu, &plan.build_where(|_| true, false), runs);
        let serial: Vec<f64> = (0..runs).map(|_| plan.run_serial().as_secs_f64()).collect();
        let serial = median(serial);

        println!(
            "one decode token at position {position}, {} ops, median of {runs}:",
            plan.len()
        );
        println!(
            "  forward (run + logits read back)  {:7.3} ms wall",
            forward * 1e3
        );
        println!(
            "  run (no read back)                {:7.3} ms wall",
            run * 1e3
        );
        println!(
            "  ICB replay                        {:7.3} ms wall, {:.3} ms GPU",
            wall * 1e3,
            gpu_time * 1e3
        );
        println!(
            "  ICB replay, no barriers           {:7.3} ms GPU (wrong results, lower bound)",
            free * 1e3
        );
        println!(
            "  serial encoder                    {:7.3} ms wall",
            serial * 1e3
        );
        println!("per kernel, its ops alone with barriers (GPU ms per token):");
        let mut rows = Vec::new();
        for (kernel, count) in plan.kernels() {
            let (_, t) = time(&gpu, &plan.build_where(|k| k == kernel, true), runs);
            rows.push((kernel, count, t));
        }
        rows.sort_by(|a, b| b.2.total_cmp(&a.2));
        let sum: f64 = rows.iter().map(|r| r.2).sum();
        for (kernel, count, t) in &rows {
            println!(
                "  {kernel:<22} {count:>4} ops  {:7.3} ms  {:5.1}%  {:6.1} us/op",
                t * 1e3,
                t / sum * 100.0,
                t / *count as f64 * 1e6
            );
        }
        println!(
            "  sum of kernels         {:7.3} ms (each group has its own launch cost)",
            sum * 1e3
        );
        Ok(())
    }
}

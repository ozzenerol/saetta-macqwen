//! GPU time of one sample_partial + sample_final over the full vocab.

#[cfg(target_os = "macos")]
fn main() {
    use saetta::gpu::Gpu;
    use saetta::plan::{Plan, TokenParams};
    use saetta::qmv::View;
    use saetta::sample::{Params, Sampler};
    use saetta::sample_gpu::GpuSampler;

    let gpu = Gpu::new().unwrap();
    let n = 248_320;
    let mut s = Sampler::new(1);
    let x: Vec<f32> = (0..n).map(|_| 20.0 * s.next_f32()).collect();
    let buf = View::new(gpu.buffer_from(&x));
    let sampler = GpuSampler::new(&gpu);
    let mut plan = Plan::new(&gpu);
    // 100 copies in one command buffer, so the GPU clocks up as it does
    // inside a whole token.
    for _ in 0..100 {
        sampler.record(&mut plan, &buf);
    }
    let built = plan.build();
    for k in ["sample_partial", "sample_final"] {
        let p = gpu.pipeline(k);
        println!(
            "{k}: max threads {}, simd width {}",
            p.max_total_threads_per_threadgroup(),
            p.thread_execution_width()
        );
    }
    let only_partial = plan.build_where(|k| k == "sample_partial", true);
    let only_final = plan.build_where(|k| k == "sample_final", true);
    for k in [1, 20, 64] {
        let p = Params {
            temperature: 0.7,
            top_p: 0.95,
            top_k: k,
        };
        sampler.set(n, &p, false);
        let mut t: Vec<f64> = (0..50)
            .map(|_| {
                plan.set_params(TokenParams::new(0, 0));
                built.run_timed(&gpu).1
            })
            .collect();
        t.sort_by(f64::total_cmp);
        let med = |b: &saetta::plan::Built| {
            let mut t: Vec<f64> = (0..50).map(|_| b.run_timed(&gpu).1).collect();
            t.sort_by(f64::total_cmp);
            t[25] * 1e4
        };
        println!(
            "top_k {k:2}: {:.1} us GPU per sample (median of 50); partial {:.1}, final {:.1}",
            t[25] * 1e4,
            med(&only_partial),
            med(&only_final)
        );
    }
}

#[cfg(not(target_os = "macos"))]
fn main() {}

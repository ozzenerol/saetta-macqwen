#![cfg(target_os = "macos")]
//! The GPU sampler draws what sample::Sampler draws for the same seed, and
//! its draft rule keeps the target distribution.

use saetta::gpu::Gpu;
use saetta::plan::{Plan, TokenParams};
use saetta::qmv::View;
use saetta::sample::{Params, Sampler};
use saetta::sample_gpu::{GpuSampler, slot};

const VOCAB: usize = 248_320;

fn logits(n: usize, seed: u64) -> Vec<f32> {
    let mut s = Sampler::new(seed);
    // A peaked tail like a real model's: a few high logits, most far below.
    (0..n)
        .map(|_| {
            let u = s.next_f32();
            if u < 0.001 {
                10.0 + 8.0 * s.next_f32()
            } else {
                12.0 * s.next_f32() - 6.0
            }
        })
        .collect()
}

fn setup<'g>(gpu: &'g Gpu, x: &[f32]) -> (Plan<'g>, GpuSampler) {
    let buf = View::new(gpu.buffer_from(x));
    let sampler = GpuSampler::new(gpu);
    let mut plan = Plan::new(gpu);
    sampler.record(&mut plan, &buf);
    (plan, sampler)
}

#[test]
fn matches_cpu_draws() {
    let gpu = Gpu::new().unwrap();
    let x = logits(VOCAB, 3);
    let (plan, sampler) = setup(&gpu, &x);
    let cases = [
        Params {
            temperature: 0.0,
            top_p: 1.0,
            top_k: 0,
        },
        Params {
            temperature: 0.6,
            top_p: 0.95,
            top_k: 20,
        },
        Params {
            temperature: 1.0,
            top_p: 1.0,
            top_k: 64,
        },
        Params {
            temperature: 0.7,
            top_p: 0.8,
            top_k: 40,
        },
        Params {
            temperature: 2.0,
            top_p: 0.5,
            top_k: 1,
        },
    ];
    for p in &cases {
        sampler.set(VOCAB, p, false);
        sampler.seed(11);
        let mut cpu = Sampler::new(11);
        let mut differ = 0;
        for _ in 0..300 {
            plan.set_params(TokenParams::new(0, 99));
            plan.run_serial();
            if sampler.token(100) != cpu.sample(&x, p) {
                differ += 1;
            }
        }
        assert_eq!(differ, 0, "{p:?}");
    }
}

#[test]
fn advance_feeds_the_next_position() {
    let gpu = Gpu::new().unwrap();
    let x = logits(VOCAB, 5);
    let (plan, sampler) = setup(&gpu, &x);
    let p = Params {
        temperature: 0.0,
        top_p: 1.0,
        top_k: 0,
    };
    sampler.set(VOCAB, &p, true);
    plan.set_params(TokenParams::new(0, 10));
    let built = plan.build();
    for _ in 0..3 {
        built.run(&gpu);
    }
    let want = saetta::sample::argmax(&x);
    let r = sampler.read();
    for pos in 11..14 {
        assert_eq!(r.ids[slot(pos)], want);
    }
    assert_eq!(r.probs[slot(11)], 1.0);
}

#[test]
fn draft_rule_keeps_the_distribution() {
    let gpu = Gpu::new().unwrap();
    let x = [1.0f32, 3.0, 2.0, 0.0, 3.0, -1.0, 2.5, 0.5];
    let (plan, sampler) = setup(&gpu, &x);
    let p = Params {
        temperature: 0.8,
        top_p: 0.9,
        top_k: 6,
    };
    let want: Vec<f64> = {
        let c = Sampler::new(0).candidates(&x, &p);
        let w: Vec<f64> = c
            .iter()
            .map(|&(_, l)| ((l - c[0].1) as f64 / 0.8).exp())
            .collect();
        let t: f64 = w.iter().sum();
        let mut want = vec![0.0; x.len()];
        for (&(id, _), w) in c.iter().zip(&w) {
            want[id as usize] = w / t;
        }
        want
    };
    sampler.set(x.len(), &p, false);
    sampler.seed(1);
    let n = 20_000;
    // Drafts: a likely token, an unlikely one, one the cut drops.
    for draft in [1u32, 2, 5] {
        sampler.set_draft(1, draft);
        let mut counts = vec![0usize; x.len()];
        let mut kept = 0;
        for _ in 0..n {
            plan.set_params(TokenParams::new(0, 0));
            plan.run_serial();
            let r = sampler.read();
            counts[r.ids[slot(1)] as usize] += 1;
            kept += r.accepted[slot(1)] as usize;
            assert_eq!(r.accepted[slot(1)] == 1, r.ids[slot(1)] == draft);
        }
        for i in 0..x.len() {
            let got = counts[i] as f64 / n as f64;
            assert!(
                (got - want[i]).abs() < 0.015,
                "draft {draft} token {i}: {got} vs {}",
                want[i]
            );
        }
        let k = kept as f64 / n as f64;
        assert!(
            (k - want[draft as usize]).abs() < 0.015,
            "draft {draft} kept {k}"
        );
    }
}

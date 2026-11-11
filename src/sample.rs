//! Picks the next token from the logits.
//!
//! The order follows mlx_lm's sampler: top_p and top_k cut the distribution
//! at temperature 1, then the token is drawn from what is left at the
//! requested temperature. Temperature 0 is greedy.

#[derive(Debug, Clone, Copy)]
pub struct Params {
    pub temperature: f32,
    /// 1.0 or more turns it off.
    pub top_p: f32,
    /// 0 turns it off.
    pub top_k: usize,
}

/// With top_k off, tokens more than this far below the best logit are
/// dropped before sorting. Their total probability is under 248320 * e^-30,
/// about 2e-8, so the result does not change.
const LOGIT_WINDOW: f32 = 30.0;

pub struct Sampler {
    state: u64,
}

impl Sampler {
    pub fn new(seed: u64) -> Sampler {
        Sampler { state: seed }
    }

    pub fn sample(&mut self, logits: &[f32], p: &Params) -> u32 {
        if p.temperature <= 0.0 {
            return argmax(logits);
        }
        let candidates = self.candidates(logits, p);
        let best = candidates[0].1;
        let weights: Vec<f32> = candidates
            .iter()
            .map(|&(_, l)| ((l - best) / p.temperature).exp())
            .collect();
        let mut r = self.next_f32() * weights.iter().sum::<f32>();
        for (&(id, _), w) in candidates.iter().zip(&weights) {
            r -= w;
            if r < 0.0 {
                return id;
            }
        }
        candidates.last().unwrap().0
    }

    /// The tokens that survive top_k and top_p, best first.
    fn candidates(&self, logits: &[f32], p: &Params) -> Vec<(u32, f32)> {
        let best = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut c: Vec<(u32, f32)> = logits
            .iter()
            .enumerate()
            .map(|(i, &l)| (i as u32, l))
            .filter(|&(_, l)| p.top_k > 0 || l >= best - LOGIT_WINDOW)
            .collect();
        if p.top_k > 0 && p.top_k < c.len() {
            c.select_nth_unstable_by(p.top_k - 1, |a, b| b.1.total_cmp(&a.1));
            c.truncate(p.top_k);
        }
        c.sort_unstable_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));

        if p.top_p < 1.0 {
            // Probabilities over the whole vocab, not only the candidates.
            let total: f32 = logits.iter().map(|&l| (l - best).exp()).sum();
            let mut mass = 0.0;
            let keep = c
                .iter()
                .take_while(|&&(_, l)| {
                    let before = mass;
                    mass += (l - best).exp() / total;
                    before < p.top_p
                })
                .count();
            c.truncate(keep.max(1));
        }
        c
    }

    /// splitmix64, top 24 bits as a float in [0, 1).
    fn next_f32(&mut self) -> f32 {
        self.state = self.state.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^= z >> 31;
        (z >> 40) as f32 / (1u64 << 24) as f32
    }
}

pub fn argmax(logits: &[f32]) -> u32 {
    let mut best = 0;
    for (i, &l) in logits.iter().enumerate() {
        if l > logits[best] {
            best = i;
        }
    }
    best as u32
}

#[cfg(test)]
mod tests {
    use super::*;

    const LOGITS: [f32; 5] = [1.0, 3.0, 2.0, 0.0, 3.0];

    fn params(temperature: f32, top_p: f32, top_k: usize) -> Params {
        Params {
            temperature,
            top_p,
            top_k,
        }
    }

    fn counts(p: &Params, n: usize) -> [usize; 5] {
        let mut s = Sampler::new(7);
        let mut counts = [0; 5];
        for _ in 0..n {
            counts[s.sample(&LOGITS, p) as usize] += 1;
        }
        counts
    }

    #[test]
    fn greedy_takes_first_best() {
        assert_eq!(Sampler::new(1).sample(&LOGITS, &params(0.0, 1.0, 0)), 1);
    }

    #[test]
    fn top_k_keeps_best_tokens() {
        let c = counts(&params(1.0, 1.0, 2), 2000);
        assert_eq!((c[0], c[2], c[3]), (0, 0, 0));
        assert!(c[1] > 800 && c[4] > 800);
    }

    #[test]
    fn top_p_cuts_tail() {
        // Probabilities are about 0.39, 0.39, 0.14, 0.05, 0.02 (ids 1, 4, 2, 0, 3).
        // 0.8 needs the first three.
        let c = counts(&params(1.0, 0.8, 0), 3000);
        assert_eq!((c[0], c[3]), (0, 0));
        assert!(c[2] > 200);
    }

    #[test]
    fn matches_softmax_frequencies() {
        let n = 200_000;
        let c = counts(&params(0.5, 1.0, 0), n);
        let w: Vec<f64> = LOGITS.iter().map(|&l| (l as f64 / 0.5).exp()).collect();
        let total: f64 = w.iter().sum();
        for i in 0..5 {
            let want = w[i] / total;
            let got = c[i] as f64 / n as f64;
            assert!((got - want).abs() < 0.01, "token {i}: {got} vs {want}");
        }
    }
}

//! Small f32 building blocks shared by the layers.

/// `y = W x` where `w` is row-major with shape `[out, x.len()]`.
pub fn matvec(w: &[f32], x: &[f32]) -> Vec<f32> {
    assert_eq!(
        w.len() % x.len(),
        0,
        "matvec: weight size does not match input"
    );
    w.chunks_exact(x.len()).map(|row| dot(row, x)).collect()
}

pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// RMSNorm as in `mx.fast.rms_norm`: `x / sqrt(mean(x^2) + eps) * w`.
/// The MLX checkpoint already stores the norm weights with the `+1` folded in.
pub fn rms_norm(x: &mut [f32], w: &[f32], eps: f32) {
    assert_eq!(x.len(), w.len());
    let scale = rms_scale(x, eps);
    for (v, g) in x.iter_mut().zip(w) {
        *v *= scale * g;
    }
}

/// RMSNorm without a weight.
pub fn rms_norm_plain(x: &mut [f32], eps: f32) {
    let scale = rms_scale(x, eps);
    x.iter_mut().for_each(|v| *v *= scale);
}

fn rms_scale(x: &[f32], eps: f32) -> f32 {
    let mean = x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32;
    1.0 / (mean + eps).sqrt()
}

pub fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

pub fn silu(x: f32) -> f32 {
    x * sigmoid(x)
}

pub fn softmax(x: &mut [f32]) {
    let max = x.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0;
    for v in x.iter_mut() {
        *v = (*v - max).exp();
        sum += *v;
    }
    x.iter_mut().for_each(|v| *v /= sum);
}

/// Rotary embedding on the first `dims` values of one head, in the
/// non-interleaved layout (`traditional=False` in MLX): value `i` pairs with
/// value `i + dims/2`. The rest of the head is left as it is.
pub fn rope(head: &mut [f32], dims: usize, pos: usize, theta: f32) {
    let half = dims / 2;
    for i in 0..half {
        let freq = (theta as f64).powf(-(i as f64) / half as f64);
        let angle = pos as f64 * freq;
        let (sin, cos) = (angle.sin() as f32, angle.cos() as f32);
        let (a, b) = (head[i], head[i + half]);
        head[i] = a * cos - b * sin;
        head[i + half] = a * sin + b * cos;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cpu::testdata::{Fixture, assert_close};

    #[test]
    fn matches_mlx() {
        let f = Fixture::load("ops.json");
        let x = f.vec("x");

        let mut y = x.clone();
        rms_norm(&mut y, &f.vec("w"), 1e-6);
        assert_close("rms_norm", &y, &f.vec("rms_norm"), 1e-5);

        let y: Vec<f32> = x.iter().map(|&v| silu(v)).collect();
        assert_close("silu", &y, &f.vec("silu"), 1e-6);

        let mut y = x.clone();
        softmax(&mut y);
        assert_close("softmax", &y, &f.vec("softmax"), 1e-6);
    }

    #[test]
    fn rope_matches_mlx() {
        let f = Fixture::load("ops.json");
        let input = f.vec("rope_in");
        let positions = f.usizes("rope_positions");
        for (pos, want) in positions.iter().zip(f.vecs("rope_out")) {
            let mut got = input.clone();
            for head in got.as_chunks_mut::<16>().0 {
                rope(head, 4, *pos, 1e7);
            }
            assert_close(&format!("rope at {pos}"), &got, &want, 1e-4);
        }
    }
}

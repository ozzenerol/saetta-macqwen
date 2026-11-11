//! Reference model on the CPU in f32. Slow and simple; the GPU code is
//! checked against it.

pub mod attention;
pub mod deltanet;
pub mod ops;

#[cfg(test)]
mod testdata;

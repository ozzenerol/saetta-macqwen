//! Loads the MLX reference data in `src/cpu/testdata/`, written by `gen.py`.

use serde_json::Value;

pub struct Fixture(Value);

impl Fixture {
    pub fn load(name: &str) -> Fixture {
        let path = format!("{}/src/cpu/testdata/{name}", env!("CARGO_MANIFEST_DIR"));
        let text = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{path}: {e}"));
        Fixture(serde_json::from_str(&text).unwrap())
    }

    fn get(&self, key: &str) -> &Value {
        self.0
            .get(key)
            .unwrap_or_else(|| panic!("fixture has no {key}"))
    }

    pub fn usize(&self, key: &str) -> usize {
        self.get(key).as_u64().unwrap() as usize
    }

    pub fn f32(&self, key: &str) -> f32 {
        self.get(key).as_f64().unwrap() as f32
    }

    pub fn usizes(&self, key: &str) -> Vec<usize> {
        let a = self.get(key).as_array().unwrap();
        a.iter().map(|v| v.as_u64().unwrap() as usize).collect()
    }

    pub fn vec(&self, key: &str) -> Vec<f32> {
        to_vec(self.get(key))
    }

    pub fn vecs(&self, key: &str) -> Vec<Vec<f32>> {
        self.get(key)
            .as_array()
            .unwrap()
            .iter()
            .map(to_vec)
            .collect()
    }
}

fn to_vec(v: &Value) -> Vec<f32> {
    let a = v.as_array().unwrap();
    a.iter().map(|x| x.as_f64().unwrap() as f32).collect()
}

/// Fails when any value differs by more than `tol * max(1, |want|)`.
pub fn assert_close(what: &str, got: &[f32], want: &[f32], tol: f32) {
    assert_eq!(got.len(), want.len(), "{what}: length");
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        let limit = tol * w.abs().max(1.0);
        assert!((g - w).abs() <= limit, "{what}[{i}]: got {g}, want {w}");
    }
}

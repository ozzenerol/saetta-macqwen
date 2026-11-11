//! Checks the tokenizer against ids from the MLX tokenizer, written by
//! tools/tokenizer_ref.py. Skipped when the model is not on this machine.

use saetta::tokenizer::Tokenizer;
use serde_json::Value;
use std::path::Path;

#[test]
fn matches_mlx_tokenizer() {
    let Some(home) = std::env::var_os("HOME") else {
        return;
    };
    let dir = Path::new(&home).join("llm/mlx-models/Qwen3.6-35B-A3B-5bit");
    if !dir.exists() {
        eprintln!("skipped: no model at {}", dir.display());
        return;
    }
    let t = Tokenizer::load(&dir).unwrap();
    let data: Value = serde_json::from_str(include_str!("data/tokenizer_cases.json")).unwrap();
    let cases = data["cases"].as_array().unwrap();
    let mut failed = 0;
    for case in cases {
        let text = case["text"].as_str().unwrap();
        let want: Vec<u32> = case["ids"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as u32)
            .collect();
        let got = t.encode(text);
        if got != want {
            eprintln!("encode {text:?}\n  got  {got:?}\n  want {want:?}");
            failed += 1;
        }
        let decoded = t.decode(&want);
        if decoded != case["decoded"].as_str().unwrap() {
            eprintln!("decode {want:?}: got {decoded:?}");
            failed += 1;
        }
    }
    assert_eq!(failed, 0, "{failed} of {} cases differ", cases.len());
    assert_eq!(t.token_id("<|im_end|>"), Some(248046));
    assert!(t.is_special(248046));
}

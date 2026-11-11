//! Checks the chat prompt against MLX's apply_chat_template, with cases
//! written by tools/chat_ref.py.

use saetta::chat::{Message, Options, render};
use serde_json::Value;

fn cases() -> Vec<Value> {
    let data: Value = serde_json::from_str(include_str!("data/chat_cases.json")).unwrap();
    data["cases"].as_array().unwrap().clone()
}

fn messages(v: &Value) -> Vec<Message> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|m| Message::from_json(m).unwrap())
        .collect()
}

fn tools(case: &Value) -> Vec<Value> {
    case["tools"].as_array().cloned().unwrap_or_default()
}

#[test]
fn matches_mlx_chat_template() {
    let cases = cases();
    let mut failed = 0;
    for (i, case) in cases.iter().enumerate() {
        let kwargs = &case["kwargs"];
        let opts = Options {
            add_generation_prompt: case["add_generation_prompt"] == true,
            enable_thinking: kwargs["enable_thinking"] != false,
            preserve_thinking: kwargs["preserve_thinking"] == true,
        };
        let got = render(&messages(&case["messages"]), &tools(case), &opts);
        let want = match &case["prompt"] {
            Value::String(p) => Ok(p.clone()),
            _ => Err(case["error"].as_str().unwrap().to_string()),
        };
        if got != want {
            eprintln!("case {i}:\n--- got\n{got:?}\n--- want\n{want:?}");
            failed += 1;
        }
    }
    assert_eq!(failed, 0, "{failed} of {} cases differ", cases.len());
}

/// qh reuses the cached prompt of the last request, so with
/// preserve_thinking each request must extend the one before it. A request
/// never ends between two tool results, so that split is not checked.
#[test]
fn prompt_only_grows() {
    let case = &cases()[10];
    let all = messages(&case["messages"]);
    let tools = tools(case);
    let no_think: Vec<Message> = all
        .iter()
        .map(|m| Message {
            reasoning_content: m.reasoning_content.as_ref().map(|_| String::new()),
            ..m.clone()
        })
        .collect();

    for (all, enable_thinking) in [(&all, true), (&no_think, false)] {
        let opts = |add_generation_prompt| Options {
            add_generation_prompt,
            enable_thinking,
            preserve_thinking: true,
        };
        for n in 2..all.len() {
            if all[n].role == "tool" && all[n - 1].role == "tool" {
                continue;
            }
            let longer = render(&all[..=n], &tools, &opts(false)).unwrap();
            let shorter = render(&all[..n], &tools, &opts(false)).unwrap();
            assert!(longer.starts_with(&shorter), "{n} messages");
            if all[n].role == "assistant" {
                let request = render(&all[..n], &tools, &opts(true)).unwrap();
                assert!(
                    longer.starts_with(&request),
                    "{n} messages, generation prompt"
                );
            }
        }
    }
}

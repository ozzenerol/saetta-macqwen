//! The Qwen3.6 chat prompt, written out by hand from `chat_template.jinja`.
//!
//! `render` gives the same bytes as `tokenizer.apply_chat_template` in MLX
//! for text conversations. Images and videos are not supported.

use serde_json::{Map, Value};
use std::fmt::Write;

const TOOLS_HEADER: &str = "# Tools\n\nYou have access to the following functions:\n\n<tools>";

const TOOLS_FOOTER: &str = "\n</tools>\n\nIf you choose to call a function ONLY reply in the following format with NO suffix:\n\n<tool_call>\n<function=example_function_name>\n<parameter=example_parameter_1>\nvalue_1\n</parameter>\n<parameter=example_parameter_2>\nThis is the value for the second parameter\nthat can span\nmultiple lines\n</parameter>\n</function>\n</tool_call>\n\n<IMPORTANT>\nReminder:\n- Function calls MUST follow the specified format: an inner <function=...></function> block must be nested within <tool_call></tool_call> XML tags\n- Required parameters MUST be specified\n- You may provide optional reasoning for your function call in natural language BEFORE the function call, but NOT after\n- If there is no function call available, answer the question like normal with your current knowledge and do not tell the user about function calls\n</IMPORTANT>";

#[derive(Debug, Clone, Default, PartialEq)]
pub struct Message {
    pub role: String,
    pub content: String,
    /// `None` when the request had no `reasoning_content` string.
    pub reasoning_content: Option<String>,
    pub tool_calls: Vec<ToolCall>,
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct ToolCall {
    pub name: String,
    /// Argument names and values in request order.
    pub arguments: Option<Map<String, Value>>,
}

#[derive(Debug, Clone, Copy)]
pub struct Options {
    pub add_generation_prompt: bool,
    pub enable_thinking: bool,
    pub preserve_thinking: bool,
}

impl Default for Options {
    fn default() -> Options {
        Options {
            add_generation_prompt: true,
            enable_thinking: true,
            preserve_thinking: false,
        }
    }
}

impl Message {
    /// Reads an OpenAI chat message the way mlx_lm.server prepares it:
    /// `null` content is empty, a list of text parts is joined, and tool call
    /// arguments given as a JSON string are parsed.
    pub fn from_json(v: &Value) -> Result<Message, String> {
        let role = v["role"].as_str().ok_or("message has no role")?.to_string();
        let content = match &v["content"] {
            Value::Null => String::new(),
            Value::String(s) => s.clone(),
            Value::Array(parts) => parts
                .iter()
                .map(|p| match (&p["type"], &p["text"]) {
                    (Value::String(t), Value::String(text)) if t == "text" => Ok(text.as_str()),
                    _ => Err("Only 'text' content type is supported."),
                })
                .collect::<Result<String, _>>()?,
            _ => return Err("Unexpected content type.".into()),
        };
        let reasoning_content = v["reasoning_content"].as_str().map(str::to_string);
        let mut tool_calls = Vec::new();
        if let Some(calls) = v["tool_calls"].as_array() {
            for call in calls {
                tool_calls.push(ToolCall::from_json(call)?);
            }
        }
        Ok(Message {
            role,
            content,
            reasoning_content,
            tool_calls,
        })
    }
}

impl ToolCall {
    fn from_json(v: &Value) -> Result<ToolCall, String> {
        let f = if v["function"].is_object() {
            &v["function"]
        } else {
            v
        };
        let name = f["name"]
            .as_str()
            .ok_or("tool call has no name")?
            .to_string();
        let arguments = match &f["arguments"] {
            Value::Null => None,
            Value::String(s) if s.is_empty() => None,
            Value::String(s) => match serde_json::from_str(s) {
                Ok(Value::Object(map)) => Some(map),
                _ => return Err(format!("arguments of {name} are not a JSON object")),
            },
            Value::Object(map) => Some(map.clone()),
            _ => return Err(format!("arguments of {name} are not an object")),
        };
        Ok(ToolCall { name, arguments })
    }
}

/// Renders the prompt. The errors are the template's own messages.
pub fn render(messages: &[Message], tools: &[Value], opts: &Options) -> Result<String, String> {
    if messages.is_empty() {
        return Err("No messages provided.".into());
    }
    let mut out = String::new();
    let system = messages.first().filter(|m| m.role == "system");
    if !tools.is_empty() {
        out += "<|im_start|>system\n";
        out += TOOLS_HEADER;
        for tool in tools {
            out.push('\n');
            out += &python_json(tool);
        }
        out += TOOLS_FOOTER;
        if let Some(m) = system {
            let content = py_strip(&m.content);
            if !content.is_empty() {
                out += "\n\n";
                out += content;
            }
        }
        out += "<|im_end|>\n";
    } else if let Some(m) = system {
        out += "<|im_start|>system\n";
        out += py_strip(&m.content);
        out += "<|im_end|>\n";
    }

    let last_query = last_query_index(messages).ok_or("No user query found in messages.")?;

    for (i, m) in messages.iter().enumerate() {
        let content = py_strip(&m.content);
        match m.role.as_str() {
            "system" if i == 0 => {}
            "system" => return Err("System message must be at the beginning.".into()),
            "user" => {
                write!(out, "<|im_start|>user\n{content}<|im_end|>\n").unwrap();
            }
            "assistant" => {
                let show_thinking = opts.preserve_thinking || i > last_query;
                render_assistant(&mut out, m, content, show_thinking);
            }
            "tool" => {
                let prev_is_tool = i > 0 && messages[i - 1].role == "tool";
                let next_is_tool = messages.get(i + 1).is_some_and(|n| n.role == "tool");
                if i > 0 && !prev_is_tool {
                    out += "<|im_start|>user";
                }
                write!(out, "\n<tool_response>\n{content}\n</tool_response>").unwrap();
                if !next_is_tool {
                    out += "<|im_end|>\n";
                }
            }
            _ => return Err("Unexpected message role.".into()),
        }
    }

    if opts.add_generation_prompt {
        out += "<|im_start|>assistant\n";
        out += if opts.enable_thinking {
            "<think>\n"
        } else {
            "<think>\n\n</think>\n\n"
        };
    }
    Ok(out)
}

/// The last user message that is not a wrapped tool response. Thinking is
/// shown only for assistant turns after it, unless preserve_thinking is on.
fn last_query_index(messages: &[Message]) -> Option<usize> {
    messages.iter().rposition(|m| {
        let content = py_strip(&m.content);
        m.role == "user"
            && !(content.starts_with("<tool_response>") && content.ends_with("</tool_response>"))
    })
}

fn render_assistant(out: &mut String, m: &Message, content: &str, show_thinking: bool) {
    let (reasoning, content) = match &m.reasoning_content {
        Some(r) => (r.as_str(), content),
        None => match content.split_once("</think>") {
            Some((before, _)) => {
                let after = content.rsplit("</think>").next().unwrap();
                let before = before.trim_end_matches('\n');
                let reasoning = before.rsplit("<think>").next().unwrap();
                (
                    reasoning.trim_start_matches('\n'),
                    after.trim_start_matches('\n'),
                )
            }
            None => ("", content),
        },
    };
    let reasoning = py_strip(reasoning);

    if show_thinking {
        write!(
            out,
            "<|im_start|>assistant\n<think>\n{reasoning}\n</think>\n\n{content}"
        )
        .unwrap();
    } else {
        write!(out, "<|im_start|>assistant\n{content}").unwrap();
    }

    for (i, call) in m.tool_calls.iter().enumerate() {
        if i > 0 {
            out.push('\n');
        } else if !py_strip(content).is_empty() {
            *out += "\n\n";
        }
        write!(out, "<tool_call>\n<function={}>\n", call.name).unwrap();
        for (name, value) in call.arguments.iter().flatten() {
            let value = match value {
                Value::String(s) => s.clone(),
                other => python_json(other),
            };
            write!(out, "<parameter={name}>\n{value}\n</parameter>\n").unwrap();
        }
        *out += "</function>\n</tool_call>";
    }
    *out += "<|im_end|>\n";
}

/// Python's `str.strip()`: it also strips the separators U+001C to U+001F,
/// which Rust does not count as whitespace.
pub fn py_strip(s: &str) -> &str {
    s.trim_matches(|c: char| c.is_whitespace() || ('\u{1c}'..='\u{1f}').contains(&c))
}

/// JSON as Python's `json.dumps(v, ensure_ascii=False)` writes it: `", "` and
/// `": "` separators, keys in their order, and Python's float format.
pub fn python_json(v: &Value) -> String {
    let mut out = String::new();
    write_python_json(&mut out, v);
    out
}

fn write_python_json(out: &mut String, v: &Value) {
    match v {
        Value::Null => *out += "null",
        Value::Bool(b) => *out += if *b { "true" } else { "false" },
        Value::Number(n) => match n.as_f64() {
            Some(f) if n.is_f64() => *out += &python_float(f),
            _ => *out += &n.to_string(),
        },
        // serde_json escapes the same characters as Python: quote, backslash
        // and control characters, with lowercase \u00xx.
        Value::String(s) => *out += &serde_json::to_string(s).unwrap(),
        Value::Array(items) => {
            out.push('[');
            for (i, item) in items.iter().enumerate() {
                if i > 0 {
                    *out += ", ";
                }
                write_python_json(out, item);
            }
            out.push(']');
        }
        Value::Object(map) => {
            out.push('{');
            for (i, (key, value)) in map.iter().enumerate() {
                if i > 0 {
                    *out += ", ";
                }
                *out += &serde_json::to_string(key).unwrap();
                *out += ": ";
                write_python_json(out, value);
            }
            out.push('}');
        }
    }
}

/// Python's `repr(float)`: the shortest digits that round-trip, written
/// plainly for exponents -4 to 15 and as `1e+16` or `2.5e-07` otherwise.
fn python_float(f: f64) -> String {
    let sci = format!("{f:e}");
    let (mantissa, exp) = sci.split_once('e').unwrap();
    let exp: i32 = exp.parse().unwrap();
    if (-4..16).contains(&exp) {
        let plain = format!("{f}");
        if plain.contains('.') {
            plain
        } else {
            plain + ".0"
        }
    } else {
        let sign = if exp < 0 { '-' } else { '+' };
        format!("{mantissa}e{sign}{:02}", exp.abs())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn floats_like_python() {
        let cases = [
            (0.0, "0.0"),
            (0.25, "0.25"),
            (120.5, "120.5"),
            (-3.0, "-3.0"),
            (0.001, "0.001"),
            (0.0001, "0.0001"),
            (1e-5, "1e-05"),
            (1e-7, "1e-07"),
            (2.5e-7, "2.5e-07"),
            (1e15, "1000000000000000.0"),
            (1e16, "1e+16"),
            (1.5e16, "1.5e+16"),
            (1e21, "1e+21"),
            (1e300, "1e+300"),
        ];
        for (f, want) in cases {
            assert_eq!(python_float(f), want, "{f}");
        }
    }

    #[test]
    fn json_like_python() {
        let v: Value =
            serde_json::from_str(r#"{"z": [1, 2.0, "a\"\n\u0001é"], "a": {"b": null, "c": true}}"#)
                .unwrap();
        assert_eq!(
            python_json(&v),
            r#"{"z": [1, 2.0, "a\"\n\u0001é"], "a": {"b": null, "c": true}}"#
        );
    }

    #[test]
    fn strips_like_python() {
        assert_eq!(py_strip("\u{1c} a b\u{3000}\n"), "a b");
    }
}

//! Parses the XML tool calls Qwen writes between `<tool_call>` and
//! `</tool_call>`:
//!
//! ```text
//! <function=read>
//! <parameter=path>
//! main.go
//! </parameter>
//! </function>
//! ```
//!
//! Values are converted by the parameter type in the request's tool schema,
//! the same way as mlx_lm's `qwen3_coder` parser.

use serde_json::{Map, Number, Value};

#[derive(Debug, Clone, PartialEq)]
pub struct ParsedCall {
    pub name: String,
    pub arguments: Map<String, Value>,
}

/// `text` is what the model wrote inside one `<tool_call>` block.
pub fn parse(text: &str, tools: &[Value]) -> Option<ParsedCall> {
    let start = text.find("<function=")? + "<function=".len();
    let end = text.rfind("</function>")?;
    let body = text.get(start..end)?;
    let (name, mut rest) = split_name(body)?;
    let schema = properties(name, tools);

    let mut arguments = Map::new();
    while let Some(p) = rest.find("<parameter=") {
        rest = &rest[p + "<parameter=".len()..];
        let Some(close) = rest.find("</parameter>") else {
            break;
        };
        let inner = &rest[..close];
        rest = &rest[close + "</parameter>".len()..];
        let Some((key, value)) = split_name(inner) else {
            continue;
        };
        let value = value.strip_prefix('\n').unwrap_or(value);
        let value = value.strip_suffix('\n').unwrap_or(value);
        arguments.insert(
            key.to_string(),
            convert(value, schema.and_then(|s| s.get(key))),
        );
    }
    Some(ParsedCall {
        name: name.to_string(),
        arguments,
    })
}

/// Splits `name>rest` after optional leading whitespace. The name has no
/// whitespace or angle brackets, as in mlx_lm's `\s*([^\s<>]+)>?`.
fn split_name(s: &str) -> Option<(&str, &str)> {
    let s = s.trim_start();
    let end = s
        .find(|c: char| c.is_whitespace() || c == '<' || c == '>')
        .unwrap_or(s.len());
    if end == 0 {
        return None;
    }
    let rest = &s[end..];
    Some((&s[..end], rest.strip_prefix('>').unwrap_or(rest)))
}

fn properties<'a>(name: &str, tools: &'a [Value]) -> Option<&'a Map<String, Value>> {
    tools
        .iter()
        .map(|t| &t["function"])
        .find(|f| f["name"] == name)
        .and_then(|f| f["parameters"]["properties"].as_object())
}

fn convert(value: &str, schema: Option<&Value>) -> Value {
    let text = || Value::String(value.to_string());
    if value.eq_ignore_ascii_case("null") {
        return Value::Null;
    }
    let Some(schema) = schema.filter(|s| s.is_object()) else {
        return text();
    };
    let Some(kind) = schema.get("type") else {
        return match serde_json::from_str(value) {
            Ok(v @ (Value::Object(_) | Value::Array(_))) => v,
            _ => text(),
        };
    };
    let kind = match kind {
        Value::String(s) => s.trim().to_lowercase(),
        other => other.to_string().to_lowercase(),
    };

    if ["string", "str", "text", "varchar", "char", "enum"].contains(&kind.as_str()) {
        return text();
    }
    let is_int = ["int", "uint", "long", "short", "unsigned"]
        .iter()
        .any(|p| kind.starts_with(p));
    if is_int || kind.starts_with("num") || kind.starts_with("float") {
        let Ok(f) = value.trim().parse::<f64>() else {
            return text();
        };
        if f.fract() == 0.0 && f.abs() < 9.0e15 {
            return Value::Number((f as i64).into());
        }
        if is_int {
            return text();
        }
        return Number::from_f64(f).map_or_else(text, Value::Number);
    }
    if ["boolean", "bool", "binary"].contains(&kind.as_str()) {
        return Value::Bool(value.eq_ignore_ascii_case("true"));
    }
    // object, array and anything else: JSON if it parses, else the text.
    serde_json::from_str(value).unwrap_or_else(|_| text())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn tools() -> Vec<Value> {
        vec![
            json!({"type": "function", "function": {"name": "bash", "parameters": {"type": "object", "properties": {
                "command": {"type": "string"},
                "timeout": {"type": "integer"},
                "ratio": {"type": "number"},
                "background": {"type": "boolean"},
                "env": {"type": "object"},
                "any": {"description": "no type"}
            }}}}),
        ]
    }

    #[test]
    fn parses_typed_parameters() {
        let text = "\n<function=bash>\n<parameter=command>\nls -la\necho 1\n</parameter>\n\
            <parameter=timeout>\n30\n</parameter>\n<parameter=ratio>\n0.5\n</parameter>\n\
            <parameter=background>\nTrue\n</parameter>\n<parameter=env>\n{\"A\": 1}\n</parameter>\n\
            <parameter=any>\n[1, 2]\n</parameter>\n<parameter=other>\n42\n</parameter>\n</function>\n";
        let call = parse(text, &tools()).unwrap();
        assert_eq!(call.name, "bash");
        assert_eq!(
            Value::Object(call.arguments),
            json!({"command": "ls -la\necho 1", "timeout": 30, "ratio": 0.5, "background": true,
                   "env": {"A": 1}, "any": [1, 2], "other": "42"})
        );
    }

    #[test]
    fn numbers_like_mlx() {
        let int = json!({"type": "integer"});
        let num = json!({"type": "number"});
        assert_eq!(convert("3.0", Some(&num)), json!(3));
        assert_eq!(convert("3.5", Some(&int)), json!("3.5"));
        assert_eq!(convert("abc", Some(&num)), json!("abc"));
        assert_eq!(convert("NULL", Some(&num)), Value::Null);
    }

    #[test]
    fn rejects_missing_function() {
        assert_eq!(parse("just text", &[]), None);
        assert_eq!(parse("<function=x>\n", &[]), None);
        let call = parse("<function=ls>\n</function>", &[]).unwrap();
        assert!(call.arguments.is_empty());
    }
}

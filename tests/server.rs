//! Runs the server on a local port with a fake model that writes a fixed
//! token script, and talks to it over HTTP.

use saetta::cache::{Cached, Restore};
use saetta::chat::{self, Message, Options};
use saetta::sample::Params;
use saetta::server::{Model, Server, Settings};
use saetta::tokenizer::Tokenizer;
use serde_json::{Value, json};
use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::{Arc, Mutex};
use std::thread;

const IM_START: u32 = 300;
const IM_END: u32 = 301;
const THINK: u32 = 302;
const THINK_END: u32 = 303;
const TOOL: u32 = 304;
const TOOL_END: u32 = 305;
const VOCAB: usize = 307;

/// A tokenizer with one token per byte and no merges, plus the chat tokens.
fn tokenizer() -> Tokenizer {
    let printable = |b: u8| matches!(b, b'!'..=b'~' | 0xa1..=0xac | 0xae..=0xff);
    let mut next = 0x100;
    let mut vocab = serde_json::Map::new();
    for b in 0..=255u8 {
        let c = if printable(b) {
            char::from(b)
        } else {
            next += 1;
            char::from_u32(next - 1).unwrap()
        };
        vocab.insert(c.to_string(), b.into());
    }
    let added: Vec<Value> = [
        "<|im_start|>",
        "<|im_end|>",
        "<think>",
        "</think>",
        "<tool_call>",
        "</tool_call>",
        "<|endoftext|>",
    ]
    .iter()
    .enumerate()
    .map(|(i, t)| json!({"id": 300 + i, "content": t, "special": true}))
    .collect();
    let file = json!({
        "added_tokens": added,
        "normalizer": {"type": "NFC"},
        "pre_tokenizer": {"type": "Sequence", "pretokenizers": [
            {"type": "Split", "pattern": {"Regex": "\\s+|\\S+"}, "behavior": "Isolated", "invert": false},
            {"type": "ByteLevel", "add_prefix_space": false, "trim_offsets": false, "use_regex": false}]},
        "model": {"type": "BPE", "vocab": vocab, "merges": []}
    });
    Tokenizer::parse(&file.to_string()).unwrap()
}

/// Bytes of the text as token ids, with special ids in between.
fn script(parts: &[Result<&str, u32>]) -> Vec<u32> {
    parts
        .iter()
        .flat_map(|p| match p {
            Ok(text) => text.bytes().map(u32::from).collect(),
            Err(id) => vec![*id],
        })
        .collect()
}

#[derive(Default)]
struct Log {
    prefills: Vec<usize>,
    resets: usize,
    restores: Vec<usize>,
}

/// Writes `script` after every prefill, one token per step.
struct Fake {
    script: Vec<u32>,
    at: usize,
    len: usize,
    log: Arc<Mutex<Log>>,
}

impl Fake {
    fn logits(&self) -> Vec<f32> {
        let mut l = vec![0.0; VOCAB];
        l[self.script.get(self.at).copied().unwrap_or(IM_END) as usize] = 10.0;
        l
    }
}

impl Model for Fake {
    fn prefill(&mut self, tokens: &[u32]) -> Vec<f32> {
        self.log.lock().unwrap().prefills.push(tokens.len());
        self.len += tokens.len();
        self.at = 0;
        self.logits()
    }
    fn step(&mut self, _token: u32) -> Vec<f32> {
        self.len += 1;
        self.at += 1;
        self.logits()
    }
    fn truncate(&mut self, len: usize) -> usize {
        self.len = len;
        len
    }
    fn reset(&mut self) {
        self.len = 0;
        self.log.lock().unwrap().resets += 1;
    }
}

/// The fake's only state is its length, which is enough to see where the
/// cache goes back to.
impl Restore for Fake {
    type State = usize;
    fn save(&self) -> usize {
        self.len
    }
    fn restore(&mut self, state: &usize, len: usize) {
        assert_eq!(*state, len);
        self.len = len;
        self.log.lock().unwrap().restores.push(len);
    }
}

fn fake(script: Vec<u32>) -> (Fake, Arc<Mutex<Log>>) {
    let log = Arc::new(Mutex::new(Log::default()));
    let model = Fake {
        script,
        at: 0,
        len: 0,
        log: Arc::clone(&log),
    };
    (model, log)
}

fn start(script: Vec<u32>) -> (String, Arc<Mutex<Log>>) {
    let (model, log) = fake(script);
    (serve(model), log)
}

/// The fake behind the prompt cache, with a snapshot every `every` tokens.
fn start_cached(script: Vec<u32>, every: usize) -> (String, Arc<Mutex<Log>>) {
    let (model, log) = fake(script);
    (serve(Cached::new(model, 16, every)), log)
}

fn serve(model: impl Model + 'static) -> String {
    let settings = Settings {
        model_name: "fake".into(),
        sampling: Params {
            temperature: 0.0,
            top_p: 1.0,
            top_k: 0,
        },
        max_tokens: 1000,
        eos_token_ids: vec![IM_END, 306],
    };
    let server = Arc::new(Server::new(model, tokenizer(), settings).unwrap());
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap().to_string();
    thread::spawn(move || server.serve(listener));
    addr
}

/// Sends one request and returns the status code and body.
fn http(addr: &str, method: &str, path: &str, body: &str) -> (u16, String) {
    let mut s = TcpStream::connect(addr).unwrap();
    write!(
        s,
        "{method} {path} HTTP/1.1\r\nHost: x\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{body}",
        body.len()
    )
    .unwrap();
    let mut response = String::new();
    s.read_to_string(&mut response).unwrap();
    let (head, body) = response.split_once("\r\n\r\n").unwrap();
    let status = head.split(' ').nth(1).unwrap().parse().unwrap();
    (status, body.to_string())
}

fn post(addr: &str, body: Value) -> (u16, String) {
    http(addr, "POST", "/v1/chat/completions", &body.to_string())
}

fn events(body: &str) -> Vec<Value> {
    let lines: Vec<&str> = body
        .lines()
        .filter_map(|l| l.strip_prefix("data: "))
        .collect();
    assert_eq!(lines.last(), Some(&"[DONE]"));
    lines[..lines.len() - 1]
        .iter()
        .map(|l| serde_json::from_str(l).unwrap())
        .collect()
}

fn read_tool() -> Value {
    json!([{"type": "function", "function": {"name": "read", "parameters": {"type": "object",
        "properties": {"path": {"type": "string"}, "limit": {"type": "integer"}}}}}])
}

#[test]
fn lists_model() {
    let (addr, _) = start(vec![]);
    let (status, body) = http(&addr, "GET", "/v1/models", "");
    assert_eq!(status, 200);
    let v: Value = serde_json::from_str(&body).unwrap();
    assert_eq!(v["data"][0]["id"], "fake");
}

#[test]
fn answers_without_stream() {
    // "é" is two bytes, so the answer ends with a character split over two tokens.
    let (addr, _) = start(script(&[
        Ok("Let me see.\n"),
        Err(THINK_END),
        Ok("\n\nHello é"),
    ]));
    let (status, body) = post(
        &addr,
        json!({"messages": [{"role": "user", "content": "hi"}],
               "chat_template_kwargs": {"enable_thinking": true}}),
    );
    assert_eq!(status, 200, "{body}");
    let v: Value = serde_json::from_str(&body).unwrap();
    let message = &v["choices"][0]["message"];
    assert_eq!(message["reasoning_content"], "Let me see.\n");
    assert_eq!(message["content"], "Hello é");
    assert_eq!(v["choices"][0]["finish_reason"], "stop");
    // 22 text tokens, </think> and the end token
    assert_eq!(v["usage"]["completion_tokens"], 24);
    assert_eq!(v["usage"]["prompt_tokens_details"]["cached_tokens"], 0);
}

#[test]
fn streams_tool_call() {
    let call = "\n<function=read>\n<parameter=path>\nmain.go\n</parameter>\n<parameter=limit>\n20\n</parameter>\n</function>\n";
    let (addr, _) = start(script(&[
        Err(THINK_END),
        Ok("\n\nReading."),
        Err(TOOL),
        Ok(call),
        Err(TOOL_END),
    ]));
    let (status, body) = post(
        &addr,
        json!({"messages": [{"role": "user", "content": "read main.go"}], "tools": read_tool(),
               "stream": true, "stream_options": {"include_usage": true}}),
    );
    assert_eq!(status, 200);
    let events = events(&body);
    let mut content = String::new();
    let mut reasoning = String::new();
    let mut calls = Vec::new();
    for e in &events {
        let delta = &e["choices"][0]["delta"];
        content += delta["content"].as_str().unwrap_or("");
        reasoning += delta["reasoning_content"].as_str().unwrap_or("");
        if let Some(c) = delta["tool_calls"].as_array() {
            calls.extend(c.iter().cloned());
        }
    }
    assert_eq!(content, "Reading.");
    assert_eq!(reasoning, "");
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0]["index"], 0);
    assert_eq!(calls[0]["function"]["name"], "read");
    let args: Value =
        serde_json::from_str(calls[0]["function"]["arguments"].as_str().unwrap()).unwrap();
    assert_eq!(args, json!({"path": "main.go", "limit": 20}));

    let finish = &events[events.len() - 2];
    assert_eq!(finish["choices"][0]["finish_reason"], "tool_calls");
    let usage = &events[events.len() - 1];
    assert_eq!(usage["choices"], json!([]));
    assert!(usage["usage"]["prompt_tokens"].as_u64().unwrap() > 0);
}

#[test]
fn reuses_cached_prefix() {
    let (addr, log) = start(script(&[Ok("Four.")]));
    let kwargs = json!({"enable_thinking": false, "preserve_thinking": true});
    let first = json!([{"role": "user", "content": "2+2?"}]);
    let (_, body) = post(
        &addr,
        json!({"messages": first, "chat_template_kwargs": kwargs}),
    );
    let v: Value = serde_json::from_str(&body).unwrap();
    assert_eq!(v["choices"][0]["message"]["content"], "Four.");
    let prompt_tokens = v["usage"]["prompt_tokens"].as_u64().unwrap();

    let second = json!([
        {"role": "user", "content": "2+2?"},
        {"role": "assistant", "content": "Four."},
        {"role": "user", "content": "3+3?"}
    ]);
    let (_, body) = post(
        &addr,
        json!({"messages": second, "chat_template_kwargs": kwargs}),
    );
    let v: Value = serde_json::from_str(&body).unwrap();
    let usage = &v["usage"];
    let cached = usage["prompt_tokens_details"]["cached_tokens"]
        .as_u64()
        .unwrap();
    assert!(
        cached > prompt_tokens,
        "cached {cached}, first prompt {prompt_tokens}"
    );

    let log = log.lock().unwrap();
    assert_eq!(log.resets, 0);
    assert_eq!(log.prefills[0] as u64, prompt_tokens);
    assert_eq!(
        log.prefills[1] as u64,
        usage["prompt_tokens"].as_u64().unwrap() - cached
    );
}

#[test]
fn stops_at_max_tokens() {
    let (addr, _) = start(script(&[Ok("aaaaaaaaaaaaaaaaaaaa")]));
    let (_, body) = post(
        &addr,
        json!({"messages": [{"role": "user", "content": "x"}], "max_tokens": 5,
               "chat_template_kwargs": {"enable_thinking": false}}),
    );
    let v: Value = serde_json::from_str(&body).unwrap();
    assert_eq!(v["choices"][0]["message"]["content"], "aaaaa");
    assert_eq!(v["choices"][0]["finish_reason"], "length");
}

#[test]
fn rejects_bad_requests() {
    let (addr, _) = start(vec![]);
    let (status, body) = post(
        &addr,
        json!({"messages": [{"role": "system", "content": "only"}]}),
    );
    assert_eq!(status, 400);
    assert!(body.contains("No user query found"));
    let (status, _) = http(&addr, "POST", "/v1/chat/completions", "{not json");
    assert_eq!(status, 400);
    let (status, _) = http(&addr, "GET", "/nope", "");
    assert_eq!(status, 404);
}

#[test]
fn unused_ids_are_quiet() {
    // Keeps the constants honest: the prompt tokens of a chat include these.
    let t = tokenizer();
    assert_eq!(t.token_id("<|im_start|>"), Some(IM_START));
    assert_eq!(t.token_id("<think>"), Some(THINK));
}

#[test]
fn keeps_cut_tool_call_as_text() {
    let (addr, _) = start(script(&[
        Err(TOOL),
        Ok("\n<function=read>\n<parameter=path>\nmain"),
    ]));
    let (_, body) = post(
        &addr,
        json!({"messages": [{"role": "user", "content": "x"}], "tools": read_tool(), "max_tokens": 10,
               "chat_template_kwargs": {"enable_thinking": false}}),
    );
    let v: Value = serde_json::from_str(&body).unwrap();
    let choice = &v["choices"][0];
    assert_eq!(choice["message"]["content"], "<tool_call>\n<functio");
    assert_eq!(choice["message"]["tool_calls"], Value::Null);
    assert_eq!(choice["finish_reason"], "length");
}

fn usage(body: &str) -> (u64, u64) {
    let v: Value = serde_json::from_str(body).unwrap();
    let u = &v["usage"];
    (
        u["prompt_tokens"].as_u64().unwrap(),
        u["prompt_tokens_details"]["cached_tokens"]
            .as_u64()
            .unwrap(),
    )
}

/// A qh session of 10 tool steps: every request extends the last one, so the
/// model reads each prompt token once and never goes back.
#[test]
fn tool_steps_read_each_token_once() {
    let call = "\n<function=read>\n<parameter=path>\nmain.go\n</parameter>\n</function>\n";
    let (addr, log) = start_cached(
        script(&[
            Ok("\n"),
            Err(THINK_END),
            Ok("\n\nReading.\n\n"),
            Err(TOOL),
            Ok(call),
            Err(TOOL_END),
        ]),
        64,
    );
    let mut messages = vec![json!({"role": "user", "content": "Read main.go ten times."})];
    let mut state_len = 0;
    for step in 0..10 {
        let (_, body) = post(
            &addr,
            json!({"messages": messages, "tools": read_tool(),
                   "chat_template_kwargs": {"enable_thinking": true, "preserve_thinking": true}}),
        );
        let (prompt, cached) = usage(&body);
        assert_eq!(cached, state_len, "step {step}");
        let v: Value = serde_json::from_str(&body).unwrap();
        let message = &v["choices"][0]["message"];
        let completion = v["usage"]["completion_tokens"].as_u64().unwrap();
        // The end token is sampled but never run.
        state_len = prompt + completion - 1;
        messages.push(json!({"role": "assistant", "content": message["content"],
                             "tool_calls": message["tool_calls"]}));
        messages.push(json!({"role": "tool", "content": format!("file contents {step}")}));
    }
    let log = log.lock().unwrap();
    assert!(log.restores.is_empty());
    assert_eq!(log.resets, 0);
}

/// An edit near the end of a long prompt goes back to the last snapshot
/// before it, not to the start.
#[test]
fn edit_goes_back_to_a_snapshot() {
    let (addr, log) = start_cached(script(&[Ok("ok")]), 64);
    let long = "x ".repeat(200);
    let ask = |last: &str| {
        json!({"messages": [{"role": "user", "content": long}, {"role": "assistant", "content": "seen"},
                            {"role": "user", "content": last}],
               "chat_template_kwargs": {"enable_thinking": false}})
    };
    let tokens = |request: &Value| {
        let messages: Vec<Message> = request["messages"]
            .as_array()
            .unwrap()
            .iter()
            .map(|m| Message::from_json(m).unwrap())
            .collect();
        let options = Options {
            enable_thinking: false,
            ..Options::default()
        };
        tokenizer().encode(&chat::render(&messages, &[], &options).unwrap())
    };
    let (first, second) = (
        tokens(&ask("first question")),
        tokens(&ask("second question")),
    );
    let common = first
        .iter()
        .zip(&second)
        .take_while(|(a, b)| a == b)
        .count();
    assert!(
        common % 64 > 10,
        "the test needs a snapshot well before the edit"
    );

    post(&addr, ask("first question"));
    let (_, body) = post(&addr, ask("second question"));
    let (_, cached) = usage(&body);
    assert_eq!(cached as usize, common / 64 * 64);
    let log = log.lock().unwrap();
    assert_eq!(log.restores, vec![cached as usize]);
    assert_eq!(log.resets, 0);

    drop(log);
    // The same prompt again reruns one token.
    let (_, body) = post(&addr, ask("second question"));
    let (prompt, cached) = usage(&body);
    assert_eq!(cached, prompt - 1);
}

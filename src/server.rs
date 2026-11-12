//! OpenAI-compatible HTTP server on `std::net`.
//!
//! - `GET /v1/models`
//! - `POST /v1/chat/completions`, with `stream` true or false
//!
//! One request generates at a time; the others wait on the engine lock.
//! Connections are closed after each response, so there is no keep-alive.

use crate::chat::{self, Message, Options};
use crate::sample::{Params, Sampler};
use crate::speculative::{Decider, Sampled};
use crate::tokenizer::Tokenizer;
use crate::toolcall;
use serde_json::{Value, json};
use std::io::{self, BufRead, BufReader, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{SystemTime, UNIX_EPOCH};

/// The engine behind the server. Calls append to the model's state.
pub trait Model: Send {
    /// Runs the tokens and returns the logits after the last one.
    fn prefill(&mut self, tokens: &[u32]) -> Vec<f32>;
    /// `prefill` that asks `alive` between pieces of a long input and stops
    /// when it says false. The error is the number of tokens the state holds
    /// then, all of them from the start of `tokens`' prompt.
    fn prefill_while(
        &mut self,
        tokens: &[u32],
        _alive: &mut dyn FnMut() -> bool,
    ) -> Result<Vec<f32>, usize> {
        Ok(self.prefill(tokens))
    }
    /// Runs one token and returns the logits after it.
    fn step(&mut self, token: u32) -> Vec<f32>;
    /// Goes back to at most `len` tokens and returns how many the state
    /// holds now. At 0 the server resets the model.
    fn truncate(&mut self, len: usize) -> usize;
    fn reset(&mut self);
    /// Goes to the longest state that `prompt` starts with and returns its
    /// length; `held` is what the state holds now. The default cuts the
    /// state back to the common prefix; a prompt cache may find more.
    fn resume(&mut self, held: &[u32], prompt: &[u32]) -> usize {
        let common = held.iter().zip(prompt).take_while(|(a, b)| a == b).count();
        if common == held.len() {
            return common;
        }
        let keep = self.truncate(common);
        if keep == 0 {
            self.reset();
        }
        keep
    }
    /// Runs `pending` and returns the tokens decided after it: tokens the
    /// state already holds, and the next token, which it does not. A model
    /// with a drafter returns several tokens per call.
    fn speculate(&mut self, pending: u32, decide: &mut dyn Decider) -> (Vec<u32>, u32) {
        let logits = self.step(pending);
        (Vec::new(), decide.pick(&logits))
    }
    /// Tokens the state can hold: the prompt and the answer together.
    fn capacity(&self) -> usize {
        usize::MAX
    }
}

pub struct Settings {
    pub model_name: String,
    pub sampling: Params,
    pub max_tokens: usize,
    pub eos_token_ids: Vec<u32>,
    /// Always tokenize the rendered prompt, as MLX does. When false, a
    /// conversation that continues from our last answer keeps the tokens the
    /// model generated instead of the re-rendered ones (see `splice`).
    pub exact_history: bool,
}

pub struct Server<M: Model> {
    engine: Mutex<Engine<M>>,
    tokenizer: Tokenizer,
    settings: Settings,
    special: Special,
    requests: AtomicU64,
    /// `Model::capacity`, read once.
    capacity: usize,
}

struct Engine<M> {
    model: M,
    /// Tokens the model state holds now.
    tokens: Vec<u32>,
    /// The last request, when its answer is still at the end of `tokens`.
    last: Option<Turn>,
}

/// What a request asked and what we answered.
struct Turn {
    messages: Vec<Message>,
    prompt_text: String,
    answer: Message,
}

/// Ids of the tokens that switch the output between thinking, text and tool calls.
struct Special {
    think_end: u32,
    tool_start: u32,
    tool_end: u32,
}

impl<M: Model + 'static> Server<M> {
    pub fn new(model: M, tokenizer: Tokenizer, settings: Settings) -> Result<Server<M>, String> {
        let id = |t: &str| tokenizer.token_id(t).ok_or(format!("tokenizer has no {t}"));
        let special = Special {
            think_end: id("</think>")?,
            tool_start: id("<tool_call>")?,
            tool_end: id("</tool_call>")?,
        };
        let capacity = model.capacity();
        Ok(Server {
            capacity,
            engine: Mutex::new(Engine {
                model,
                tokens: Vec::new(),
                last: None,
            }),
            tokenizer,
            settings,
            special,
            requests: AtomicU64::new(0),
        })
    }

    /// Accepts connections until the listener fails, one thread each.
    pub fn serve(self: Arc<Self>, listener: TcpListener) -> io::Result<()> {
        for stream in listener.incoming() {
            let stream = stream?;
            let server = Arc::clone(&self);
            thread::spawn(move || {
                if let Err(e) = server.handle(stream) {
                    eprintln!("connection: {e}");
                }
            });
        }
        Ok(())
    }

    fn handle(&self, stream: TcpStream) -> io::Result<()> {
        stream.set_nodelay(true)?;
        let mut reader = BufReader::new(stream.try_clone()?);
        let mut out = stream;
        let request = match read_request(&mut reader) {
            Ok(r) => r,
            Err(e) => return send_error(&mut out, 400, &e),
        };
        match (request.method.as_str(), request.path.as_str()) {
            ("GET", "/v1/models") => send_json(&mut out, 200, &self.models()),
            ("POST", "/v1/chat/completions") => match serde_json::from_slice(&request.body) {
                Ok(body) => self.chat(&mut out, &body),
                Err(e) => send_error(&mut out, 400, &format!("bad JSON: {e}")),
            },
            _ => send_error(&mut out, 404, "not found"),
        }
    }

    fn models(&self) -> Value {
        json!({"object": "list", "data": [{
            "id": self.settings.model_name, "object": "model", "created": now(), "owned_by": "saetta"
        }]})
    }

    fn chat(&self, out: &mut TcpStream, body: &Value) -> io::Result<()> {
        let request = match self.parse_chat(body) {
            Ok(r) => r,
            Err(e) => return send_error(out, 400, &e),
        };
        if request.prompt.len() >= self.capacity {
            let e = format!(
                "the prompt is {} tokens, and the model's context holds {}",
                request.prompt.len(),
                self.capacity
            );
            return send_error(out, 400, &e);
        }
        let id = request.id.clone();
        let model = self.settings.model_name.as_str();
        let created = now();
        let chunk = |delta: Value, finish: Value| {
            json!({"id": id, "object": "chat.completion.chunk", "created": created, "model": model,
                   "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]})
        };

        if !request.stream {
            let mut reasoning = String::new();
            let mut content = String::new();
            let mut calls = Vec::new();
            let watch = out.try_clone()?;
            let done = self.generate(
                &request,
                || client_alive(&watch),
                |event| {
                    match event {
                        Event::Reasoning(s) => reasoning += &s,
                        Event::Content(s) => content += &s,
                        Event::ToolCall(c) => calls.push(c),
                    }
                    true
                },
            );
            let mut message = json!({"role": "assistant", "content": content});
            if !reasoning.is_empty() {
                message["reasoning_content"] = reasoning.into();
            }
            if !calls.is_empty() {
                message["tool_calls"] = calls.iter().map(ToolCallOut::to_json).collect();
            }
            let response = json!({"id": id, "object": "chat.completion", "created": created, "model": model,
                "choices": [{"index": 0, "message": message, "finish_reason": done.finish_reason()}],
                "usage": done.usage()});
            return send_json(out, 200, &response);
        }

        out.write_all(
            b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\nConnection: close\r\n\r\n",
        )?;
        send_event(
            out,
            &chunk(json!({"role": "assistant", "content": ""}), Value::Null),
        )?;
        let mut calls = 0;
        let mut failed = None;
        let watch = out.try_clone()?;
        let done = self.generate(
            &request,
            || client_alive(&watch),
            |event| {
                let delta = match event {
                    Event::Reasoning(s) => json!({"reasoning_content": s}),
                    Event::Content(s) => json!({"content": s}),
                    Event::ToolCall(c) => {
                        let mut call = c.to_json();
                        call["index"] = calls.into();
                        calls += 1;
                        json!({"tool_calls": [call]})
                    }
                };
                let sent = send_event(out, &chunk(delta, Value::Null));
                let ok = sent.is_ok();
                failed = sent.err();
                ok
            },
        );
        if let Some(e) = failed {
            return Err(e);
        }
        send_event(out, &chunk(json!({}), done.finish_reason().into()))?;
        if request.include_usage {
            let usage = json!({"id": id, "object": "chat.completion.chunk", "created": created,
                               "model": model, "choices": [], "usage": done.usage()});
            send_event(out, &usage)?;
        }
        out.write_all(b"data: [DONE]\n\n")?;
        out.flush()
    }

    fn parse_chat(&self, body: &Value) -> Result<ChatRequest, String> {
        let messages = body["messages"]
            .as_array()
            .ok_or("messages must be a list")?
            .iter()
            .map(Message::from_json)
            .collect::<Result<Vec<_>, _>>()?;
        let tools = body["tools"].as_array().cloned().unwrap_or_default();
        let kwargs = &body["chat_template_kwargs"];
        let options = Options {
            add_generation_prompt: true,
            enable_thinking: kwargs["enable_thinking"] != false,
            preserve_thinking: kwargs["preserve_thinking"] == true,
        };
        let prompt = chat::render(&messages, &tools, &options)?;
        let thinking = prompt.ends_with("<think>\n");

        let defaults = self.settings.sampling;
        let max_tokens = body["max_completion_tokens"]
            .as_u64()
            .or(body["max_tokens"].as_u64())
            .map_or(self.settings.max_tokens, |n| n as usize);
        Ok(ChatRequest {
            id: format!(
                "chatcmpl-{}-{}",
                now(),
                self.requests.fetch_add(1, Ordering::Relaxed)
            ),
            prompt: self.tokenizer.encode(&prompt),
            prompt_text: prompt,
            messages,
            options,
            thinking,
            tools,
            stream: body["stream"] == true,
            include_usage: body["stream_options"]["include_usage"] == true,
            max_tokens,
            sampling: Params {
                temperature: body["temperature"]
                    .as_f64()
                    .map_or(defaults.temperature, |v| v as f32),
                top_p: body["top_p"].as_f64().map_or(defaults.top_p, |v| v as f32),
                top_k: body["top_k"]
                    .as_u64()
                    .map_or(defaults.top_k, |v| v as usize),
            },
            seed: body["seed"].as_u64().unwrap_or_else(|| now() ^ 0x5eed),
        })
    }

    /// Runs one completion. `emit` returns false to stop early, for example
    /// when the client has gone. `alive` is asked every `ALIVE_EVERY` tokens,
    /// since a tool call sends nothing until it closes.
    fn generate(
        &self,
        r: &ChatRequest,
        mut alive: impl FnMut() -> bool,
        mut emit: impl FnMut(Event) -> bool,
    ) -> Done {
        let mut engine = self.engine.lock().unwrap_or_else(|e| e.into_inner());
        let _trace = crate::memtrace::Span::new("request");
        let held = engine.tokens.clone();
        let spliced = match engine.last.take() {
            Some(last) if !self.settings.exact_history => self.splice(&last, &held, r),
            _ => None,
        };
        let how = if spliced.is_some() {
            "splice"
        } else {
            "render"
        };
        let prompt = spliced.unwrap_or_else(|| r.prompt.clone());
        if prompt.len() >= self.capacity {
            // Only a spliced prompt can get here; `chat` checks the rendered one.
            return Done {
                prompt_tokens: prompt.len(),
                cached_tokens: 0,
                completion_tokens: 0,
                hit_limit: true,
                tool_calls: 0,
            };
        }
        let started = engine.start(&prompt, &mut alive);
        let cached = match started {
            Ok((_, cached)) | Err((_, cached)) => cached,
        };
        eprintln!(
            "{}",
            reuse_report(how, &held, &prompt, cached, &self.tokenizer)
        );
        let logits = match started {
            Ok((logits, _)) => logits,
            Err((kept, _)) => {
                eprintln!(
                    "client left during the prefill, at {kept} of {} tokens",
                    prompt.len()
                );
                return Done {
                    prompt_tokens: prompt.len(),
                    cached_tokens: cached,
                    completion_tokens: 0,
                    hit_limit: false,
                    tool_calls: 0,
                };
            }
        };
        let mut decide = Sampled {
            params: r.sampling,
            sampler: Sampler::new(r.seed),
        };
        let mut output = Output::new(r.thinking, &r.tools, &r.id);
        let mut answer = Message {
            role: "assistant".into(),
            ..Message::default()
        };
        let mut emit = |event: Event| {
            record(&mut answer, &event);
            emit(event)
        };
        let mut done = Done {
            prompt_tokens: prompt.len(),
            cached_tokens: cached,
            completion_tokens: 0,
            hit_limit: false,
            tool_calls: 0,
        };

        // Emits one decided token; false when generation ends here.
        let mut take = |token: u32, done: &mut Done| -> Take {
            // The token taken now is run next, so it must fit in the state.
            let full = prompt.len() + done.completion_tokens >= self.capacity;
            if done.completion_tokens == r.max_tokens || full {
                done.hit_limit = true;
                return Take::Stop;
            }
            done.completion_tokens += 1;
            if self.settings.eos_token_ids.contains(&token) {
                return Take::Stop;
            }
            let events = output.push(token, &self.tokenizer, &self.special);
            if !self.emit_all(events, &mut emit, done) {
                return Take::Gone;
            }
            if done.completion_tokens.is_multiple_of(ALIVE_EVERY) && !alive() {
                return Take::Gone;
            }
            Take::Go
        };

        let mut pending = decide.pick(&logits);
        'outer: loop {
            match take(pending, &mut done) {
                Take::Go => {}
                Take::Stop => break,
                Take::Gone => return done,
            }
            let (accepted, next) = engine.model.speculate(pending, &mut decide);
            engine.tokens.push(pending);
            engine.tokens.extend(&accepted);
            for t in accepted {
                match take(t, &mut done) {
                    Take::Go => {}
                    Take::Stop => break 'outer,
                    Take::Gone => return done,
                }
            }
            pending = next;
        }
        let events = output.finish();
        if self.emit_all(events, &mut emit, &mut done) {
            engine.last = Some(Turn {
                messages: r.messages.clone(),
                prompt_text: r.prompt_text.clone(),
                answer,
            });
        }
        done
    }

    /// The prompt as the tokens the state holds (our last prompt and the
    /// tokens the model generated for its answer) plus the rest of the new
    /// render, when the new conversation is the last one followed by exactly
    /// that answer. The model then sees its own tokens, not the template's
    /// re-render of them, and nothing has to run again. `None` when anything
    /// differs.
    fn splice(&self, last: &Turn, held: &[u32], r: &ChatRequest) -> Option<Vec<u32>> {
        const END: &str = "<|im_end|>\n";
        const EMPTY_THINK: &str = "<think>\n\n</think>\n\n";
        let n = last.messages.len();
        if r.messages.len() <= n
            || r.messages[..n] != last.messages[..]
            || !same_answer(&r.messages[n], &last.answer)
        {
            return None;
        }
        let options = Options {
            add_generation_prompt: false,
            ..r.options
        };
        let upto = chat::render(&r.messages[..=n], &r.tools, &options).ok()?;
        if !upto.starts_with(&last.prompt_text) || !upto.ends_with(END) {
            return None;
        }
        // With thinking off, the answer starts with an empty think block that
        // the template keeps for the last answer only, so the new render has
        // the answer without it. The state keeps the tokens the model saw.
        let upto = if r.prompt_text.starts_with(&upto) {
            upto
        } else {
            let base = last.prompt_text.strip_suffix(EMPTY_THINK)?;
            let upto = format!("{base}{}", &upto[last.prompt_text.len()..]);
            if !r.prompt_text.starts_with(&upto) {
                return None;
            }
            upto
        };
        let rest = &r.prompt_text[upto.len() - END.len()..];
        let mut tokens = held.to_vec();
        tokens.extend(self.tokenizer.encode(rest));
        Some(tokens)
    }

    fn emit_all(
        &self,
        events: Vec<Event>,
        emit: &mut impl FnMut(Event) -> bool,
        done: &mut Done,
    ) -> bool {
        for event in events {
            if matches!(event, Event::ToolCall(_)) {
                done.tool_calls += 1;
            }
            if !emit(event) {
                return false;
            }
        }
        true
    }
}

enum Take {
    Go,
    Stop,
    Gone,
}

impl<M: Model> Engine<M> {
    /// Reuses the longest state the model has for the prompt, runs the rest,
    /// and returns the logits with the number of reused tokens. When the
    /// client leaves first, the error has the tokens the state holds instead
    /// of the logits.
    fn start(
        &mut self,
        prompt: &[u32],
        alive: &mut dyn FnMut() -> bool,
    ) -> Result<(Vec<f32>, usize), (usize, usize)> {
        // At least one token has to run to get logits.
        let usable = &prompt[..prompt.len().saturating_sub(1)];
        let keep = self.model.resume(&self.tokens, usable);
        match self.model.prefill_while(&prompt[keep..], alive) {
            Ok(logits) => {
                self.tokens = prompt.to_vec();
                Ok((logits, keep))
            }
            Err(held) => {
                self.tokens = prompt[..held].to_vec();
                Err((held, keep))
            }
        }
    }
}

/// Adds an output event to the answer as the client will send it back.
fn record(answer: &mut Message, event: &Event) {
    match event {
        Event::Reasoning(s) => {
            *answer.reasoning_content.get_or_insert_with(String::new) += s;
        }
        Event::Content(s) => answer.content += s,
        Event::ToolCall(c) => answer.tool_calls.push(chat::ToolCall {
            name: c.name.clone(),
            arguments: serde_json::from_str(&c.arguments).ok(),
        }),
    }
}

/// Whether a message from the client is the answer we sent: the same
/// content, reasoning and tool calls, up to the whitespace the template trims.
fn same_answer(m: &Message, answer: &Message) -> bool {
    let text = |s: &Option<String>| chat::py_strip(s.as_deref().unwrap_or("")).to_string();
    // Without reasoning_content the template takes the reasoning out of a
    // content with </think>; our answers never have one.
    let reasoning_in_content = m.reasoning_content.is_none() && m.content.contains("</think>");
    m.role == "assistant"
        && !reasoning_in_content
        && chat::py_strip(&m.content) == chat::py_strip(&answer.content)
        && text(&m.reasoning_content) == text(&answer.reasoning_content)
        && m.tool_calls == answer.tool_calls
}

/// One log line per request: whether the prompt was spliced or rendered, how
/// much of it the state already had, and where it first differs from the
/// tokens the state holds.
fn reuse_report(
    how: &str,
    held: &[u32],
    prompt: &[u32],
    reused: usize,
    tokenizer: &Tokenizer,
) -> String {
    let common = held.iter().zip(prompt).take_while(|(a, b)| a == b).count();
    let mut line = format!(
        "request: {how}, prompt {}, held {}, common {common}, reused {reused}",
        prompt.len(),
        held.len()
    );
    if common < held.len() && common < prompt.len() {
        let text = |tokens: &[u32]| {
            let from = common.saturating_sub(8);
            let to = (common + 16).min(tokens.len());
            format!(
                "{:?} | {:?}",
                tokenizer.decode(&tokens[from..common]),
                tokenizer.decode(&tokens[common..to])
            )
        };
        line += &format!("; held {} ; prompt {}", text(held), text(prompt));
    }
    line
}

struct ChatRequest {
    id: String,
    messages: Vec<Message>,
    options: Options,
    prompt_text: String,
    prompt: Vec<u32>,
    thinking: bool,
    tools: Vec<Value>,
    stream: bool,
    include_usage: bool,
    max_tokens: usize,
    sampling: Params,
    seed: u64,
}

struct Done {
    prompt_tokens: usize,
    cached_tokens: usize,
    completion_tokens: usize,
    hit_limit: bool,
    tool_calls: usize,
}

impl Done {
    fn finish_reason(&self) -> &'static str {
        if self.tool_calls > 0 {
            "tool_calls"
        } else if self.hit_limit {
            "length"
        } else {
            "stop"
        }
    }

    fn usage(&self) -> Value {
        json!({"prompt_tokens": self.prompt_tokens, "completion_tokens": self.completion_tokens,
               "total_tokens": self.prompt_tokens + self.completion_tokens,
               "prompt_tokens_details": {"cached_tokens": self.cached_tokens}})
    }
}

#[derive(Debug, PartialEq)]
enum Event {
    Reasoning(String),
    Content(String),
    ToolCall(ToolCallOut),
}

#[derive(Debug, PartialEq)]
struct ToolCallOut {
    id: String,
    name: String,
    /// JSON text, as OpenAI sends it.
    arguments: String,
}

impl ToolCallOut {
    fn to_json(&self) -> Value {
        json!({"id": self.id, "type": "function",
               "function": {"name": self.name, "arguments": self.arguments}})
    }
}

#[derive(Clone, Copy, PartialEq)]
enum Part {
    Reasoning,
    Content,
    ToolCall,
}

/// Turns generated tokens into text events. Bytes are held until they form
/// whole UTF-8 characters, and leading newlines of the answer are dropped.
struct Output<'a> {
    part: Part,
    pending: Vec<u8>,
    tool_text: String,
    /// The tool call started inside the reasoning, before `</think>`.
    tool_in_reasoning: bool,
    content_started: bool,
    tools: &'a [Value],
    id: &'a str,
    calls: usize,
}

impl<'a> Output<'a> {
    fn new(thinking: bool, tools: &'a [Value], id: &'a str) -> Output<'a> {
        Output {
            part: if thinking {
                Part::Reasoning
            } else {
                Part::Content
            },
            pending: Vec::new(),
            tool_text: String::new(),
            tool_in_reasoning: false,
            content_started: false,
            tools,
            id,
            calls: 0,
        }
    }

    fn push(&mut self, token: u32, tokenizer: &Tokenizer, special: &Special) -> Vec<Event> {
        let mut events = Vec::new();
        if token == special.think_end && self.part == Part::Reasoning {
            self.flush(&mut events, true);
            self.part = Part::Content;
        } else if token == special.tool_start && self.part != Part::ToolCall {
            self.flush(&mut events, true);
            self.tool_in_reasoning = self.part == Part::Reasoning;
            self.part = Part::ToolCall;
            self.tool_text.clear();
        } else if token == special.tool_end && self.part == Part::ToolCall {
            self.flush(&mut events, true);
            let event = self.tool_call(true);
            // mlx_lm leaves the reasoning for a tool call too, but a call
            // that is only written about in the reasoning stays there.
            self.part = match event {
                Event::Reasoning(_) => Part::Reasoning,
                _ => Part::Content,
            };
            events.push(event);
        } else if !tokenizer.is_special(token) {
            self.pending.extend_from_slice(tokenizer.token_bytes(token));
            self.flush(&mut events, false);
        }
        events
    }

    fn finish(&mut self) -> Vec<Event> {
        let mut events = Vec::new();
        self.flush(&mut events, true);
        if self.part == Part::ToolCall {
            // The model stopped inside a tool call; parse what is there.
            events.push(self.tool_call(false));
        }
        events
    }

    /// A tool call event, or the raw text as written when it does not parse.
    /// Inside the reasoning, the call must also name one of the request's
    /// tools, else its text goes back to the reasoning: the model sometimes
    /// writes `<tool_call>` there while thinking about the format.
    fn tool_call(&mut self, closed: bool) -> Event {
        let text = std::mem::take(&mut self.tool_text);
        let known = |name: &str| self.tools.iter().any(|t| t["function"]["name"] == name);
        let call = toolcall::parse(&text, self.tools)
            .filter(|c| !self.tool_in_reasoning || known(&c.name));
        let end = if closed { "</tool_call>" } else { "" };
        match call {
            None if self.tool_in_reasoning => Event::Reasoning(format!("<tool_call>{text}{end}")),
            None => Event::Content(format!("<tool_call>{text}{end}")),
            Some(call) => {
                self.calls += 1;
                Event::ToolCall(ToolCallOut {
                    id: format!(
                        "call_{}_{}",
                        self.id.trim_start_matches("chatcmpl-"),
                        self.calls
                    ),
                    name: call.name,
                    arguments: chat::python_json(&Value::Object(call.arguments)),
                })
            }
        }
    }

    /// Emits the complete UTF-8 characters held so far. With `all`, invalid
    /// or cut bytes go out too, as U+FFFD.
    fn flush(&mut self, events: &mut Vec<Event>, all: bool) {
        let n = if all {
            self.pending.len()
        } else {
            match std::str::from_utf8(&self.pending) {
                Ok(_) => self.pending.len(),
                Err(e) if e.error_len().is_none() => e.valid_up_to(),
                Err(_) => self.pending.len(),
            }
        };
        if n == 0 {
            return;
        }
        let text = String::from_utf8_lossy(&self.pending[..n]).into_owned();
        self.pending.drain(..n);
        match self.part {
            Part::Reasoning => events.push(Event::Reasoning(text)),
            Part::ToolCall => self.tool_text += &text,
            Part::Content => {
                let text = if self.content_started {
                    text.as_str()
                } else {
                    text.trim_start_matches('\n')
                };
                if !text.is_empty() {
                    self.content_started = true;
                    events.push(Event::Content(text.to_string()));
                }
            }
        }
    }
}

struct Request {
    method: String,
    path: String,
    body: Vec<u8>,
}

const MAX_BODY: usize = 64 << 20;

/// Tokens between checks that the client is still there.
const ALIVE_EVERY: usize = 16;

/// False once the client has closed the connection. Clients do not send
/// anything while they wait for the answer, so a peek that would block
/// means the connection is open.
fn client_alive(stream: &TcpStream) -> bool {
    if stream.set_nonblocking(true).is_err() {
        return true;
    }
    let alive = match stream.peek(&mut [0]) {
        Ok(n) => n > 0,
        Err(e) => e.kind() == io::ErrorKind::WouldBlock,
    };
    let _ = stream.set_nonblocking(false);
    alive
}

fn read_request(reader: &mut impl BufRead) -> Result<Request, String> {
    let mut line = String::new();
    reader.read_line(&mut line).map_err(|e| e.to_string())?;
    let mut parts = line.split_whitespace();
    let (Some(method), Some(target)) = (parts.next(), parts.next()) else {
        return Err("bad request line".into());
    };
    let path = target.split('?').next().unwrap_or("").to_string();
    let method = method.to_string();

    let mut length = 0;
    loop {
        line.clear();
        if reader.read_line(&mut line).map_err(|e| e.to_string())? == 0 {
            return Err("connection closed in headers".into());
        }
        let header = line.trim_end();
        if header.is_empty() {
            break;
        }
        if let Some((name, value)) = header.split_once(':')
            && name.eq_ignore_ascii_case("content-length")
        {
            length = value.trim().parse().map_err(|_| "bad Content-Length")?;
        }
    }
    if length > MAX_BODY {
        return Err("body too large".into());
    }
    let mut body = vec![0; length];
    reader.read_exact(&mut body).map_err(|e| e.to_string())?;
    Ok(Request { method, path, body })
}

fn send_json(out: &mut impl Write, status: u16, body: &Value) -> io::Result<()> {
    let body = body.to_string();
    let reason = match status {
        200 => "OK",
        400 => "Bad Request",
        404 => "Not Found",
        _ => "Error",
    };
    write!(
        out,
        "HTTP/1.1 {status} {reason}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
        body.len()
    )?;
    out.flush()
}

fn send_error(out: &mut impl Write, status: u16, message: &str) -> io::Result<()> {
    send_json(
        out,
        status,
        &json!({"error": {"message": message, "type": "invalid_request_error"}}),
    )
}

fn send_event(out: &mut impl Write, data: &Value) -> io::Result<()> {
    write!(out, "data: {data}\n\n")?;
    out.flush()
}

fn now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_secs())
}

use saetta::cache::{self, Cached, Restore};
use saetta::config::{Config, LayerType};
use saetta::cpu::deltanet::DeltaNetState;
use saetta::cpu::model::{LayerState, Model};
use saetta::sample::Params;
use saetta::server::{self, Server, Settings};
use saetta::tokenizer::Tokenizer;
use saetta::weights::Weights;
#[cfg(target_os = "macos")]
use std::cell::RefCell;
use std::env;
use std::error::Error;
use std::net::TcpListener;
use std::path::Path;
use std::process::ExitCode;
use std::sync::Arc;
use std::time::Instant;

const USAGE: &str = "usage: saetta gpu-check | saetta info <model dir> | saetta convert <mlx dir> <out dir> | saetta convert-mtp <hf mtp file> <model dir> | saetta serve <model dir> [--gpu] [--draft K] [--port N] [--exact-history] | saetta eval <model dir> <ref dir> [--gpu] [--long N] [--decode] [--no-ppl] | saetta bench-qmv <model dir> | saetta bench-moe <model dir> | saetta bench-prefill <model dir> <tokens> [chunk] | saetta bench-attention | saetta bench-decode <model dir> | saetta profile-token <model dir>";

fn main() -> ExitCode {
    let args: Vec<String> = env::args().skip(1).collect();
    let args: Vec<&str> = args.iter().map(String::as_str).collect();
    match args[..] {
        ["gpu-check"] => saetta::gpu::check(),
        ["info", dir] => report(info(Path::new(dir))),
        ["convert", src, out] => report(convert(Path::new(src), Path::new(out))),
        ["convert-mtp", file, dir] => report(convert_mtp(Path::new(file), Path::new(dir))),
        ["serve", dir, ref rest @ ..] => match serve_flags(rest) {
            Ok(flags) => report(serve(Path::new(dir), flags)),
            Err(e) => {
                eprintln!("{e}\n{USAGE}");
                ExitCode::from(2)
            }
        },
        ["eval", dir, reference, ref options @ ..] => match eval_options(options) {
            Some(options) => match eval(Path::new(dir), Path::new(reference), options) {
                Ok(true) => ExitCode::SUCCESS,
                Ok(false) => ExitCode::FAILURE,
                Err(e) => {
                    eprintln!("error: {e}");
                    ExitCode::FAILURE
                }
            },
            None => {
                eprintln!("{USAGE}");
                ExitCode::from(2)
            }
        },
        #[cfg(target_os = "macos")]
        ["bench-qmv", dir] => report(saetta::qmv::bench(Path::new(dir), 5, 20)),
        #[cfg(target_os = "macos")]
        ["bench-moe", dir] => report(saetta::moe::bench(Path::new(dir), 5)),
        #[cfg(target_os = "macos")]
        ["bench-prefill", dir, tokens] => bench_prefill(dir, tokens, "1024"),
        #[cfg(target_os = "macos")]
        ["bench-prefill", dir, tokens, chunk] => bench_prefill(dir, tokens, chunk),
        #[cfg(target_os = "macos")]
        ["bench-qmm", dir, rows] => match rows.parse() {
            Ok(m) => report(saetta::prefill::bench_qmm(Path::new(dir), m, 10)),
            Err(_) => {
                eprintln!("{USAGE}");
                ExitCode::FAILURE
            }
        },
        #[cfg(target_os = "macos")]
        ["bench-decode", dir] => report(saetta::attention::bench_decode(
            Path::new(dir),
            &[40, 1024, 4096, 8192, 16384, 32768],
            5,
        )),
        #[cfg(target_os = "macos")]
        ["profile-token", dir] => report(saetta::profile::profile_token(Path::new(dir), 40, 9)),
        #[cfg(target_os = "macos")]
        ["bench-attention"] => {
            report(saetta::attention::bench(&[1024, 8192, 16384, 32768], 10, 5).map_err(Into::into))
        }
        _ => {
            eprintln!("{USAGE}");
            ExitCode::from(2)
        }
    }
}

fn report(result: Result<(), Box<dyn Error>>) -> ExitCode {
    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("error: {e}");
            ExitCode::FAILURE
        }
    }
}

fn convert(src: &Path, out: &Path) -> Result<(), Box<dyn Error>> {
    let s = saetta::convert::convert(src, out)?;
    println!(
        "wrote {} tensors, {} bytes, in {} shards to {}",
        s.tensors,
        s.bytes,
        s.shards,
        out.display()
    );
    let checked = saetta::convert::verify(src, out)?;
    println!("checked {checked} tensors: same bytes as the source, all aligned");
    Ok(())
}

fn convert_mtp(file: &Path, dir: &Path) -> Result<(), Box<dyn Error>> {
    let n = saetta::convert::convert_mtp(file, dir)?;
    println!(
        "wrote {n} MTP tensors to {}",
        dir.join(saetta::convert::MTP_FILE).display()
    );
    let err = saetta::convert::verify_mtp(file, dir)?;
    println!("checked: norms exact, all aligned, largest relative RMS error of a matrix {err:.4}");
    Ok(())
}

struct ServeFlags {
    gpu: bool,
    /// Tokens the MTP layer drafts per round; 0 turns drafting off.
    draft: usize,
    port: u16,
    exact_history: bool,
}

/// `[--gpu] [--draft K] [--port N] [--exact-history]` after `serve <dir>`.
fn serve_flags(flags: &[&str]) -> Result<ServeFlags, String> {
    let mut f = ServeFlags {
        gpu: false,
        draft: 0,
        port: 8090,
        exact_history: false,
    };
    let mut it = flags.iter();
    while let Some(&flag) = it.next() {
        match flag {
            "--gpu" => f.gpu = true,
            "--draft" => {
                let value = it.next().ok_or("--draft needs a number")?;
                f.draft = value
                    .parse()
                    .map_err(|_| format!("bad draft count {value}"))?;
                if f.draft > 7 {
                    return Err("--draft takes 0 to 7".into());
                }
            }
            "--exact-history" => f.exact_history = true,
            "--port" => {
                let value = it.next().ok_or("--port needs a number")?;
                f.port = value.parse().map_err(|_| format!("bad port {value}"))?;
            }
            _ => return Err(format!("unknown flag {flag}")),
        }
    }
    if f.draft > 0 && !f.gpu {
        return Err("--draft needs --gpu".into());
    }
    Ok(f)
}

/// Serves the model on 127.0.0.1, on the GPU with `gpu`, else on the CPU
/// (slow, but it runs the whole path).
fn serve(dir: &Path, flags: ServeFlags) -> Result<(), Box<dyn Error>> {
    let ServeFlags {
        gpu,
        draft,
        port,
        exact_history,
    } = flags;
    // The model borrows these for as long as the server runs.
    let config: &'static Config = Box::leak(Box::new(Config::load(dir)?));
    let weights: &'static Weights = Box::leak(Box::new(Weights::open(dir)?));
    let tokenizer = Tokenizer::load(dir)?;
    let settings = Settings {
        model_name: dir.display().to_string(),
        // The settings mlx_lm.server runs with on the Mac.
        sampling: Params {
            temperature: 1.0,
            top_p: 0.95,
            top_k: 20,
        },
        max_tokens: 16384,
        eos_token_ids: config.eos_token_ids.clone(),
        exact_history,
    };
    let listener = TcpListener::bind(("127.0.0.1", port))?;
    let what = format!("{} on 127.0.0.1:{port}", dir.display());
    if gpu {
        return serve_gpu(config, weights, tokenizer, settings, listener, &what, draft);
    }
    let model = Model::new(config, weights)?;
    let states = model.new_states();
    let server = Arc::new(Server::new(
        Cached::new(
            CpuModel { model, states },
            cache::DEFAULT_LIMIT,
            cache::DEFAULT_EVERY,
        ),
        tokenizer,
        settings,
    )?);
    eprintln!("serving {what} (CPU model)");
    server.serve(listener)?;
    Ok(())
}

#[cfg(not(target_os = "macos"))]
fn serve_gpu(
    _: &Config,
    _: &Weights,
    _: Tokenizer,
    _: Settings,
    _: TcpListener,
    _: &str,
    _: usize,
) -> Result<(), Box<dyn Error>> {
    Err("--gpu needs macOS".into())
}

/// Tokens of context the GPU model has room for: 40 KB of keys and values
/// per token over the 10 attention layers, 1.3 GB in all.
#[cfg(target_os = "macos")]
const GPU_CONTEXT: usize = 32768;

#[cfg(target_os = "macos")]
fn serve_gpu(
    config: &'static Config,
    weights: &'static Weights,
    tokenizer: Tokenizer,
    settings: Settings,
    listener: TcpListener,
    what: &str,
    draft: usize,
) -> Result<(), Box<dyn Error>> {
    let served = load_gpu(config, weights, draft)?;
    let model = Cached::new(served, cache::DEFAULT_LIMIT, cache::DEFAULT_EVERY);
    let server = Arc::new(Server::new(model, tokenizer, settings)?);
    let drafts = if draft > 0 {
        format!(", MTP drafts {draft} per round")
    } else {
        String::new()
    };
    eprintln!("serving {what} (GPU model, {GPU_CONTEXT} tokens of context{drafts})");
    server.serve(listener)?;
    Ok(())
}

/// The 40-layer GPU model with its prefill, snapshot pool and, with
/// `draft` > 0, MTP drafting, as served.
#[cfg(target_os = "macos")]
fn load_gpu(
    config: &'static Config,
    weights: &'static Weights,
    draft: usize,
) -> Result<GpuServed, Box<dyn Error>> {
    let gpu: &'static saetta::gpu::Gpu = Box::leak(Box::new(saetta::gpu::Gpu::new()?));
    let model =
        saetta::gpu_model::GpuModel::load(gpu, config, weights, config.num_layers(), GPU_CONTEXT)?;
    let layers = model.deltanet_states();
    let pool = saetta::gpu_cache::SnapshotPool::new(gpu, &layers, cache::DEFAULT_LIMIT + 1);
    let mut prefill =
        saetta::prefill::Prefill::new(gpu, config, weights, PREFILL_CHUNK, GPU_CONTEXT)?;
    prefill.exact = exact_prefill();
    if draft > 0 {
        prefill.enable_verify(gpu, draft + 1);
    }
    let spec = if draft > 0 {
        let mtp = saetta::gpu_mtp::GpuMtp::load(gpu, config, weights, GPU_CONTEXT)?;
        let norm = weights.tensor("language_model.model.norm.weight")?.to_f32();
        Some(Drafting {
            spec: saetta::speculative::GpuSpec::new(&model, mtp, config, draft),
            norm,
            eps: config.rms_norm_eps,
        })
    } else {
        None
    };
    Ok(GpuServed {
        model,
        prefill,
        pool,
        last_blit: RefCell::new(None),
        drafting: spec,
    })
}

#[cfg(target_os = "macos")]
struct GpuServed {
    model: saetta::gpu_model::GpuModel<'static>,
    pool: saetta::gpu_cache::SnapshotPool,
    /// The last snapshot copy, which may still run. The CPU must wait for it
    /// before it writes the states, as `reset` does.
    last_blit: RefCell<Option<metal::CommandBuffer>>,
    prefill: saetta::prefill::Prefill<'static>,
    drafting: Option<Drafting>,
}

/// Inputs shorter than this run as decode steps. Measured with bench-prefill
/// and bench-decode (2026-11-11, 3 runs): prefill takes 27 ms for 2 tokens,
/// 55 ms for 8, 179 ms for 17 and 458 ms for 64; a decode step takes 10 to
/// 12 ms up to 8k of context. So prefill wins from about 8 tokens.
#[cfg(target_os = "macos")]
const SHORT_PREFILL: usize = 8;

/// The MTP drafter, with the final norm to turn prefill rows into the
/// normed hidden states it reads.
#[cfg(target_os = "macos")]
struct Drafting {
    spec: saetta::speculative::GpuSpec<'static>,
    norm: Vec<f32>,
    eps: f32,
}

/// Prompt tokens that get an MTP cache entry after a prefill.
#[cfg(target_os = "macos")]
const MTP_WINDOW: usize = 32;

/// Tokens per prefill dispatch.
#[cfg(target_os = "macos")]
const PREFILL_CHUNK: usize = 1024;

#[cfg(target_os = "macos")]
fn exact_prefill() -> bool {
    !saetta::prefill::fast_from_env()
}

// The server uses the model from one thread at a time, behind its engine
// lock. The model is not Send only because Gpu caches pipelines in a
// RefCell, which nothing else touches.
#[cfg(target_os = "macos")]
unsafe impl Send for GpuServed {}

#[cfg(target_os = "macos")]
impl server::Model for GpuServed {
    fn prefill(&mut self, tokens: &[u32]) -> Vec<f32> {
        let m = &mut self.model;
        assert!(
            m.position() + tokens.len() <= m.capacity(),
            "prompt of {} tokens does not fit in {} of context",
            tokens.len(),
            m.capacity()
        );
        if tokens.is_empty() {
            return Vec::new();
        }
        if let Some(d) = &mut self.drafting {
            d.spec.report();
        }
        let start = Instant::now();
        let logits = if tokens.len() < SHORT_PREFILL {
            // Decode steps; with drafting, each also adds its MTP entry.
            let (&last, rest) = tokens.split_last().unwrap();
            match &mut self.drafting {
                None => {
                    rest.iter().for_each(|&t| m.run(t));
                    m.forward(last)
                }
                Some(d) => {
                    // Only the last MTP_WINDOW tokens get an MTP entry, as
                    // after a prefill.
                    let plain = tokens.len().saturating_sub(MTP_WINDOW);
                    for (i, &t) in rest.iter().enumerate() {
                        if i < plain {
                            d.spec.step_plain(m, t);
                        } else {
                            d.spec.step(m, t);
                        }
                    }
                    d.spec.step(m, last)
                }
            }
        } else {
            let pos = m.position();
            let states: Vec<_> = m.states().collect();
            let logits = match &mut self.drafting {
                None => self.prefill.run(m.gpu(), tokens, pos, &states),
                Some(d) => {
                    // The same work as `run`, keeping the hidden states of the
                    // last tokens for the MTP cache.
                    let gpu = m.gpu();
                    let n = tokens.len();
                    let x = self.prefill.embed(gpu, tokens);
                    for (l, state) in states.iter().enumerate() {
                        self.prefill.layer(gpu, l, &x, n, pos, state);
                    }
                    let logits = self.prefill.logits(gpu, &x, n);
                    let h = d.norm.len();
                    let first = n.saturating_sub(MTP_WINDOW + 1);
                    let mut rows = saetta::gpu::contents::<f32>(&x, n * h)[first * h..].to_vec();
                    for row in rows.chunks_exact_mut(h) {
                        saetta::cpu::ops::rms_norm(row, &d.norm, d.eps);
                    }
                    d.spec
                        .after_prefill(&tokens[first..], &rows, pos + first, MTP_WINDOW);
                    logits
                }
            };
            m.set_position(pos + tokens.len());
            logits
        };
        let secs = start.elapsed().as_secs_f64();
        eprintln!(
            "prefill {} tokens in {secs:.2} s ({:.1} ms/token)",
            tokens.len(),
            secs * 1e3 / tokens.len() as f64
        );
        logits
    }

    fn step(&mut self, token: u32) -> Vec<f32> {
        match &mut self.drafting {
            Some(d) => d.spec.step(&mut self.model, token),
            None => self.model.forward(token),
        }
    }

    fn speculate(
        &mut self,
        pending: u32,
        decide: &mut dyn saetta::speculative::Decider,
    ) -> (Vec<u32>, u32) {
        match &mut self.drafting {
            Some(d) => d
                .spec
                .speculate(&mut self.model, &self.prefill, pending, decide),
            None => {
                let logits = self.model.forward(pending);
                (Vec::new(), decide.pick(&logits))
            }
        }
    }

    /// `Cached` goes back through snapshots instead.
    fn truncate(&mut self, _len: usize) -> usize {
        0
    }

    fn reset(&mut self) {
        if let Some(blit) = self.last_blit.take() {
            blit.wait_until_completed();
        }
        self.model.reset();
        if let Some(d) = &mut self.drafting {
            d.spec.reset();
        }
    }

    /// A drafting round runs up to 8 tokens past the ones it keeps
    /// (`--draft` is at most 7), so they must fit too.
    fn capacity(&self) -> usize {
        self.model.capacity() - 8
    }
}

#[cfg(target_os = "macos")]
impl Restore for GpuServed {
    /// The DeltaNet states, and with drafting the last hidden state, which
    /// the MTP layer needs for its first entry after a restore.
    type State = (saetta::gpu_cache::GpuSnapshot, Vec<f32>);

    fn save(&self) -> Self::State {
        let m = &self.model;
        let (snapshot, blit) = saetta::gpu_cache::save(m.gpu(), &self.pool, &m.deltanet_states());
        *self.last_blit.borrow_mut() = Some(blit);
        let last = self
            .drafting
            .as_ref()
            .map_or(Vec::new(), |d| d.spec.last_hidden());
        (snapshot, last)
    }

    fn restore(&mut self, state: &Self::State, len: usize) {
        let m = &mut self.model;
        let blit = saetta::gpu_cache::restore(m.gpu(), &state.0, &m.deltanet_states());
        *self.last_blit.borrow_mut() = Some(blit);
        m.set_position(len);
        if let Some(d) = &mut self.drafting {
            d.spec.after_restore(len, &state.1);
        }
    }
}

struct CpuModel<'a> {
    model: Model<'a>,
    states: Vec<LayerState>,
}

impl CpuModel<'_> {
    fn run(&mut self, token: u32) -> Vec<f32> {
        self.model
            .forward(token, &mut self.states)
            .unwrap_or_else(|e| panic!("forward failed: {e}"))
    }
}

impl server::Model for CpuModel<'_> {
    fn prefill(&mut self, tokens: &[u32]) -> Vec<f32> {
        let start = Instant::now();
        let mut logits = Vec::new();
        for (i, &t) in tokens.iter().enumerate() {
            logits = self.run(t);
            if (i + 1) % 100 == 0 {
                eprintln!("prefill {}/{}", i + 1, tokens.len());
            }
        }
        let secs = start.elapsed().as_secs_f64();
        eprintln!(
            "prefill {} tokens in {secs:.1} s ({:.0} ms/token)",
            tokens.len(),
            secs * 1e3 / tokens.len().max(1) as f64
        );
        logits
    }

    fn step(&mut self, token: u32) -> Vec<f32> {
        self.run(token)
    }

    /// `Cached` goes back through snapshots instead.
    fn truncate(&mut self, _len: usize) -> usize {
        0
    }

    fn reset(&mut self) {
        self.states = self.model.new_states();
    }
}

impl Restore for CpuModel<'_> {
    type State = Vec<DeltaNetState>;

    fn save(&self) -> Self::State {
        cache::save_cpu(&self.states)
    }

    fn restore(&mut self, state: &Self::State, len: usize) {
        cache::restore_cpu(&mut self.states, state, len);
    }
}

struct EvalOptions {
    gpu: bool,
    gate: saetta::eval::Options,
}

/// `--gpu` runs the GPU model instead of the CPU one; `--long N` runs the
/// first N tokens of the long prompt (default 0, off), through prefill, or
/// one token at a time with `--decode`; `--no-ppl` skips the perplexity text.
fn eval_options(options: &[&str]) -> Option<EvalOptions> {
    let mut o = EvalOptions {
        gpu: false,
        gate: saetta::eval::Options {
            ppl: true,
            long: 0,
            decode: false,
        },
    };
    let mut rest = options.iter();
    while let Some(&option) = rest.next() {
        match option {
            "--gpu" => o.gpu = true,
            "--no-ppl" => o.gate.ppl = false,
            "--decode" => o.gate.decode = true,
            "--long" => o.gate.long = rest.next()?.parse().ok()?,
            _ => return None,
        }
    }
    Some(o)
}

fn eval(dir: &Path, reference: &Path, o: EvalOptions) -> Result<bool, Box<dyn Error>> {
    let reference = saetta::eval::Reference::load(reference)?;
    // The GPU model borrows these for the whole run, as in serve.
    let config: &'static Config = Box::leak(Box::new(Config::load(dir)?));
    let weights: &'static Weights = Box::leak(Box::new(Weights::open(dir)?));
    if o.gpu {
        return eval_gpu(config, weights, &reference, o);
    }
    let model = Model::new(config, weights)?;
    let states = model.new_states();
    let mut model = CpuModel { model, states };
    saetta::eval::gate(&mut model, &reference, &o.gate)
}

#[cfg(target_os = "macos")]
fn eval_gpu(
    config: &'static Config,
    weights: &'static Weights,
    reference: &saetta::eval::Reference,
    o: EvalOptions,
) -> Result<bool, Box<dyn Error>> {
    let mut model = load_gpu(config, weights, 0)?;
    let prefill = if model.prefill.exact { "exact" } else { "fast" };
    println!("GPU model, {prefill} prefill");
    saetta::eval::gate(&mut model, reference, &o.gate)
}

#[cfg(not(target_os = "macos"))]
fn eval_gpu(
    _: &'static Config,
    _: &'static Weights,
    _: &saetta::eval::Reference,
    _: EvalOptions,
) -> Result<bool, Box<dyn Error>> {
    Err("--gpu needs macOS".into())
}

fn info(dir: &Path) -> Result<(), Box<dyn Error>> {
    let c = Config::load(dir)?;
    let w = Weights::open(dir)?;

    let full = c
        .layer_types
        .iter()
        .filter(|&&t| t == LayerType::FullAttention)
        .count();
    println!(
        "layers          {} ({} Gated DeltaNet, {} full attention)",
        c.num_layers(),
        c.num_layers() - full,
        full
    );
    println!("hidden          {}", c.hidden_size);
    println!("vocab           {}", c.vocab_size);
    println!("rms norm eps    {:e}", c.rms_norm_eps);
    println!(
        "attention       {} query heads, {} kv heads, head dim {}, output gate {}",
        c.num_attention_heads, c.num_key_value_heads, c.head_dim, c.attn_output_gate
    );
    println!(
        "rope            theta {}, partial factor {}",
        c.rope_theta, c.partial_rotary_factor
    );
    println!(
        "delta net       {} key heads of {}, {} value heads of {}, conv kernel {}",
        c.linear_num_key_heads,
        c.linear_key_head_dim,
        c.linear_num_value_heads,
        c.linear_value_head_dim,
        c.linear_conv_kernel_dim
    );
    println!(
        "moe             {} experts, {} per token, size {}, shared expert {}",
        c.num_experts,
        c.num_experts_per_tok,
        c.moe_intermediate_size,
        c.shared_expert_intermediate_size
    );
    println!(
        "quantization    {} bits, group {}, {} modules overridden",
        c.quantization.default.bits,
        c.quantization.default.group_size,
        c.quantization.overrides.len()
    );
    println!("eos tokens      {:?}", c.eos_token_ids);

    let text = w
        .names()
        .filter(|n| n.starts_with("language_model."))
        .count();
    println!(
        "tensors         {} in {} shards ({} text, {} other)",
        w.len(),
        w.shard_count(),
        text,
        w.len() - text
    );
    Ok(())
}

#[cfg(target_os = "macos")]
fn bench_prefill(dir: &str, tokens: &str, chunk: &str) -> ExitCode {
    match (tokens.parse(), chunk.parse()) {
        (Ok(t), Ok(c)) => report(saetta::prefill::bench(
            Path::new(dir),
            t,
            c,
            3,
            exact_prefill(),
        )),
        _ => {
            eprintln!("{USAGE}");
            ExitCode::FAILURE
        }
    }
}

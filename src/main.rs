use saetta::cache::{self, Cached, Restore};
use saetta::config::{Config, LayerType};
use saetta::cpu::deltanet::DeltaNetState;
use saetta::cpu::model::{LayerState, Model};
use saetta::kv_quant::KvBits;
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

const USAGE: &str = "usage: saetta gpu-check | saetta info <model dir> | saetta convert <mlx dir> <out dir> | saetta convert-mtp <hf mtp file> <model dir> | saetta serve <model dir> [--gpu] [--draft K] [--context N] [--cache-mb N] [--cache-idle-min M] [--kv-bits 16|8|4] [--port N] [--exact-history] | saetta eval <model dir> <ref dir> [--gpu] [--long N] [--decode] [--no-ppl] [--kv-bits 16|8|4] | saetta bench-qmv <model dir> | saetta bench-moe <model dir> | saetta bench-prefill <model dir> <tokens> [chunk [16|8|4]] | saetta bench-serve-prefill <model dir> <tokens> <every> | saetta profile-prefill <model dir> <tokens> [last] | saetta bench-attention | saetta bench-attention-prefill [context...] | saetta bench-attention-gemm [context...] | saetta kernel-info <kernel...> | saetta bench-decode <model dir> [context [16|8|4]] | saetta profile-token <model dir>";

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
        ["bench-serve-prefill", dir, tokens, every] => match (tokens.parse(), every.parse()) {
            (Ok(t), Ok(e)) => report(bench_serve_prefill(dir, t, e)),
            _ => {
                eprintln!("{USAGE}");
                ExitCode::FAILURE
            }
        },
        #[cfg(target_os = "macos")]
        ["bench-attention-gemm", ref contexts @ ..] => {
            let parsed: Result<Vec<usize>, _> = contexts.iter().map(|c| c.parse()).collect();
            match parsed {
                Ok(c) if c.is_empty() => {
                    report(saetta::attention_gemm::bench(&[32768, 118784], 2048, 10))
                }
                Ok(c) => report(saetta::attention_gemm::bench(&c, 2048, 10)),
                Err(_) => {
                    eprintln!("{USAGE}");
                    ExitCode::FAILURE
                }
            }
        }
        #[cfg(target_os = "macos")]
        ["bench-attention-prefill", ref contexts @ ..] => {
            let parsed: Result<Vec<usize>, _> = contexts.iter().map(|c| c.parse()).collect();
            match parsed {
                Ok(c) if c.is_empty() => {
                    report(saetta::attention_prefill::bench(&[32768, 118784], 2048, 10))
                }
                Ok(c) => report(saetta::attention_prefill::bench(&c, 2048, 10)),
                Err(_) => {
                    eprintln!("{USAGE}");
                    ExitCode::FAILURE
                }
            }
        }
        #[cfg(target_os = "macos")]
        ["kernel-info", ref names @ ..] => {
            // Compiles only; runs nothing on the GPU.
            let gpu = match saetta::gpu::Gpu::new() {
                Ok(g) => g,
                Err(e) => {
                    eprintln!("{e}");
                    return ExitCode::FAILURE;
                }
            };
            for name in names {
                let p = gpu.pipeline(name);
                println!(
                    "{name}: max {} threads per threadgroup, {} bytes static threadgroup memory",
                    p.max_total_threads_per_threadgroup(),
                    p.static_threadgroup_memory_length()
                );
            }
            ExitCode::SUCCESS
        }
        #[cfg(target_os = "macos")]
        ["bench-qmv", dir] => report(saetta::qmv::bench(Path::new(dir), 5, 20)),
        #[cfg(target_os = "macos")]
        ["bench-moe", dir] => report(saetta::moe::bench(Path::new(dir), 5)),
        #[cfg(target_os = "macos")]
        ["bench-prefill", dir, tokens] => bench_prefill(dir, tokens, "1024", "16"),
        #[cfg(target_os = "macos")]
        ["bench-prefill", dir, tokens, chunk] => bench_prefill(dir, tokens, chunk, "16"),
        #[cfg(target_os = "macos")]
        ["bench-prefill", dir, tokens, chunk, bits] => bench_prefill(dir, tokens, chunk, bits),
        #[cfg(target_os = "macos")]
        ["profile-prefill", dir, tokens, ref rest @ ..] if rest.len() <= 1 => {
            match (tokens.parse(), rest.first().map(|l| l.parse()).transpose()) {
                (Ok(t), Ok(last)) => report(saetta::prefill::bench_profile(
                    Path::new(dir),
                    t,
                    PREFILL_CHUNK,
                    exact_prefill(),
                    last,
                )),
                _ => {
                    eprintln!("{USAGE}");
                    ExitCode::FAILURE
                }
            }
        }
        #[cfg(target_os = "macos")]
        ["bench-qmm2", dir] => report(saetta::qmm::bench(Path::new(dir), &[512, 2048], 10)),
        #[cfg(target_os = "macos")]
        ["bench-qmm", dir, rows] => match rows.parse() {
            Ok(m) => report(saetta::prefill::bench_qmm(Path::new(dir), m, 10)),
            Err(_) => {
                eprintln!("{USAGE}");
                ExitCode::FAILURE
            }
        },
        #[cfg(target_os = "macos")]
        ["bench-decode", dir] => bench_decode(dir, None, None),
        #[cfg(target_os = "macos")]
        ["bench-decode", dir, context] => bench_decode(dir, Some(context), None),
        #[cfg(target_os = "macos")]
        ["bench-decode", dir, context, bits] => bench_decode(dir, Some(context), Some(bits)),
        #[cfg(target_os = "macos")]
        ["profile-token", dir] => report(saetta::profile::profile_token(Path::new(dir), 40, 9)),
        #[cfg(target_os = "macos")]
        ["bench-attention"] => report(
            saetta::attention::bench(&[1024, 8192, 32768, 131072, 262144], 10, 5)
                .map_err(Into::into),
        ),
        _ => {
            eprintln!("{USAGE}");
            ExitCode::from(2)
        }
    }
}

/// Decode at lengths up to the context, 32768 by default. The model is
/// loaded with room for the context and a KV cache of `bits`, 16 by default.
#[cfg(target_os = "macos")]
fn bench_decode(dir: &str, context: Option<&str>, bits: Option<&str>) -> ExitCode {
    let Some(context) = context.map_or(Some(32768), |c| c.parse::<usize>().ok()) else {
        eprintln!("bad context");
        return ExitCode::FAILURE;
    };
    let Some(bits) = bits.map_or(Some(KvBits::Half), KvBits::parse) else {
        eprintln!("bad KV bits: 16, 8 or 4");
        return ExitCode::FAILURE;
    };
    let lengths: Vec<usize> = [40, 1024, 4096, 8192, 16384, 32768, 65536, 131072, 262144]
        .into_iter()
        .filter(|&n| n <= context)
        .collect();
    report(saetta::attention::bench_decode(
        Path::new(dir),
        &lengths,
        context,
        bits,
        5,
    ))
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

#[derive(Clone, Copy)]
struct ServeFlags {
    gpu: bool,
    /// Tokens the MTP layer drafts per round; 0 turns drafting off. With
    /// `--gpu` it is 3 unless `--draft` says otherwise.
    draft: usize,
    port: u16,
    exact_history: bool,
    memory: Memory,
}

/// What the GPU server may allocate besides the weights.
#[derive(Clone, Copy)]
struct Memory {
    /// Tokens of context: the KV caches may grow to this many. They start at
    /// `gpu_model::KV_START` and double when a prompt needs more, since
    /// Metal makes a whole buffer resident once a kernel uses it.
    context: usize,
    /// Bytes for prompt cache snapshots.
    cache_bytes: usize,
    /// How the KV caches store keys and values.
    kv_bits: KvBits,
    /// Snapshots unused for this long go when they keep the KV caches more
    /// than twice as long as a request needs; None keeps them.
    cache_idle: Option<std::time::Duration>,
}

impl Default for Memory {
    fn default() -> Memory {
        Memory {
            // The full context of Qwen3.6.
            context: 262144,
            cache_bytes: 1 << 30,
            kv_bits: KvBits::Half,
            cache_idle: Some(std::time::Duration::from_secs(15 * 60)),
        }
    }
}

/// `[--gpu] [--draft K] [--context N] [--cache-mb N] [--cache-idle-min M] [--kv-bits 16|8|4] [--port N]
/// [--exact-history]` after `serve <dir>`.
fn serve_flags(flags: &[&str]) -> Result<ServeFlags, String> {
    let mut draft = None;
    let mut f = ServeFlags {
        gpu: false,
        draft: 0,
        port: 8090,
        exact_history: false,
        memory: Memory::default(),
    };
    let mut it = flags.iter();
    while let Some(&flag) = it.next() {
        match flag {
            "--gpu" => f.gpu = true,
            "--draft" => {
                let value = it.next().ok_or("--draft needs a number")?;
                let k: usize = value
                    .parse()
                    .map_err(|_| format!("bad draft count {value}"))?;
                if k > 7 {
                    return Err("--draft takes 0 to 7".into());
                }
                draft = Some(k);
            }
            "--exact-history" => f.exact_history = true,
            "--context" => {
                let value = it.next().ok_or("--context needs a number of tokens")?;
                let n: usize = value.parse().map_err(|_| format!("bad context {value}"))?;
                if n < 1024 {
                    return Err("--context takes at least 1024 tokens".into());
                }
                // Prefill's attention reads keys in blocks of 32.
                f.memory.context = n.next_multiple_of(32);
            }
            "--kv-bits" => {
                let value = it.next().ok_or("--kv-bits needs 16, 8 or 4")?;
                f.memory.kv_bits = KvBits::parse(value).ok_or(format!("bad --kv-bits {value}"))?;
            }
            "--cache-mb" => {
                let value = it.next().ok_or("--cache-mb needs a number")?;
                let mb: usize = value
                    .parse()
                    .map_err(|_| format!("bad cache size {value}"))?;
                f.memory.cache_bytes = mb << 20;
            }
            "--cache-idle-min" => {
                let value = it.next().ok_or("--cache-idle-min needs minutes")?;
                let min: f64 = value
                    .parse()
                    .ok()
                    .filter(|m: &f64| *m >= 0.0 && m.is_finite())
                    .ok_or(format!("bad --cache-idle-min {value}"))?;
                f.memory.cache_idle =
                    (min > 0.0).then(|| std::time::Duration::from_secs_f64(min * 60.0));
            }
            "--port" => {
                let value = it.next().ok_or("--port needs a number")?;
                f.port = value.parse().map_err(|_| format!("bad port {value}"))?;
            }
            _ => return Err(format!("unknown flag {flag}")),
        }
    }
    f.draft = match draft {
        Some(k) if k > 0 && !f.gpu => return Err("--draft needs --gpu".into()),
        Some(k) => k,
        None if f.gpu => 3,
        None => 0,
    };
    Ok(f)
}

/// Serves the model on 127.0.0.1, on the GPU with `gpu`, else on the CPU
/// (slow, but it runs the whole path).
fn serve(dir: &Path, flags: ServeFlags) -> Result<(), Box<dyn Error>> {
    let ServeFlags {
        gpu,
        port,
        exact_history,
        ..
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
        return serve_gpu(config, weights, tokenizer, settings, listener, &what, flags);
    }
    let model = Model::new(config, weights)?;
    let states = model.new_states();
    let model = Cached::new(
        CpuModel { model, states },
        cache::DEFAULT_LIMIT,
        cache::DEFAULT_EVERY,
    );
    let model = anchored(model, &tokenizer);
    let server = Arc::new(Server::new(model, tokenizer, settings)?);
    eprintln!("serving {what} (CPU model)");
    server.serve(listener)?;
    Ok(())
}

/// The cache keeps the snapshot at the end of the system prompt.
fn anchored<M: saetta::server::Model + Restore>(
    model: Cached<M>,
    tokenizer: &Tokenizer,
) -> Cached<M> {
    match tokenizer.token_id("<|im_end|>") {
        Some(end) => model.anchor_after(end),
        None => model,
    }
}

#[cfg(not(target_os = "macos"))]
fn serve_gpu(
    _: &Config,
    _: &Weights,
    _: Tokenizer,
    _: Settings,
    _: TcpListener,
    _: &str,
    _: ServeFlags,
) -> Result<(), Box<dyn Error>> {
    Err("--gpu needs macOS".into())
}

#[cfg(target_os = "macos")]
fn serve_gpu(
    config: &'static Config,
    weights: &'static Weights,
    tokenizer: Tokenizer,
    settings: Settings,
    listener: TcpListener,
    what: &str,
    flags: ServeFlags,
) -> Result<(), Box<dyn Error>> {
    let ServeFlags { draft, memory, .. } = flags;
    let served = load_gpu(config, weights, draft, memory)?;
    let snapshots = served.snapshots;
    let mut model = anchored(
        Cached::new(served, snapshots, SNAPSHOT_EVERY).min_piece(PREFILL_CHUNK),
        &tokenizer,
    );
    if let Some(idle) = memory.cache_idle {
        model = model.expire_after(idle);
    }
    let server = Arc::new(Server::new(model, tokenizer, settings)?);
    let drafts = if draft > 0 {
        format!(", MTP drafts {draft} per round")
    } else {
        String::new()
    };
    eprintln!(
        "serving {what} (GPU model, {} tokens of context{drafts})",
        memory.context
    );
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
    memory: Memory,
) -> Result<GpuServed, Box<dyn Error>> {
    let gpu: &'static saetta::gpu::Gpu = Box::leak(Box::new(saetta::gpu::Gpu::new()?));
    let context = memory.context;
    let mut model = saetta::gpu_model::GpuModel::load_with_bits(
        gpu,
        config,
        weights,
        config.num_layers(),
        context,
        memory.kv_bits,
    )?;
    let layers = model.deltanet_states();
    let snapshot: usize = layers
        .iter()
        .map(|s| (s.conv.length() + s.recurrent.length()) as usize)
        .sum();
    // The cache adds a snapshot before it evicts one, so the pool has one more.
    let snapshots = (memory.cache_bytes / snapshot).max(2) - 1;
    let pool = saetta::gpu_cache::SnapshotPool::new(gpu, &layers, snapshots + 1);
    let mut prefill = saetta::prefill::Prefill::new(gpu, config, weights, PREFILL_CHUNK)?;
    prefill.exact = exact_prefill();
    if draft > 0 {
        model.enable_verify(draft + 1);
    }
    let spec = if draft > 0 {
        let mtp = saetta::gpu_mtp::GpuMtp::load(gpu, config, weights, context)?;
        let norm = weights.tensor("language_model.model.norm.weight")?.to_f32();
        Some(Drafting {
            spec: saetta::speculative::GpuSpec::new(&model, mtp, config, draft),
            norm,
            eps: config.rms_norm_eps,
        })
    } else {
        None
    };
    report_memory(&model, memory, draft > 0, snapshots, snapshot);
    Ok(GpuServed {
        model,
        prefill,
        pool,
        snapshots,
        last_blit: RefCell::new(None),
        drafting: spec,
        rows: saetta::gpu_cache::RowBackup::new(),
    })
}

/// Prints what the server allocates besides the weights, at most: the
/// decode KV caches as allocated, prefill's f16 copy of them, the MTP
/// layer's cache (one more attention layer), the snapshot pool, and the KV
/// rows a detour keeps aside (`cache::BACKUP_ROWS`).
#[cfg(target_os = "macos")]
fn report_memory(
    model: &saetta::gpu_model::GpuModel,
    memory: Memory,
    mtp: bool,
    snapshots: usize,
    snapshot: usize,
) {
    use saetta::gpu_model::State;
    let caches: Vec<_> = model
        .states()
        .filter_map(|s| match s {
            State::Full(c) => Some(c),
            State::Linear(_) => None,
        })
        .collect();
    let decode: usize = caches
        .iter()
        .map(|c| (c.keys.length() + c.values.length()) as usize)
        .sum();
    // Keys and values of one token in every attention layer (and the MTP
    // layer's). The caches start at `allocated` tokens and grow with the
    // context, so the KV is counted both ways.
    let row = decode / caches[0].capacity;
    let one_layer = row / caches.len();
    let token = row + usize::from(mtp) * one_layer;
    let allocated = model.allocated();
    let (start, full) = (token * allocated, token * memory.context);
    let cache = (snapshots + 1) * snapshot;
    let backup = cache::BACKUP_ROWS * row;
    let mb = |b: usize| b >> 20;
    eprintln!(
        "memory besides the weights: KV {} MB for the first {} tokens, growing to {} MB at {} ({} KB a token), prompt cache {} MB ({} snapshots of {} MB) plus {} MB of KV rows, {} to {} MB in all",
        mb(start),
        allocated,
        mb(full),
        memory.context,
        token >> 10,
        mb(cache),
        snapshots + 1,
        mb(snapshot),
        mb(backup),
        mb(start + cache + backup),
        mb(full + cache + backup)
    );
}

#[cfg(target_os = "macos")]
struct GpuServed {
    model: saetta::gpu_model::GpuModel<'static>,
    pool: saetta::gpu_cache::SnapshotPool,
    /// The last snapshot copy, which may still run. The CPU must wait for it
    /// before it writes the states, as `reset` does.
    last_blit: RefCell<Option<metal::CommandBuffer>>,
    prefill: saetta::prefill::Prefill<'static>,
    /// The prompt cache's snapshot limit, from `--cache-mb`.
    snapshots: usize,
    drafting: Option<Drafting>,
    /// KV rows kept aside during a detour (cache::Restore::backup_rows).
    rows: saetta::gpu_cache::RowBackup,
}

/// The KV cache of each attention layer.
#[cfg(target_os = "macos")]
fn kv_caches<'m>(
    model: &'m saetta::gpu_model::GpuModel<'static>,
) -> Vec<&'m saetta::gpu_layers::GpuKvCache> {
    model
        .states()
        .filter_map(|s| match s {
            saetta::gpu_model::State::Full(c) => Some(c),
            saetta::gpu_model::State::Linear(_) => None,
        })
        .collect()
}

/// Inputs shorter than this run as decode steps. Measured with bench-prefill
/// and bench-decode (2026-11-11, 3 runs): prefill takes 27 ms for 2 tokens,
/// 55 ms for 8, 179 ms for 17 and 458 ms for 64; a decode step takes 10 to
/// 12 ms up to 8k of context. So prefill wins from about 8 tokens.
#[cfg(target_os = "macos")]
const SHORT_PREFILL: usize = 8;

/// SAETTA_TAIL_STEPS=1: with drafting, a prompt shorter than SHORT_PREFILL
/// runs as decode steps, as before GpuSpec::run_rows.
#[cfg(target_os = "macos")]
fn tail_steps() -> bool {
    std::env::var("SAETTA_TAIL_STEPS").is_ok_and(|v| v == "1")
}

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

/// Tokens per prefill dispatch. Measured with bench-prefill: 2048 is about
/// 5% faster than 1024 at 4k and 15k tokens (more rows per expert tile),
/// and 4096 no better at 15k.
#[cfg(target_os = "macos")]
const PREFILL_CHUNK: usize = 2048;

/// The GPU prompt cache takes a snapshot every this many tokens of a long
/// prefill, and each piece between two is one prefill call. Pieces of
/// cache::DEFAULT_EVERY (1024) would run half chunks and pay the fixed cost
/// of a call four times as often.
/// A follow-up turn of 3k tokens that crosses a multiple of it runs in one
/// call, not two: `Cached::min_piece` skips a snapshot that leaves a piece
/// under PREFILL_CHUNK, which costs about as much as a full chunk.
#[cfg(target_os = "macos")]
const SNAPSHOT_EVERY: usize = 2 * PREFILL_CHUNK;

#[cfg(target_os = "macos")]
/// Fast prefill unless SAETTA_EXACT=1: track F's eval found it within
/// 0.04% of the f32 perplexity, closer than MLX in bf16.
fn exact_prefill() -> bool {
    saetta::prefill::exact_from_env()
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
        // The prefill writes the KV rows of all the tokens, and the MTP cache
        // never holds more entries than the model has tokens.
        let need = m.position() + tokens.len();
        m.reserve(need);
        if let Some(d) = &mut self.drafting {
            d.spec.mtp.reserve(need);
            d.spec.report(m.position());
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
                // The multi-row verify, k + 1 tokens a pass, unless
                // SAETTA_TAIL_STEPS=1 (to compare).
                Some(d) if !tail_steps() => d.spec.run_rows(m, tokens),
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
                    // `run`, keeping the hidden states of the last tokens for
                    // the MTP cache.
                    let n = tokens.len();
                    let (logits, x) = self.prefill.run_hidden(m.gpu(), tokens, pos, &states);
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
            "prefill {} tokens in {secs:.2} s ({:.1} ms/token), GPU buffers {} MB",
            tokens.len(),
            secs * 1e3 / tokens.len() as f64,
            m.gpu().device.current_allocated_size() >> 20
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
            Some(d) => d.spec.speculate(&mut self.model, pending, decide),
            None => match decide.sampling() {
                Some((p, sampler)) if saetta::sample_gpu::supports(&p) => {
                    (Vec::new(), self.model.run_sampled(pending, &p, sampler))
                }
                _ => {
                    let logits = self.model.forward(pending);
                    (Vec::new(), decide.pick(&logits))
                }
            },
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

    /// The KV rows stay in their slots.
    fn keeps_rows(&self) -> bool {
        true
    }

    fn backup_rows(&mut self, from: usize, n: usize) {
        let caches = kv_caches(&self.model);
        self.rows.save(self.model.gpu(), &caches, from, n);
        eprintln!(
            "kv rows {from}..{} backed up, {:.0} MB",
            from + n,
            self.rows.bytes() as f64 / 1e6
        );
    }

    fn restore_rows(&mut self) {
        let caches = kv_caches(&self.model);
        self.rows.restore(self.model.gpu(), &caches);
        eprintln!("kv rows put back");
    }

    fn discard_rows(&mut self) {
        self.rows.clear();
    }

    /// Shrinks the KV caches after a long conversation, once no snapshot
    /// needs its rows.
    fn release_rows(&mut self, need: usize) -> usize {
        let held = self.model.shrink(need);
        if let Some(d) = &mut self.drafting {
            d.spec.mtp.shrink(held);
        }
        held
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
    kv_bits: KvBits,
    gate: saetta::eval::Options,
}

/// `--gpu` runs the GPU model instead of the CPU one; `--long N` runs the
/// first N tokens of the long prompt (default 0, off), through prefill, or
/// one token at a time with `--decode`; `--no-ppl` skips the perplexity text;
/// `--kv-bits 16|8|4` sets the GPU model's KV cache.
fn eval_options(options: &[&str]) -> Option<EvalOptions> {
    let mut o = EvalOptions {
        gpu: false,
        kv_bits: KvBits::Half,
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
            "--kv-bits" => o.kv_bits = KvBits::parse(rest.next()?)?,
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
    let memory = Memory {
        kv_bits: o.kv_bits,
        ..Memory::default()
    };
    let mut model = load_gpu(config, weights, 0, memory)?;
    let prefill = if model.prefill.exact { "exact" } else { "fast" };
    println!("GPU model, {prefill} prefill, {:?} KV cache", o.kv_bits);
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
/// Runs a prompt of `tokens` tokens through the model as `serve --gpu`
/// builds it (no drafting), with prompt cache snapshots every `every`
/// tokens. The model prints the time of each piece; this prints the total.
#[cfg(target_os = "macos")]
fn bench_serve_prefill(dir: &str, tokens: usize, every: usize) -> Result<(), Box<dyn Error>> {
    use saetta::server::Model as _;
    let dir = Path::new(dir);
    let config: &'static Config = Box::leak(Box::new(Config::load(dir)?));
    let weights: &'static Weights = Box::leak(Box::new(Weights::open(dir)?));
    let memory = Memory {
        context: (tokens + 64).next_multiple_of(1024),
        ..Memory::default()
    };
    let served = load_gpu(config, weights, 0, memory)?;
    let snapshots = served.snapshots;
    let mut model = Cached::new(served, snapshots, every);
    let prompt = saetta::prefill::test_tokens(tokens);
    let start = Instant::now();
    model.prefill(&prompt);
    let secs = start.elapsed().as_secs_f64();
    println!(
        "prompt of {tokens} tokens, snapshots every {every}: {secs:.1} s to the logits ({:.0} tokens/s)",
        tokens as f64 / secs
    );
    Ok(())
}

#[cfg(target_os = "macos")]
fn bench_prefill(dir: &str, tokens: &str, chunk: &str, bits: &str) -> ExitCode {
    match (tokens.parse(), chunk.parse(), KvBits::parse(bits)) {
        (Ok(t), Ok(c), Some(b)) => report(saetta::prefill::bench(
            Path::new(dir),
            t,
            c,
            3,
            exact_prefill(),
            b,
        )),
        _ => {
            eprintln!("{USAGE}");
            ExitCode::FAILURE
        }
    }
}

#[cfg(test)]
mod tests {
    use super::serve_flags;

    #[test]
    fn draft_defaults_to_3_on_the_gpu() {
        let draft = |flags: &[&str]| serve_flags(flags).map(|f| f.draft);
        assert_eq!(draft(&["--gpu"]), Ok(3));
        assert_eq!(draft(&["--gpu", "--draft", "0"]), Ok(0));
        assert_eq!(draft(&["--draft", "5", "--gpu"]), Ok(5));
        assert_eq!(draft(&[]), Ok(0));
        assert_eq!(draft(&["--draft", "0"]), Ok(0));
        assert!(draft(&["--draft", "2"]).is_err());
        assert!(draft(&["--gpu", "--draft", "8"]).is_err());
    }
}

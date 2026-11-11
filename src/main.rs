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

const USAGE: &str = "usage: saetta gpu-check | saetta info <model dir> | saetta convert <mlx dir> <out dir> | saetta convert-mtp <hf mtp file> <model dir> | saetta serve <model dir> [--gpu] [--port N] | saetta bench-qmv <model dir> | saetta bench-moe <model dir> | saetta bench-prefill <model dir> <tokens> [chunk]";

fn main() -> ExitCode {
    let args: Vec<String> = env::args().skip(1).collect();
    let args: Vec<&str> = args.iter().map(String::as_str).collect();
    match args[..] {
        ["gpu-check"] => saetta::gpu::check(),
        ["info", dir] => report(info(Path::new(dir))),
        ["convert", src, out] => report(convert(Path::new(src), Path::new(out))),
        ["convert-mtp", file, dir] => report(convert_mtp(Path::new(file), Path::new(dir))),
        ["serve", dir, ref rest @ ..] => match serve_flags(rest) {
            Ok((gpu, port)) => report(serve(Path::new(dir), port, gpu)),
            Err(e) => {
                eprintln!("{e}\n{USAGE}");
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

/// `[--gpu] [--port N]` after `serve <dir>`.
fn serve_flags(flags: &[&str]) -> Result<(bool, u16), String> {
    let (mut gpu, mut port) = (false, 8090);
    let mut it = flags.iter();
    while let Some(&flag) = it.next() {
        match flag {
            "--gpu" => gpu = true,
            "--port" => {
                let value = it.next().ok_or("--port needs a number")?;
                port = value.parse().map_err(|_| format!("bad port {value}"))?;
            }
            _ => return Err(format!("unknown flag {flag}")),
        }
    }
    Ok((gpu, port))
}

/// Serves the model on 127.0.0.1, on the GPU with `gpu`, else on the CPU
/// (slow, but it runs the whole path).
fn serve(dir: &Path, port: u16, gpu: bool) -> Result<(), Box<dyn Error>> {
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
    };
    let listener = TcpListener::bind(("127.0.0.1", port))?;
    let what = format!("{} on 127.0.0.1:{port}", dir.display());
    if gpu {
        return serve_gpu(config, weights, tokenizer, settings, listener, &what);
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
) -> Result<(), Box<dyn Error>> {
    let gpu: &'static saetta::gpu::Gpu = Box::leak(Box::new(saetta::gpu::Gpu::new()?));
    let model =
        saetta::gpu_model::GpuModel::load(gpu, config, weights, config.num_layers(), GPU_CONTEXT)?;
    let layers = model.deltanet_states();
    let pool = saetta::gpu_cache::SnapshotPool::new(gpu, &layers, cache::DEFAULT_LIMIT + 1);
    let served = GpuServed {
        model,
        pool,
        last_blit: RefCell::new(None),
    };
    let model = Cached::new(served, cache::DEFAULT_LIMIT, cache::DEFAULT_EVERY);
    let server = Arc::new(Server::new(model, tokenizer, settings)?);
    eprintln!("serving {what} (GPU model, {GPU_CONTEXT} tokens of context)");
    server.serve(listener)?;
    Ok(())
}

#[cfg(target_os = "macos")]
struct GpuServed {
    model: saetta::gpu_model::GpuModel<'static>,
    pool: saetta::gpu_cache::SnapshotPool,
    /// The last snapshot copy, which may still run. The CPU must wait for it
    /// before it writes the states, as `reset` does.
    last_blit: RefCell<Option<metal::CommandBuffer>>,
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
        let start = Instant::now();
        let Some((&last, rest)) = tokens.split_last() else {
            return Vec::new();
        };
        for &t in rest {
            m.run(t);
        }
        let logits = m.forward(last);
        let secs = start.elapsed().as_secs_f64();
        eprintln!(
            "prefill {} tokens in {secs:.2} s ({:.1} ms/token)",
            tokens.len(),
            secs * 1e3 / tokens.len() as f64
        );
        logits
    }

    fn step(&mut self, token: u32) -> Vec<f32> {
        self.model.forward(token)
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
    }
}

#[cfg(target_os = "macos")]
impl Restore for GpuServed {
    type State = saetta::gpu_cache::GpuSnapshot;

    fn save(&self) -> Self::State {
        let m = &self.model;
        let (snapshot, blit) = saetta::gpu_cache::save(m.gpu(), &self.pool, &m.deltanet_states());
        *self.last_blit.borrow_mut() = Some(blit);
        snapshot
    }

    fn restore(&mut self, state: &Self::State, len: usize) {
        let m = &mut self.model;
        let blit = saetta::gpu_cache::restore(m.gpu(), state, &m.deltanet_states());
        *self.last_blit.borrow_mut() = Some(blit);
        m.set_position(len);
    }
}

struct CpuModel {
    model: Model<'static>,
    states: Vec<LayerState>,
}

impl CpuModel {
    fn run(&mut self, token: u32) -> Vec<f32> {
        self.model
            .forward(token, &mut self.states)
            .unwrap_or_else(|e| panic!("forward failed: {e}"))
    }
}

impl server::Model for CpuModel {
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

impl Restore for CpuModel {
    type State = Vec<DeltaNetState>;

    fn save(&self) -> Self::State {
        cache::save_cpu(&self.states)
    }

    fn restore(&mut self, state: &Self::State, len: usize) {
        cache::restore_cpu(&mut self.states, state, len);
    }
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
        (Ok(t), Ok(c)) => report(saetta::prefill::bench(Path::new(dir), t, c, 3)),
        _ => {
            eprintln!("{USAGE}");
            ExitCode::FAILURE
        }
    }
}

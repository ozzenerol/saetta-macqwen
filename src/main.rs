use saetta::config::{Config, LayerType};
use saetta::cpu::model::{LayerState, Model};
use saetta::sample::Params;
use saetta::server::{self, Server, Settings};
use saetta::tokenizer::Tokenizer;
use saetta::weights::Weights;
use std::env;
use std::error::Error;
use std::net::TcpListener;
use std::path::Path;
use std::process::ExitCode;
use std::sync::Arc;
use std::time::Instant;

const USAGE: &str = "usage: saetta gpu-check | saetta info <model dir> | saetta convert <mlx dir> <out dir> | saetta serve <model dir> [--port N] | saetta bench-qmv <model dir> | saetta bench-moe <model dir>";

fn main() -> ExitCode {
    let args: Vec<String> = env::args().skip(1).collect();
    let args: Vec<&str> = args.iter().map(String::as_str).collect();
    match args[..] {
        ["gpu-check"] => saetta::gpu::check(),
        ["info", dir] => report(info(Path::new(dir))),
        ["convert", src, out] => report(convert(Path::new(src), Path::new(out))),
        ["serve", dir] => report(serve(Path::new(dir), 8090)),
        ["serve", dir, "--port", port] => match port.parse() {
            Ok(port) => report(serve(Path::new(dir), port)),
            Err(_) => {
                eprintln!("bad port {port}");
                ExitCode::from(2)
            }
        },
        #[cfg(target_os = "macos")]
        ["bench-qmv", dir] => report(saetta::qmv::bench(Path::new(dir), 5, 20)),
        #[cfg(target_os = "macos")]
        ["bench-moe", dir] => report(saetta::moe::bench(Path::new(dir), 5)),
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

/// Serves the CPU model on 127.0.0.1. Slow, but it runs the whole path
/// until the GPU model replaces it behind the same trait.
fn serve(dir: &Path, port: u16) -> Result<(), Box<dyn Error>> {
    // The model borrows these for as long as the server runs.
    let config: &'static Config = Box::leak(Box::new(Config::load(dir)?));
    let weights: &'static Weights = Box::leak(Box::new(Weights::open(dir)?));
    let model = Model::new(config, weights)?;
    let states = model.new_states();
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
    let server = Arc::new(Server::new(
        CpuModel { model, states },
        tokenizer,
        settings,
    )?);
    let listener = TcpListener::bind(("127.0.0.1", port))?;
    eprintln!("serving {} on 127.0.0.1:{port} (CPU model)", dir.display());
    server.serve(listener)?;
    Ok(())
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

    /// The recurrent state cannot go back yet; snapshots come later.
    fn truncate(&mut self, _len: usize) -> bool {
        false
    }

    fn reset(&mut self) {
        self.states = self.model.new_states();
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

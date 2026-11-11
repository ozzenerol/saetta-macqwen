use saetta::config::{Config, LayerType};
use saetta::weights::Weights;
use std::env;
use std::error::Error;
use std::path::Path;
use std::process::ExitCode;

const USAGE: &str = "usage: saetta gpu-check | saetta info <model dir>";

fn main() -> ExitCode {
    let args: Vec<String> = env::args().skip(1).collect();
    let args: Vec<&str> = args.iter().map(String::as_str).collect();
    match args[..] {
        ["gpu-check"] => saetta::gpu::check(),
        ["info", dir] => match info(Path::new(dir)) {
            Ok(()) => ExitCode::SUCCESS,
            Err(e) => {
                eprintln!("error: {e}");
                ExitCode::FAILURE
            }
        },
        _ => {
            eprintln!("{USAGE}");
            ExitCode::from(2)
        }
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

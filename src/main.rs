use std::env;
use std::process::ExitCode;

fn main() -> ExitCode {
    let args: Vec<String> = env::args().skip(1).collect();
    match args.first().map(String::as_str) {
        Some("gpu-check") => saetta::gpu::check(),
        _ => {
            eprintln!("usage: saetta gpu-check");
            ExitCode::from(2)
        }
    }
}

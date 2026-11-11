#![cfg(target_os = "macos")]

use saetta::gpu::{Gpu, contents, dispatch_1d};
use std::io::Write;

#[test]
fn mapped_file_reads_on_gpu() {
    // Not a multiple of the page size, so the buffer covers a partial page.
    let words: Vec<u32> = (0..10_000u32).map(|i| i.wrapping_mul(2654435761)).collect();
    let path = std::env::temp_dir().join(format!("saetta-gpu-test-{}", std::process::id()));
    let mut file = std::fs::File::create(&path).unwrap();
    for w in &words {
        file.write_all(&w.to_le_bytes()).unwrap();
    }
    drop(file);

    let gpu = Gpu::new().unwrap();
    let mapped = gpu.map_file(&path).unwrap();
    std::fs::remove_file(&path).unwrap();

    // Read from an offset, the way a tensor inside a shard is read.
    let skip = 1000;
    let n = words.len() - skip;
    let out = gpu.buffer(n * 4);
    let copy = gpu.pipeline("copy_u32");
    gpu.run(|enc| {
        dispatch_1d(
            enc,
            &copy,
            &[(&mapped.buffer, (skip * 4) as u64), (&out, 0)],
            n,
        )
    });
    assert_eq!(contents::<u32>(&out, n), &words[skip..]);
    assert_eq!(&mapped.bytes()[..4], &words[0].to_le_bytes());

    // Wrap a range that starts and ends inside pages, as the loader will.
    let range = &mapped.bytes()[(5000 * 4)..(9000 * 4)];
    let (buffer, offset) = unsafe { gpu.buffer_no_copy(range) };
    let n = 4000;
    let out = gpu.buffer(n * 4);
    gpu.run(|enc| dispatch_1d(enc, &copy, &[(&buffer, offset), (&out, 0)], n));
    assert_eq!(contents::<u32>(&out, n), &words[5000..9000]);
}

/// Maps a 5 GB model shard and reads a 64 KB range past the 4 GB mark on
/// the GPU.
/// Skipped when the model is not there.
#[test]
fn mapped_shard_reads_past_4gb() {
    let home = std::env::var("HOME").unwrap();
    let path = std::path::Path::new(&home)
        .join("llm/mlx-models/Qwen3.6-35B-A3B-5bit/model-00001-of-00005.safetensors");
    if !path.exists() {
        eprintln!("skipped: no model shard");
        return;
    }
    let gpu = Gpu::new().unwrap();
    let file = std::fs::File::open(&path).unwrap();
    let map = unsafe { memmap2::Mmap::map(&file).unwrap() };
    let offset = (4usize << 30) + 12;
    let n = 16384;
    let cpu = &map[offset..offset + n * 4];
    let (buffer, start) = unsafe { gpu.buffer_no_copy(cpu) };
    let out = gpu.buffer(n * 4);
    let copy = gpu.pipeline("copy_u32");
    gpu.run(|enc| dispatch_1d(enc, &copy, &[(&buffer, start), (&out, 0)], n));
    let gpu_bytes: Vec<u8> = contents::<u32>(&out, n)
        .iter()
        .flat_map(|w| w.to_le_bytes())
        .collect();
    assert_eq!(gpu_bytes, cpu);
}

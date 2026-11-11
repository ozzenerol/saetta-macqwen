"""Downloads only the MTP tensors of the Hugging Face checkpoint.

The MLX conversion drops the `mtp.*` tensors. This reads the checkpoint's
safetensors index and shard headers, then fetches each MTP tensor with an
HTTP range request, so the 67 GB of bf16 weights are never downloaded.
The result is one plain safetensors file with the tensors as they are.

    python3 tools/fetch_mtp.py <out file> [repo]
"""

import json
import struct
import subprocess
import sys
from pathlib import Path


def fetch(url, start, end, out=None):
    """Bytes start..end inclusive, through curl, which follows the CDN redirect."""
    cmd = ["curl", "-sSfL", "--retry", "5", "--max-time", "1800", "-r", f"{start}-{end}", url]
    if out is not None:
        with open(out, "ab") as f:
            subprocess.run(cmd, stdout=f, check=True)
        return None
    return subprocess.run(cmd, capture_output=True, check=True).stdout


def main():
    out = Path(sys.argv[1])
    repo = sys.argv[2] if len(sys.argv) > 2 else "Qwen/Qwen3.6-35B-A3B"
    base = f"https://huggingface.co/{repo}/resolve/main/"
    index = json.loads(fetch(base + "model.safetensors.index.json", 0, 100_000_000))
    names = sorted(k for k in index["weight_map"] if k.startswith("mtp."))
    shards = sorted({index["weight_map"][k] for k in names})

    plan = []  # (name, info, shard url, absolute start, absolute end)
    for shard in shards:
        url = base + shard
        n = struct.unpack("<Q", fetch(url, 0, 7))[0]
        header = json.loads(fetch(url, 8, 8 + n - 1))
        for name in names:
            if index["weight_map"][name] == shard:
                info = header[name]
                a, b = info["data_offsets"]
                plan.append((name, info, url, 8 + n + a, 8 + n + b))

    header, offset = {"__metadata__": {"format": "pt", "source": repo}}, 0
    for name, info, _, a, b in plan:
        header[name] = {"dtype": info["dtype"], "shape": info["shape"], "data_offsets": [offset, offset + b - a]}
        offset += b - a
    head = json.dumps(header).encode()
    head += b" " * (-(8 + len(head)) % 8)
    tmp = out.with_suffix(".part")
    tmp.write_bytes(struct.pack("<Q", len(head)) + head)
    for name, _, url, a, b in plan:
        print(f"{name}: {b - a} bytes", flush=True)
        fetch(url, a, b - 1, out=tmp)
    if tmp.stat().st_size != 8 + len(head) + offset:
        sys.exit(f"size mismatch: {tmp.stat().st_size}")
    tmp.rename(out)
    print(f"wrote {len(plan)} tensors, {offset} bytes, to {out}")


if __name__ == "__main__":
    main()

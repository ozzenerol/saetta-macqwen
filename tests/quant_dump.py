# Writes a few quantized tensor slices and their mlx.core.dequantize output
# for tests/quant.rs. Run on the Mac:
#   ~/llm/mlx/bin/python tests/quant_dump.py ~/llm/mlx-models/Qwen3.6-35B-A3B-5bit ~/saetta/quant-data
# Only the needed rows are read from the safetensors files.
import json
import struct
import sys
from pathlib import Path

import mlx.core as mx
import numpy as np

model = Path(sys.argv[1])
out = Path(sys.argv[2])
out.mkdir(parents=True, exist_ok=True)
index = json.load(open(model / "model.safetensors.index.json"))["weight_map"]
prefix = "language_model.model.layers.0."

# name, tensor, expert index or None, rows to take, bits
CASES = [
    ("qkv_5bit", "linear_attn.in_proj_qkv", None, 64, 5),
    ("expert_down_5bit", "mlp.switch_mlp.down_proj", 7, 64, 5),
    ("router_8bit", "mlp.gate", None, 256, 8),
    ("shared_gate_8bit", "mlp.shared_expert_gate", None, 1, 8),
]


def read_rows(name, expert, rows):
    path = model / index[name]
    with open(path, "rb") as f:
        size = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(size))
        info = header[name]
        start, _ = info["data_offsets"]
        shape = info["shape"]
        if expert is not None:
            shape = shape[1:]
        cols = shape[1]
        width = {"U32": 4, "BF16": 2}[info["dtype"]]
        row_bytes = cols * width
        skip = 0 if expert is None else expert * shape[0] * row_bytes
        f.seek(8 + size + start + skip)
        data = f.read(rows * row_bytes)
    kind = np.uint32 if info["dtype"] == "U32" else np.uint16
    return np.frombuffer(data, dtype=kind).reshape(rows, cols)


cases = {}
for case, tensor, expert, rows, bits in CASES:
    w = read_rows(prefix + tensor + ".weight", expert, rows)
    s = read_rows(prefix + tensor + ".scales", expert, rows)
    b = read_rows(prefix + tensor + ".biases", expert, rows)
    scales = mx.array(s).view(mx.bfloat16)
    biases = mx.array(b).view(mx.bfloat16)
    ref_bf16 = mx.dequantize(mx.array(w), scales, biases, group_size=64, bits=bits)
    ref_f32 = mx.dequantize(
        mx.array(w),
        scales.astype(mx.float32),
        biases.astype(mx.float32),
        group_size=64,
        bits=bits,
    )
    w.tofile(out / f"{case}.weight.u32")
    s.tofile(out / f"{case}.scales.bf16")
    b.tofile(out / f"{case}.biases.bf16")
    np.array(ref_bf16.astype(mx.float32)).tofile(out / f"{case}.ref_bf16.f32")
    np.array(ref_f32).tofile(out / f"{case}.ref_f32.f32")
    cases[case] = {"rows": rows, "cols": int(ref_f32.shape[1]), "bits": bits}
    print(case, tensor, "rows", rows, "cols", ref_f32.shape[1], "bits", bits)

json.dump(cases, open(out / "index.json", "w"), indent=1)

# Asks the running mlx_lm.server for a greedy completion of the golden
# prompt and writes the tokens and top logprobs to GOLDEN_DIR/greedy.json.
# The server keeps its model in memory, so no second copy is loaded.
# Run on the Mac:
#   ~/llm/mlx/bin/python tools/greedy_ref.py
import json
import urllib.request
from pathlib import Path

URL = "http://127.0.0.1:8080/v1/completions"
GOLDEN_DIR = Path.home() / "saetta/golden"
TOKENS = 16
TOP = 5


def main():
    index = json.loads((GOLDEN_DIR / "index.json").read_text())
    body = {
        "prompt": index["prompt"],
        "max_tokens": TOKENS,
        "temperature": 0,
        "logprobs": True,
        "top_logprobs": TOP,
    }
    req = urllib.request.Request(
        URL, json.dumps(body).encode(), {"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=120) as r:
        reply = json.load(r)

    choice = reply["choices"][0]
    steps = choice["logprobs"]["content"]
    out = {
        "prompt": index["prompt"],
        "prompt_tokens": reply["usage"]["prompt_tokens"],
        "text": choice["text"],
        "tokens": [s["id"] for s in steps],
        "top_ids": [[t["id"] for t in s["top_logprobs"]] for s in steps],
        "top_logprobs": [[t["logprob"] for t in s["top_logprobs"]] for s in steps],
    }
    (GOLDEN_DIR / "greedy.json").write_text(json.dumps(out, indent=1) + "\n")
    print(repr(out["text"]), out["tokens"])


if __name__ == "__main__":
    main()

# Decode speed of a server after prompts of growing length, for deciding
# whether --draft pays at every context. The prompt is this repo's Rust
# source, cut to about N tokens, followed by a coding request; greedy, with
# thinking off.
#   ~/llm/mlx/bin/python tools/context_bench.py --url http://127.0.0.1:8093 --out plain.json
#   ~/llm/mlx/bin/python tools/context_bench.py --compare plain.json draft.json
import argparse
import json
import pathlib
import sys
import time
import urllib.request

LENGTHS = [40, 1000, 4000, 16000, 64000, 128000]
ASK = "Write a Rust function that parses a version string like 1.2.3 into a struct, with tests."
# Rust source runs at about 3.2 characters per token.
CHARS_PER_TOKEN = 3.2


def source(root):
    files = sorted(pathlib.Path(root, "src").rglob("*.rs"))
    return "".join(f"// {f}\n{f.read_text()}\n" for f in files)


def prompt(text, tokens):
    if tokens <= 100:
        return ASK
    need = int(tokens * CHARS_PER_TOKEN)
    body = (text * (need // len(text) + 1))[:need]
    return f"Here is some code.\n\n{body}\n\n{ASK}"


def one_run(url, content, max_tokens, run):
    body = {
        "model": "saetta",
        "messages": [{"role": "system", "content": f"Run {run}."},
                     {"role": "user", "content": content}],
        "max_tokens": max_tokens,
        "stream": True,
        "temperature": 0,
        "stream_options": {"include_usage": True},
        "chat_template_kwargs": {"enable_thinking": False},
    }
    req = urllib.request.Request(url + "/v1/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    start = time.perf_counter()
    first = None
    usage = {}
    text = []
    for line in urllib.request.urlopen(req, timeout=3600):
        if not line.startswith(b"data: {"):
            continue
        d = json.loads(line[6:])
        usage = d.get("usage") or usage
        for c in d.get("choices") or []:
            piece = c.get("delta", {}).get("content") or ""
            if piece:
                first = first or time.perf_counter()
            text.append(piece)
    end = time.perf_counter()
    tokens = usage.get("completion_tokens", 0)
    return {
        "prompt_tokens": usage.get("prompt_tokens", 0),
        "tokens": tokens,
        "ttft": first - start,
        "decode": (tokens - 1) / (end - first),
        "text": "".join(text),
    }


def bench(url, root, lengths, max_tokens):
    text = source(root)
    out = {}
    for n in lengths:
        r = one_run(url, prompt(text, n), max_tokens, n)
        print(f"{r['prompt_tokens']:>7} tokens of prompt: decode {r['decode']:.1f} tok/s, "
              f"first token {r['ttft']:.2f} s", flush=True)
        out[str(n)] = r
    return out


def compare(a_path, b_path):
    a, b = json.load(open(a_path)), json.load(open(b_path))
    wins = True
    for n in a:
        if n not in b:
            continue
        x, y = a[n], b[n]
        same = "" if x["text"] == y["text"] else ", outputs differ"
        ratio = y["decode"] / x["decode"]
        wins &= ratio > 1
        print(f"{x['prompt_tokens']:>7} tokens: {x['decode']:.1f} -> {y['decode']:.1f} tok/s "
              f"({ratio:.2f}x){same}")
    print("second wins at every length" if wins else "second loses at some length")
    return wins


def margin(a_path, b_path, root, out_dir):
    """For each length where the outputs differ, writes margin-N.json with
    the request and the answer up to the first differing character, for
    the margin_at_first_difference test."""
    a, b = json.load(open(a_path)), json.load(open(b_path))
    text = source(root)
    for n in a:
        x, y = a[n]["text"], b.get(n, {}).get("text")
        if y is None or x == y:
            continue
        i = next((j for j, (p, q) in enumerate(zip(x, y)) if p != q), min(len(x), len(y)))
        case = {"system": f"Run {n}.", "user": prompt(text, int(n)), "prefix": x[:i]}
        path = pathlib.Path(out_dir, f"margin-{n}.json")
        json.dump(case, open(path, "w"))
        print(f"{path}: differs at character {i}: {x[i:i + 20]!r} vs {y[i:i + 20]!r}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8093")
    ap.add_argument("--root", default=".")
    ap.add_argument("--lengths", type=int, nargs="+", default=LENGTHS)
    ap.add_argument("--tokens", type=int, default=256)
    ap.add_argument("--out")
    ap.add_argument("--compare", nargs=2)
    ap.add_argument("--margin", nargs=3, metavar=("PLAIN", "DRAFT", "OUT_DIR"))
    args = ap.parse_args()
    if args.margin:
        margin(*args.margin[:2], args.root, args.margin[2])
        return
    if args.compare:
        sys.exit(0 if compare(*args.compare) else 1)
    result = bench(args.url, args.root, args.lengths, args.tokens)
    if args.out:
        json.dump(result, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()

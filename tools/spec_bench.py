# Decode speed and output of a server on qh-like requests: code, a tool
# call and prose, greedy with thinking off. Writes the replies to a JSON
# file so two servers (plain and --draft) can be checked for equal output.
#   ~/llm/mlx/bin/python tools/spec_bench.py --url http://127.0.0.1:8093 --out plain.json
#   ~/llm/mlx/bin/python tools/spec_bench.py --compare plain.json draft.json
# With --temperature above 0 it samples with top_k 20 and top_p 0.95 and a
# fixed seed per run, so two servers that sample alike still match.
import argparse
import json
import statistics
import sys
import time
import urllib.request

READ_FILE = {"type": "function", "function": {
    "name": "read_file",
    "description": "Read a file from the workspace",
    "parameters": {"type": "object",
                   "properties": {"path": {"type": "string", "description": "Path of the file"}},
                   "required": ["path"]}}}

CASES = {
    "code": ("Write a Rust function that parses a version string like 1.2.3 into a "
             "struct with major, minor and patch fields, and returns an error for bad "
             "input. Include tests.", []),
    "tool call": ("Open src/main.rs and tell me what the main function does.", [READ_FILE]),
    "prose": ("Explain in a few paragraphs why the sky is blue and why sunsets are red.", []),
}


def one_run(url, prompt, tools, max_tokens, run, temperature):
    body = {
        "model": "saetta",
        # A different first line per run, so the prompt cache cannot help.
        "messages": [{"role": "system", "content": f"Run {run}."},
                     {"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "stream": True,
        "temperature": temperature,
        "top_k": 20,
        "top_p": 0.95,
        "seed": 1000 + run,
        "stream_options": {"include_usage": True},
        "chat_template_kwargs": {"enable_thinking": False},
    }
    if tools:
        body["tools"] = tools
    req = urllib.request.Request(url + "/v1/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    start = time.perf_counter()
    first = None
    usage = {}
    text = []
    calls = []
    for line in urllib.request.urlopen(req):
        if not line.startswith(b"data: {"):
            continue
        d = json.loads(line[6:])
        usage = d.get("usage") or usage
        for c in d.get("choices") or []:
            delta = c.get("delta", {})
            if delta.get("content") or delta.get("tool_calls"):
                first = first or time.perf_counter()
            text.append(delta.get("content") or "")
            calls.extend(delta.get("tool_calls") or [])
    end = time.perf_counter()
    tokens = usage.get("completion_tokens", 0)
    # A tool call arrives in one piece at the end, so time it from the
    # request, prompt included.
    since = start if calls and not "".join(text).strip() else first
    return {
        "tokens": tokens,
        "ttft": first - start,
        "decode": (tokens - 1) / (end - since),
        "text": "".join(text),
        # The ids differ per request.
        "tool_calls": [c["function"] for c in calls],
    }


def bench(url, runs, max_tokens, temperature):
    out = {}
    for name, (prompt, tools) in CASES.items():
        rs = [one_run(url, prompt, tools, max_tokens, r, temperature) for r in range(runs)]
        speeds = [r["decode"] for r in rs]
        ttft = statistics.median(r["ttft"] for r in rs)
        print(f"{name}: {rs[0]['tokens']} tokens, decode {statistics.median(speeds):.1f} tok/s "
              f"(runs {', '.join(f'{s:.1f}' for s in speeds)}), first token {ttft:.2f} s",
              flush=True)
        out[name] = rs
    return out


def compare(a_path, b_path):
    a, b = json.load(open(a_path)), json.load(open(b_path))
    same = True
    for name in CASES:
        for i, (x, y) in enumerate(zip(a[name], b[name])):
            if (x["text"], x["tool_calls"]) != (y["text"], y["tool_calls"]):
                same = False
                n = next((j for j, (p, q) in enumerate(zip(x["text"], y["text"])) if p != q),
                         min(len(x["text"]), len(y["text"])))
                print(f"{name} run {i}: differs at character {n}: "
                      f"{x['text'][n:n + 40]!r} vs {y['text'][n:n + 40]!r}")
        sa = statistics.median(r["decode"] for r in a[name])
        sb = statistics.median(r["decode"] for r in b[name])
        print(f"{name}: {sa:.1f} -> {sb:.1f} tok/s ({sb / sa:.2f}x)")
    print("outputs equal" if same else "outputs differ")
    return same


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8093")
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--tokens", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=0)
    ap.add_argument("--out")
    ap.add_argument("--compare", nargs=2)
    args = ap.parse_args()
    if args.compare:
        sys.exit(0 if compare(*args.compare) else 1)
    result = bench(args.url, args.runs, args.tokens, args.temperature)
    if args.out:
        json.dump(result, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()

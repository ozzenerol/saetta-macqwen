# Measures an OpenAI-compatible server the same way for MLX and saetta:
# time to first token, prefill and decode speed at several prompt lengths,
# and a qh-like conversation that grows turn by turn.
#   ~/llm/mlx/bin/python tools/bench.py --url http://127.0.0.1:8080 \
#       --label mlx --contexts 40,1k,4k --runs 3 --out mlx.json
#   python3 tools/bench.py --report saetta.json mlx.json > table.md
# Thinking is off and temperature is 0, so the runs are repeatable. Each run
# starts with its own line, so no prompt cache helps a single-prompt case.
import argparse
import json
import statistics
import sys
import threading
import time
import urllib.request

STORY = "Write a long story about a lighthouse keeper."
FILLER = "Here is some text number {}. "


def model_id(url):
    models = json.load(urllib.request.urlopen(url + "/v1/models"))["data"]
    ids = [m["id"] for m in models]
    return next((i for i in ids if i.startswith("/")), ids[0])


def tokens_of(text):
    """'40', '1k', '128k' as a number of tokens."""
    text = text.strip().lower()
    return int(float(text[:-1]) * 1024) if text.endswith("k") else int(text)


def one_run(url, model, messages, max_tokens):
    body = json.dumps({
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "stream": True,
        "temperature": 0,
        "stream_options": {"include_usage": True},
        "chat_template_kwargs": {"enable_thinking": False},
    }).encode()
    req = urllib.request.Request(url + "/v1/chat/completions", body,
                                 {"Content-Type": "application/json"})
    start = time.perf_counter()
    first = None
    chunks = 0
    text = []
    usage = {}
    for line in urllib.request.urlopen(req, timeout=3600):
        if not line.startswith(b"data: {"):
            continue
        d = json.loads(line[6:])
        usage = d.get("usage") or usage
        choices = d.get("choices") or []
        piece = choices[0]["delta"].get("content") if choices else None
        if piece:
            text.append(piece)
            chunks += 1
            if first is None:
                first = time.perf_counter()
    end = time.perf_counter()
    if first is None:
        raise RuntimeError(f"no content came back: usage {usage}")
    tokens = usage.get("completion_tokens", chunks)
    prompt = usage.get("prompt_tokens") or 0
    cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens") or 0
    ttft = first - start
    return {
        "ttft": ttft,
        "prefill": (prompt - cached) / ttft if prompt else None,
        "decode": (tokens - 1) / (end - first) if tokens > 1 and end > first else None,
        "prompt_tokens": prompt,
        "cached_tokens": cached,
        "completion_tokens": tokens,
        "answer": "".join(text),
    }


class Progress:
    """Keeps the latest status and prints it every `every` seconds, so a long
    run can be followed from its log."""

    def __init__(self, label, every):
        self.label = label
        self.start = time.time()
        self.status = "starting"
        self.lock = threading.Lock()
        if every > 0:
            threading.Thread(target=self._loop, args=(every,), daemon=True).start()

    def set(self, status):
        with self.lock:
            self.status = status

    def line(self):
        m, s = divmod(int(time.time() - self.start), 60)
        with self.lock:
            return f"progress {self.label} {m}m{s:02d}s: {self.status}"

    def _loop(self, every):
        while True:
            time.sleep(every)
            print(self.line(), flush=True)


def median(rows, key):
    values = [r[key] for r in rows if r.get(key) is not None]
    return statistics.median(values) if values else None


def fit_filler(url, model):
    """Filler sentences for a prompt of about `target` tokens, from one short
    calibration request."""
    probe = 400
    base = one_run(url, model, [{"role": "user", "content": "Calibrate.\n" + STORY}], 2)
    full = one_run(url, model, [{"role": "user", "content": "Calibrate.\n" + "".join(
        FILLER.format(i) for i in range(probe)) + "\n" + STORY}], 2)
    fixed = base["prompt_tokens"]
    per = (full["prompt_tokens"] - fixed) / probe
    return lambda tokens: max(0, round((tokens - fixed) / per))


def bench_contexts(args, url, model, progress):
    sentences = fit_filler(url, model)
    stamp = int(time.time())
    out = []
    for name in args.contexts.split(","):
        target = tokens_of(name)
        filler = "".join(FILLER.format(i) for i in range(sentences(target)))
        runs = []
        for r in range(args.runs):
            progress.set(f"context {name} run {r + 1}/{args.runs}")
            head = f"Run {stamp}-{name}-{r}. "
            try:
                row = one_run(url, model, [{"role": "user", "content": head + filler + "\n" + STORY}],
                              args.tokens)
            except Exception as e:
                # MLX runs out of GPU memory near 150k tokens; keep what came
                # before and skip the longer contexts.
                print(f"  {name} run {r + 1}: failed: {e}", flush=True)
                out.append({"context": name, "error": str(e)[:200]})
                return out
            row.pop("answer")
            runs.append(row)
            print(f"  {name} run {r + 1}: prompt {row['prompt_tokens']}, "
                  f"ttft {row['ttft']:.2f} s, decode {row['decode'] or 0:.1f} tok/s", flush=True)
        result = {
            "context": name,
            "prompt_tokens": runs[0]["prompt_tokens"],
            "ttft": median(runs, "ttft"),
            "prefill": median(runs, "prefill"),
            "decode": median(runs, "decode"),
            "runs": runs,
        }
        out.append(result)
        print(f"{name}: prompt {result['prompt_tokens']}, ttft {result['ttft']:.2f} s, "
              f"prefill {result['prefill'] or 0:.0f} tok/s, decode {result['decode'] or 0:.1f} tok/s "
              f"(median of {args.runs})", flush=True)
    return out


SYSTEM = ("You are a coding agent working in a Go project. Use short answers. "
          + "".join(f"Rule {i}: keep functions small, test each change, and explain nothing "
                    f"that was not asked about. " for i in range(120)))


def tool_result(turn):
    """A file listing of about 1.5k tokens, standing in for a read tool."""
    lines = [f"{n:4}\tfunc step{turn}_{n}(x int) int {{ return x*{n} + {turn} }}"
             for n in range(1, 120)]
    return "\n".join(lines)


def bench_multiturn(args, url, model, progress):
    """A conversation like a qh session: a long system prompt, then turns that
    each send back the last answer and add a file read and a question. The
    prompt cache should make each turn's prefill only the new part."""
    stamp = int(time.time())
    messages = [{"role": "system", "content": f"Session {stamp}. " + SYSTEM}]
    turns = []
    for t in range(args.turns):
        progress.set(f"multi-turn {t + 1}/{args.turns}")
        messages.append({"role": "user", "content":
                         f"Here is file step{t}.go:\n{tool_result(t)}\n"
                         f"In two sentences, what does step{t}_7 return for x = 2?"})
        row = one_run(url, model, messages, args.turn_tokens)
        messages.append({"role": "assistant", "content": row.pop("answer")})
        turns.append(row)
        print(f"  turn {t + 1}: prompt {row['prompt_tokens']}, cached {row['cached_tokens']}, "
              f"ttft {row['ttft']:.2f} s, decode {row['decode'] or 0:.1f} tok/s", flush=True)
    return {
        "turns": turns,
        "ttft": median(turns[1:], "ttft"),
        "decode": median(turns, "decode"),
        "last_prompt_tokens": turns[-1]["prompt_tokens"],
    }


def fmt(v, spec):
    return "-" if v is None else format(v, spec)


def report(paths):
    results = [json.load(open(p)) for p in paths]
    labels = [r["label"] for r in results]
    print("| context | prompt tokens | " + " | ".join(
        f"{l} TTFT s | {l} prefill tok/s | {l} decode tok/s" for l in labels) + " |")
    print("|---|---|" + "---|---|---|" * len(labels))
    names = []
    for r in results:
        names += [c["context"] for c in r.get("contexts", []) if c["context"] not in names]
    for name in names:
        cells = []
        prompt = None
        for r in results:
            c = next((c for c in r.get("contexts", []) if c["context"] == name), None)
            if c is None:
                cells += ["-", "-", "-"]
                continue
            if "error" in c:
                cells += ["failed", "-", "-"]
                continue
            prompt = prompt or c["prompt_tokens"]
            cells += [fmt(c["ttft"], ".2f"), fmt(c["prefill"], ".0f"), fmt(c["decode"], ".1f")]
        print(f"| {name} | {prompt} | " + " | ".join(cells) + " |")
    multi = [(r["label"], r["multiturn"]) for r in results if r.get("multiturn")]
    if multi:
        print()
        print("| multi-turn | turns | last prompt tokens | median TTFT after turn 1, s | median decode tok/s |")
        print("|---|---|---|---|---|")
        for label, m in multi:
            print(f"| {label} | {len(m['turns'])} | {m['last_prompt_tokens']} | "
                  f"{fmt(m['ttft'], '.2f')} | {fmt(m['decode'], '.1f')} |")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8080")
    ap.add_argument("--label", default="server")
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--tokens", type=int, default=256, help="answer tokens per run")
    ap.add_argument("--contexts", default="40,1k", help="prompt sizes, such as 40,1k,4k,16k,32k")
    ap.add_argument("--turns", type=int, default=0, help="turns of the multi-turn case; 0 skips it")
    ap.add_argument("--turn-tokens", type=int, default=128)
    ap.add_argument("--progress", type=int, default=120, help="seconds between progress lines")
    ap.add_argument("--out", help="JSON file for the results")
    ap.add_argument("--report", nargs="+", metavar="JSON", help="print a markdown table and exit")
    args = ap.parse_args()
    if args.report:
        report(args.report)
        return
    model = model_id(args.url)
    progress = Progress(args.label, args.progress)
    result = {"label": args.label, "url": args.url, "model": model, "date": time.ctime(),
              "runs": args.runs, "answer_tokens": args.tokens}
    if args.contexts:
        result["contexts"] = bench_contexts(args, args.url, model, progress)
    if args.turns:
        result["multiturn"] = bench_multiturn(args, args.url, model, progress)
    progress.set("done")
    print(progress.line(), flush=True)
    if args.out:
        with open(args.out, "w") as f:
            json.dump(result, f, indent=1)


if __name__ == "__main__":
    sys.exit(main())

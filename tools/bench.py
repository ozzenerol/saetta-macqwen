# Measures an OpenAI-compatible server the same way for MLX and saetta:
# time to first token and decode speed, at a short and a long prompt.
#   ~/llm/mlx/bin/python tools/bench.py --url http://127.0.0.1:8080 --runs 3
# Thinking is off and temperature is 0, so the runs are repeatable. Each run
# uses a new prompt prefix so no prompt cache helps the long case.
import argparse
import json
import statistics
import time
import urllib.request

STORY = "Write a long story about a lighthouse keeper."


def model_id(url):
    models = json.load(urllib.request.urlopen(url + "/v1/models"))["data"]
    ids = [m["id"] for m in models]
    return next((i for i in ids if i.startswith("/")), ids[0])


def one_run(url, model, prompt, max_tokens):
    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
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
    usage = {}
    for line in urllib.request.urlopen(req):
        if not line.startswith(b"data: {"):
            continue
        d = json.loads(line[6:])
        usage = d.get("usage") or usage
        choices = d.get("choices") or []
        if choices and choices[0]["delta"].get("content"):
            chunks += 1
            if first is None:
                first = time.perf_counter()
    end = time.perf_counter()
    tokens = usage.get("completion_tokens", chunks)
    return {
        "ttft": first - start,
        "decode": (tokens - 1) / (end - first),
        "prompt_tokens": usage.get("prompt_tokens"),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8080")
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--tokens", type=int, default=256)
    ap.add_argument("--long", type=int, default=1500, help="filler sentences in the long prompt")
    args = ap.parse_args()
    model = model_id(args.url)
    stamp = int(time.time())
    for name, filler in [("short", 0), ("long", args.long)]:
        results = []
        for r in range(args.runs):
            head = f"Run {stamp}-{name}-{r}. "
            text = "".join(f"Here is some text number {i}. " for i in range(filler))
            results.append(one_run(args.url, model, head + text + "\n" + STORY, args.tokens))
        ttft = statistics.median(x["ttft"] for x in results)
        decode = statistics.median(x["decode"] for x in results)
        print(f"{name}: prompt {results[0]['prompt_tokens']} tokens, "
              f"ttft {ttft:.2f} s, decode {decode:.1f} tok/s "
              f"(median of {args.runs}; decode runs {[round(x['decode'], 1) for x in results]})")


if __name__ == "__main__":
    main()

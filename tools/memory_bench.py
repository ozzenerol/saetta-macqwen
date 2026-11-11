# Memory of a running saetta server over a qh-like session, on the Mac:
#   python3 tools/memory_bench.py --pid $(pgrep -f 'saetta serve') \
#       --turns 20 --long 45k --others 4
# After each request it prints the footprint at once and --settle seconds
# later, with the cached tokens of the request. Steps:
# 1. a growing conversation of --turns turns (the last prompt, its answer
#    and a tool result each time), as qh sends them
# 2. one prompt of --long tokens
# 3. --others short conversations with different system prompts, which push
#    the long one's snapshots out, so the KV caches can shrink
# Thinking is off and temperature 0, so runs repeat.
import argparse
import json
import re
import subprocess
import time
import urllib.request

FILLER = "Here is some text number {}. "


def tokens_of(text):
    text = text.strip().lower()
    return int(float(text[:-1]) * 1024) if text.endswith("k") else int(text)


def footprint_mb(pid):
    """phys_footprint from footprint(1), and the RSS from ps, in MB."""
    out = subprocess.run(["footprint", str(pid)], capture_output=True, text=True).stdout
    m = re.search(r"[Ff]ootprint:\s*([\d.]+)\s*([KMG])B", out)
    foot = None
    if m:
        foot = float(m.group(1)) * {"K": 1 / 1024, "M": 1, "G": 1024}[m.group(2)]
    rss = subprocess.run(["ps", "-o", "rss=", "-p", str(pid)], capture_output=True, text=True).stdout
    return foot, int(rss.strip() or 0) / 1024


def chat(url, messages, max_tokens):
    body = json.dumps({
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0,
        "chat_template_kwargs": {"enable_thinking": False},
    }).encode()
    req = urllib.request.Request(url + "/v1/chat/completions", body,
                                 {"Content-Type": "application/json"})
    start = time.time()
    r = json.load(urllib.request.urlopen(req))
    usage = r["usage"]
    cached = usage.get("prompt_tokens_details", {}).get("cached_tokens", 0)
    return r["choices"][0]["message"], usage["prompt_tokens"], cached, time.time() - start


def measure(args, label, messages):
    message, prompt, cached, secs = chat(args.url, messages, args.tokens)
    now = footprint_mb(args.pid)
    time.sleep(args.settle)
    later = footprint_mb(args.pid)
    fmt = lambda f: "?" if f[0] is None else f"{f[0]:.0f}"
    print(f"{label}: prompt {prompt}, cached {cached}, {secs:.1f} s, footprint {fmt(now)} MB "
          f"then {fmt(later)} MB after {args.settle} s, rss {now[1]:.0f} then {later[1]:.0f} MB",
          flush=True)
    return message


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8090")
    ap.add_argument("--pid", type=int, required=True)
    ap.add_argument("--tokens", type=int, default=128, help="answer tokens per request")
    ap.add_argument("--turns", type=int, default=20)
    ap.add_argument("--long", default="45k")
    ap.add_argument("--others", type=int, default=4)
    ap.add_argument("--settle", type=float, default=6)
    args = ap.parse_args()

    f = footprint_mb(args.pid)
    print(f"idle: footprint {f[0]} MB, rss {f[1]:.0f} MB", flush=True)

    messages = [{"role": "system", "content": "You are a coding agent."},
                {"role": "user", "content": "Read the files and summarize them."}]
    for turn in range(args.turns):
        answer = measure(args, f"turn {turn + 1}", messages)
        messages.append({"role": "assistant", "content": answer.get("content") or ""})
        result = " ".join(FILLER.format(turn * 100 + i) for i in range(60))
        messages.append({"role": "user", "content": f"Tool result {turn + 1}: {result}"})

    words = tokens_of(args.long) // 7
    long = " ".join(FILLER.format(i) for i in range(words))
    measure(args, f"long {args.long}", [{"role": "user", "content": long + "\nSummarize."}])

    for i in range(args.others):
        convo = [{"role": "system", "content": f"You are helper number {i}."},
                 {"role": "user", "content": "Say hello."}]
        for turn in range(4):
            answer = measure(args, f"other {i + 1} turn {turn + 1}", convo)
            convo.append({"role": "assistant", "content": answer.get("content") or ""})
            convo.append({"role": "user", "content": f"And again, {turn}."})


if __name__ == "__main__":
    main()

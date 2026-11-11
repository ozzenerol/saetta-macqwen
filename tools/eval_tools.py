"""Sends qh-style tool conversations to saetta and checks that the tool
calls parse to the same names and arguments as MLX's. Greedy, thinking off,
streamed and put together the way qh's llm.go does it.

MLX's answers can come from the running mlx_lm.server, or from a file
recorded earlier, for when the server is stopped to make room for the GPU
model:

    ~/llm/mlx/bin/python tools/eval_tools.py --record ~/saetta/eval/tool_calls.json
    ~/llm/mlx/bin/python tools/eval_tools.py --saetta http://127.0.0.1:8097 --reference ~/saetta/eval/tool_calls.json
    ~/llm/mlx/bin/python tools/eval_tools.py --saetta http://127.0.0.1:8097

Exits 1 when a conversation gives different calls.
"""

import argparse
import json
import sys
import time
import urllib.request


def tool(name, description, properties, required):
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {"type": "object", "properties": properties, "required": required},
        },
    }


def string(description):
    return {"type": "string", "description": description}


TOOLS = [
    tool(
        "read",
        "Read a text file. Lines come back numbered.",
        {"path": string("file path"), "limit": {"type": "integer", "description": "number of lines"}},
        ["path"],
    ),
    tool(
        "grep",
        "Search file contents. Returns file:line:text.",
        {"pattern": string("regular expression"), "path": string("file or directory, default .")},
        ["pattern"],
    ),
    tool(
        "bash",
        "Run a shell command in the working directory.",
        {"command": string("the command"), "timeout": {"type": "integer", "description": "seconds"}},
        ["command"],
    ),
]

SYSTEM = "You are a coding agent working in a Go project. Use the tools to answer. Call one tool at a time."


def call(id, name, arguments):
    return {"id": id, "type": "function", "function": {"name": name, "arguments": json.dumps(arguments)}}


CONVERSATIONS = [
    [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": "Read the file cmd/server/main.go."},
    ],
    [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": "Find every line that contains TODO in the internal directory."},
    ],
    [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": "Read go.mod, then build everything with go build ./..."},
        {"role": "assistant", "content": "", "tool_calls": [call("call_1", "read", {"path": "go.mod"})]},
        {"role": "tool", "tool_call_id": "call_1", "content": "1\tmodule example.com/shop\n2\n3\tgo 1.24\n"},
    ],
]


def model_id(url):
    models = json.load(urllib.request.urlopen(url + "/v1/models"))["data"]
    ids = [m["id"] for m in models]
    return next((i for i in ids if i.startswith("/")), ids[0])


def chat(url, messages):
    body = json.dumps({
        "model": model_id(url),
        "messages": messages,
        "tools": TOOLS,
        "stream": True,
        "stream_options": {"include_usage": True},
        "max_tokens": 150,
        "temperature": 0,
        "chat_template_kwargs": {"enable_thinking": False, "preserve_thinking": True},
    }).encode()
    req = urllib.request.Request(url + "/v1/chat/completions", body, {"Content-Type": "application/json"})
    text, calls, usage = "", {}, {}
    for line in urllib.request.urlopen(req, timeout=3600):
        if not line.startswith(b"data: {"):
            continue
        chunk = json.loads(line[6:])
        usage = chunk.get("usage") or usage
        for choice in chunk.get("choices") or []:
            delta = choice.get("delta", {})
            text += delta.get("content") or ""
            for tc in delta.get("tool_calls") or []:
                cur = calls.setdefault(tc.get("index", 0), {"name": "", "arguments": ""})
                cur["name"] += tc.get("function", {}).get("name") or ""
                cur["arguments"] += tc.get("function", {}).get("arguments") or ""
    parsed = [(c["name"], json.loads(c["arguments"] or "{}")) for _, c in sorted(calls.items())]
    return text, parsed, usage


def run(label, url, messages):
    start = time.time()
    text, calls, usage = chat(url, messages)
    print(f"{label}: {usage.get('prompt_tokens')} prompt tokens, {time.time() - start:.0f} s, "
          f"calls {calls}, text {text.strip()[:80]!r}", flush=True)
    # Lists, not tuples, so a recorded answer compares equal after JSON.
    return {"text": text, "calls": [[name, args] for name, args in calls]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mlx", default="http://127.0.0.1:8080")
    ap.add_argument("--saetta", default="http://127.0.0.1:8097")
    ap.add_argument("--record", help="write MLX's answers to this file and stop")
    ap.add_argument("--reference", help="MLX's answers from --record, instead of asking --mlx")
    args = ap.parse_args()

    if args.record:
        answers = [run(f"conversation {i} mlx", args.mlx, m) for i, m in enumerate(CONVERSATIONS)]
        with open(args.record, "w") as f:
            json.dump({"conversations": CONVERSATIONS, "answers": answers}, f, indent=1)
        print(f"wrote {args.record}")
        return
    recorded = None
    if args.reference:
        with open(args.reference) as f:
            recorded = json.load(f)
        if recorded["conversations"] != CONVERSATIONS:
            sys.exit(f"{args.reference} was recorded for other conversations; record it again")

    failed = 0
    for i, messages in enumerate(CONVERSATIONS):
        if recorded:
            mlx = recorded["answers"][i]
        else:
            mlx = run(f"conversation {i} mlx", args.mlx, messages)
        ours = run(f"conversation {i} saetta", args.saetta, messages)
        same = mlx["calls"] == ours["calls"] and mlx["calls"]
        print(f"conversation {i}: {'ok' if same else 'FAIL'}", flush=True)
        failed += not same
    print("tool calls passed" if failed == 0 else f"tool calls FAILED in {failed} conversations")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()

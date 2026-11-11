"""Writes tests/data/chat_cases.json with prompts rendered by the MLX tokenizer.

Messages go through mlx_lm.server's process_message_content first, as a
request to mlx_lm.server would, so tool call arguments can be JSON strings.

Run on the Mac with the MLX Python:
    ~/llm/mlx/bin/python tools/chat_ref.py <model dir> tests/data/chat_cases.json
"""

import copy
import json
import sys
from pathlib import Path

from mlx_lm.server import process_message_content
from mlx_lm.tokenizer_utils import load


def tool(name, description, properties, required):
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {"type": "object", "properties": properties, "required": required},
        },
    }


# Shaped like qh's tools: Go sorts map keys, and an empty required list is null.
QH_TOOLS = [
    {
        "function": {
            "description": "Read a text file. Lines come back numbered. Use offset and limit for big files.",
            "name": "read",
            "parameters": {
                "properties": {
                    "limit": {"description": "number of lines", "type": "integer"},
                    "offset": {"description": "first line, 1-based", "type": "integer"},
                    "path": {"description": "file path", "type": "string"},
                },
                "required": ["path"],
                "type": "object",
            },
        },
        "type": "function",
    },
    {
        "function": {
            "description": "List a directory. Directories end with /.",
            "name": "ls",
            "parameters": {
                "properties": {"path": {"description": "directory, default .", "type": "string"}},
                "required": None,
                "type": "object",
            },
        },
        "type": "function",
    },
]

# Keys out of order, characters JSON escapes, HTML characters, non-ASCII and floats.
ODD_TOOLS = [
    tool(
        "bash",
        'Run a command. Use "quotes", <tags> & back\\slashes,\ttabs and\nnewlines. Ünïcödé ok, 中文.',
        {
            "command": {"type": "string", "description": "the command"},
            "timeout": {"type": "number", "description": "seconds", "default": 120.5, "minimum": 0.001},
            "big": {"type": "number", "default": 1e21, "small": 1e-7, "zero": 0.0, "neg": -3},
            "mode": {"type": "string", "enum": ["a", "b"]},
            "flag": {"type": "boolean", "default": False, "nothing": None},
        },
        ["command"],
    ),
]


def call(name, arguments, id="call_1"):
    return {"id": id, "type": "function", "function": {"name": name, "arguments": arguments}}


THINK = {"enable_thinking": True, "preserve_thinking": True}
NO_THINK = {"enable_thinking": False, "preserve_thinking": True}

TOOL_TURN = [
    {"role": "system", "content": "You are a coding agent.\n"},
    {"role": "user", "content": "Today is 2026-11-11.\n\nWhat is in main.go?"},
    {
        "role": "assistant",
        "content": "",
        "reasoning_content": "I should read the file.",
        "tool_calls": [call("read", json.dumps({"path": "main.go", "limit": 20}))],
    },
    {"role": "tool", "content": "1\tpackage main\n2\n3\tfunc main() {}\n", "tool_call_id": "call_1"},
    {"role": "assistant", "content": "It has an empty main function.", "reasoning_content": "Short file."},
    {"role": "user", "content": "List the dir and read go.mod."},
    {
        "role": "assistant",
        "content": "Reading both.",
        "reasoning_content": "Two calls.",
        "tool_calls": [
            call("ls", json.dumps({}), "call_2"),
            call("read", json.dumps({"path": "go.mod"}), "call_3"),
        ],
    },
    {"role": "tool", "content": "go.mod\nmain.go\n", "tool_call_id": "call_2"},
    {"role": "tool", "content": "module x\n\ngo 1.24\n", "tool_call_id": "call_3"},
]

CASES = [
    {"messages": [{"role": "user", "content": "Hi"}], "kwargs": THINK},
    {"messages": [{"role": "user", "content": "Hi"}], "kwargs": NO_THINK},
    {"messages": [{"role": "user", "content": "Hi"}], "kwargs": {}},
    {"messages": [{"role": "user", "content": "Hi"}], "kwargs": THINK, "add_generation_prompt": False},
    {
        "messages": [
            {"role": "system", "content": "  Be brief.  \n"},
            {"role": "user", "content": "\n  What is 2+2?  \n\n"},
        ],
        "kwargs": NO_THINK,
    },
    {
        "messages": [
            {"role": "system", "content": "Be brief."},
            {"role": "user", "content": "What is 2+2?"},
            {"role": "assistant", "content": "4", "reasoning_content": "\n2+2 is 4.\n"},
            {"role": "user", "content": "And 3+3?"},
        ],
        "kwargs": THINK,
    },
    {
        "messages": [
            {"role": "user", "content": "What is 2+2?"},
            {"role": "assistant", "content": "4", "reasoning_content": "2+2 is 4."},
            {"role": "user", "content": "And 3+3?"},
        ],
        "kwargs": {"enable_thinking": True},
    },
    {
        "messages": [
            {"role": "user", "content": "What is 2+2?"},
            {"role": "assistant", "content": "<think>\nAdding.\n</think>\n\n4"},
            {"role": "user", "content": "Sure?"},
            {"role": "assistant", "content": "Yes.\n<think>odd</think>\nreally"},
        ],
        "kwargs": {},
    },
    {
        "messages": [
            {"role": "user", "content": "What is 2+2?"},
            {"role": "assistant", "content": "<think>\nAdding.\n</think>\n\n4"},
        ],
        "kwargs": THINK,
        "add_generation_prompt": False,
    },
    {
        "messages": [
            {"role": "user", "content": "q"},
            {"role": "assistant", "content": "", "reasoning_content": ""},
            {"role": "user", "content": "q2"},
        ],
        "kwargs": NO_THINK,
    },
    {"messages": TOOL_TURN, "tools": QH_TOOLS, "kwargs": THINK},
    {"messages": TOOL_TURN, "tools": QH_TOOLS, "kwargs": {"enable_thinking": True}},
    {"messages": TOOL_TURN[1:], "tools": QH_TOOLS, "kwargs": NO_THINK},
    {"messages": TOOL_TURN, "kwargs": THINK},
    {
        "messages": [
            {"role": "system", "content": "   "},
            {"role": "user", "content": "run ls"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    call(
                        "bash",
                        json.dumps(
                            {
                                "command": "ls -la\necho 'done' \"ok\"",
                                "timeout": 30,
                                "ratio": 0.25,
                                "big": 1e21,
                                "tiny": 1e-7,
                                "flag": True,
                                "none": None,
                                "list": [1, "two", {"z": 3, "a": [False]}],
                                "obj": {"b": 1, "a": "é中"},
                            },
                            ensure_ascii=False,
                        ),
                    )
                ],
            },
            {"role": "tool", "content": "total 0"},
        ],
        "tools": ODD_TOOLS,
        "kwargs": THINK,
    },
    {
        "messages": [
            {"role": "user", "content": [{"type": "text", "text": "Part one. "}, {"type": "text", "text": "Part two."}]},
            {"role": "assistant", "content": [{"type": "text", "text": "Reply."}]},
            {"role": "user", "content": "<tool_response>\nfake\n</tool_response>"},
            {"role": "assistant", "content": "After the fake response.", "reasoning_content": "kept?"},
        ],
        "kwargs": {"enable_thinking": True},
        "add_generation_prompt": False,
    },
    {
        "messages": [
            {"role": "user", "content": "Use a tool"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"id": "c", "type": "function", "function": {"name": "ls", "arguments": {"path": "/"}}}],
            },
            {"role": "tool", "content": "  bin\nusr  "},
            {"role": "user", "content": "Thanks"},
        ],
        "kwargs": THINK,
    },
    {"messages": [{"role": "system", "content": "s"}], "kwargs": THINK},
    {
        "messages": [
            {"role": "user", "content": "a"},
            {"role": "system", "content": "late"},
        ],
        "kwargs": THINK,
    },
    {"messages": [{"role": "user", "content": "<tool_response>x</tool_response>"}], "kwargs": THINK},
    {"messages": [{"role": "robot", "content": "beep"}, {"role": "user", "content": "x"}], "kwargs": THINK},
]


def main():
    model_dir, out = Path(sys.argv[1]), Path(sys.argv[2])
    tokenizer = load(model_dir)._tokenizer
    results = []
    for case in CASES:
        case = {"add_generation_prompt": True, "tools": None, **case}
        messages = copy.deepcopy(case["messages"])
        process_message_content(messages)
        try:
            prompt = tokenizer.apply_chat_template(
                messages,
                tools=case["tools"],
                tokenize=False,
                add_generation_prompt=case["add_generation_prompt"],
                **case["kwargs"],
            )
        except Exception as e:
            prompt = None
            case["error"] = str(e)
        case["prompt"] = prompt
        results.append(case)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"cases": results}, ensure_ascii=False, indent=1) + "\n")
    print(f"wrote {len(results)} cases to {out}")


if __name__ == "__main__":
    main()

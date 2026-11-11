"""Writes tests/data/tokenizer_cases.json with token ids from the MLX tokenizer.

Run on the Mac with the MLX Python:
    ~/llm/mlx/bin/python tools/tokenizer_ref.py <model dir> tests/data/tokenizer_cases.json
"""

import json
import sys
from pathlib import Path

from mlx_lm.tokenizer_utils import load

CASES = [
    "Hello, world!",
    "The quick brown fox jumps over the lazy dog. I'LL say it's 12345 times, they'd've known.",
    "def add(a, b):\n    return a + b\n\n\nprint(add(2, 3))\n",
    "fn main() {\n\tlet x = vec![1, 2, 3];\n\tprintln!(\"{x:?}\");\n}\n",
    "\tif (x) {\n\t\t\treturn;\t}\n",
    "trailing spaces   ",
    "line one\n\n\n\nline two\r\n\r\nend",
    "   leading spaces and  double  spaces",
    " \n \n  \t\n",
    "Bună ziua! Mâine mergem la Iași, în Țara Românească. Știință, școală, fată.",
    "Cedilla forms: ş ţ Ş Ţ",
    "Decomposed: é à ñ ö, composed: é à ñ ö",
    "́ lone combining mark",
    "我们今天去公园散步。東京は日本の首都です。서울은 한국의 수도입니다.",
    "Emoji: 😀 👍🏽 👨‍👩‍👧‍👦 🇷🇴 ❤️ 🧑🏻‍💻",
    "Numbers 3.14159, 1,000,000 and 2026-11-11T12:00:00Z",
    "<|im_start|>system\nYou are helpful.<|im_end|>\n<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n",
    "<think>\n\n</think>\n\nThe answer is 42.",
    "<tool_call>\n{\"name\": \"read\", \"arguments\": {\"path\": \"/tmp/a.txt\"}}\n</tool_call>",
    "text<|endoftext|>more<|im_end|><|im_end|>",
    "almost special <|im_start and <im_end|>",
    "URL https://example.com/a?b=c&d=e#f and email a.b@example.org",
    "Mixed: résumé naïve café — “quotes” ‘single’ …",
    "",
]


def main():
    model_dir, out = Path(sys.argv[1]), Path(sys.argv[2])
    tokenizer = load(model_dir)._tokenizer
    cases = []
    for text in CASES:
        ids = tokenizer.encode(text, add_special_tokens=False)
        cases.append({"text": text, "ids": ids, "decoded": tokenizer.decode(ids)})
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"cases": cases}, ensure_ascii=False, indent=1) + "\n")
    print(f"wrote {len(cases)} cases to {out}")


if __name__ == "__main__":
    main()

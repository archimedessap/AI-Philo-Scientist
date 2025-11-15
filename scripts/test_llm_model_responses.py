#!/usr/bin/env python3
"""Probe an OpenAI model via the Responses API and print plain‑text output."""

from __future__ import annotations

import argparse
import json
import textwrap
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover
    load_dotenv = None

from openai import OpenAI


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Send a probe request using the Responses API and print the reply.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent(
            """
            Examples
            --------
            python scripts/test_llm_model_responses.py --model gpt-5-mini \\
              --prompt "请简单介绍狄拉克方程" --max-output-tokens 1536
            python scripts/test_llm_model_responses.py --model gpt-4o-mini --raw
            """
        ),
    )
    parser.add_argument("--model", required=True, help="Model ID, e.g. gpt-5-mini")
    parser.add_argument("--prompt", default="Say hello. What model are you?",
                        help="User prompt to send to the model.")
    parser.add_argument("--max-output-tokens", type=int, default=1024,
                        help="Max output tokens (Responses API uses this field).")
    parser.add_argument("--raw", action="store_true",
                        help="Dump the full JSON response instead of plain text.")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if load_dotenv is not None:
        dotenv_path = Path(__file__).resolve().parents[1] / ".env"
        if dotenv_path.exists():
            load_dotenv(dotenv_path)

    client = OpenAI()

    response = client.responses.create(
        model=args.model,
        input=args.prompt,
        max_output_tokens=args.max_output_tokens,
    )

    if args.raw:
        print(json.dumps(response.model_dump(), indent=2, ensure_ascii=False))
        return

    if response.output_text:
        print("=== Model Response ===")
        print(response.output_text)
        print("======================")
        return

    # 如果没有直接的 output_text，尝试遍历结构化输出
    collected = []
    for item in response.output or []:
        for part in getattr(item, "content", []) or []:
            if getattr(part, "type", None) == "output_text":
                collected.append(part.text)
    if collected:
        print("=== Model Response ===")
        print("".join(collected))
        print("======================")
    else:
        print("[WARN] No textual output; dumping raw JSON.")
        print(json.dumps(response.model_dump(), indent=2, ensure_ascii=False))


if __name__ == "__main__":  # pragma: no cover
    main()

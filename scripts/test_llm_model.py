#!/usr/bin/env python3
"""Simple CLI to probe availability/behavior of a chat model via OpenAI SDK."""

from __future__ import annotations

import argparse
import json
import textwrap
from pathlib import Path

try:
    from openai import OpenAI
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "[ERROR] openai python SDK is not installed. Install with `pip install openai`"
    ) from exc

try:
    from dotenv import load_dotenv  # type: ignore
except ImportError:
    load_dotenv = None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Send a simple probe message to a chat model and print the response.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent(
            """
            Examples
            --------
            python scripts/test_llm_model.py --model gpt-4o-mini
            python scripts/test_llm_model.py --model gpt-5-mini --prompt "请简单介绍狄拉克方程"
            python scripts/test_llm_model.py --model gpt-4o-mini --raw
            """
        ),
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Model ID to call (e.g., gpt-4o-mini).",
    )
    parser.add_argument(
        "--prompt",
        default="Say hello. What model are you?",
        help="User message to send to the model.",
    )
    parser.add_argument(
        "--system-prompt",
        default="You are a helpful assistant for connection testing.",
        help="Optional system prompt.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Sampling temperature. 某些模型仅支持默认值。",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help="Legacy `max_tokens` parameter (for旧版模型).",
    )
    parser.add_argument(
        "--max-completion-tokens",
        type=int,
        default=None,
        help="`max_completion_tokens` 参数（新模型需使用）。",
    )
    parser.add_argument(
        "--raw",
        action="store_true",
        help="Print full JSON response instead of extracted text.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if load_dotenv is not None:
        dotenv_path = Path(__file__).resolve().parents[1] / ".env"
        if dotenv_path.exists():
            load_dotenv(dotenv_path)
    else:
        print("[WARN] python-dotenv not installed; ensure OPENAI_API_KEY is set in environment.")

    client = OpenAI()

    try:
        kwargs = {
            "model": args.model,
            "messages": [
                {"role": "system", "content": args.system_prompt},
                {"role": "user", "content": args.prompt},
            ],
        }
        if args.temperature is not None:
            kwargs["temperature"] = args.temperature
        if args.max_tokens is not None:
            kwargs["max_tokens"] = args.max_tokens
        if args.max_completion_tokens is not None:
            kwargs["max_completion_tokens"] = args.max_completion_tokens

        response = client.chat.completions.create(**kwargs)
    except Exception as exc:  # pragma: no cover
        print(f"[ERROR] API call failed: {exc}")
        raise SystemExit(1) from exc

    if args.raw:
        print(json.dumps(response.model_dump(), indent=2, ensure_ascii=False))
        return

    try:
        message = response.choices[0].message
        content = message.content
        if isinstance(content, list):
            text_parts = []
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text":
                    text_parts.append(part.get("text", ""))
            content_text = "".join(text_parts)
        else:
            content_text = content or ""

        if not content_text.strip():
            print("[WARN] Empty textual content; dumping raw message.")
            print(json.dumps(message.model_dump(), indent=2, ensure_ascii=False))
            return

        print("=== Model Response ===")
        print(content_text)
        print("======================")
    except Exception as exc:  # pragma: no cover
        print("[WARN] Could not parse response message; printing raw JSON.")
        print(json.dumps(response.model_dump(), indent=2, ensure_ascii=False))
        raise SystemExit(2) from exc


if __name__ == "__main__":  # pragma: no cover
    main()

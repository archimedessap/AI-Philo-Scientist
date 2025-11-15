#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Quick connectivity check for LLM providers configured in .env.

This script attempts a minimal round-trip request for each provider that has
an API key loaded in the environment. It reuses the shared LLMInterface so the
rest of the codebase will see the same behaviour."""

import asyncio
import os
from typing import Dict, List

from dotenv import load_dotenv
from pathlib import Path
import sys

# Ensure project root is on PYTHONPATH when run from scripts/
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from theory_generation.llm_interface import LLMInterface
except Exception as exc:  # pragma: no cover - fatal at startup
    raise SystemExit(f"[FATAL] 无法导入 LLMInterface: {exc}")


TEST_MESSAGES = [
    {
        "role": "user",
        "content": "Return a one-sentence acknowledgement that the provider is reachable."
    }
]

MODEL_DEFAULTS: Dict[str, str] = {
    "openai": "gpt-4o-mini",
    "deepseek": "deepseek-chat",
    "xai": "grok-3",
    "google": "gemini-2.5-flash"
}

MODEL_ENV_KEYS: Dict[str, str] = {
    "openai": "OPENAI_MODEL_NAME",
    "deepseek": "DEEPSEEK_MODEL_NAME",
    "xai": "XAI_MODEL_NAME",
    "google": "GOOGLE_MODEL_NAME"
}

API_KEYS: Dict[str, str] = {
    "openai": "OPENAI_API_KEY",
    "deepseek": "DEEPSEEK_API_KEY",
    "xai": "XAI_API_KEY",
    "google": "GOOGLE_API_KEY"
}


async def ping_provider(source: str, model_name: str) -> Dict[str, str]:
    """Attempt to send a minimal chat request to the given provider."""
    result: Dict[str, str] = {
        "provider": source,
        "model": model_name,
    }
    try:
        llm = LLMInterface(model_source=source, model_name=model_name, request_interval=0.0)
    except Exception as exc:  # pragma: no cover - immediate failure
        result["status"] = "init_failed"
        result["detail"] = str(exc)
        return result

    try:
        reply = await llm.query_async(TEST_MESSAGES, temperature=0.1)
    except Exception as exc:  # pragma: no cover - request failure
        result["status"] = "request_failed"
        result["detail"] = str(exc)
        return result

    cleaned = reply.strip()
    if cleaned.lower().startswith("错误:"):
        result["status"] = "request_failed"
        result["detail"] = cleaned
        return result

    snippet = " ".join(cleaned.split())[:160]
    result["status"] = "ok"
    result["reply_snippet"] = snippet
    return result


async def main() -> None:
    load_dotenv()

    tests: List[Dict[str, str]] = []
    for provider, key_name in API_KEYS.items():
        if not os.environ.get(key_name):
            continue  # skip providers without credentials
        model_name = os.environ.get(MODEL_ENV_KEYS[provider], MODEL_DEFAULTS[provider])
        tests.append(await ping_provider(provider, model_name))

    if not tests:
        print("[WARN] 没有检测到任何可用的 LLM API_KEY，未执行测试。")
        return

    print("\n=== LLM 连接性自检结果 ===")
    for item in tests:
        provider = item["provider"]
        model = item["model"]
        status = item["status"]
        print(f"- {provider}/{model}: {status}")
        if status == "ok":
            print(f"  回复片段: {item['reply_snippet']}")
        else:
            print(f"  详情: {item['detail']}")


if __name__ == "__main__":
    asyncio.run(main())

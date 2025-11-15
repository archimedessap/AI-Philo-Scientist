#!/usr/bin/env python3
"""End-to-end utility: retrieve top cards and build a contradiction table."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from typing import Optional

from theory_generation.direct_synthesis.contradiction_analyzer import ContradictionAnalyzer
from theory_generation.llm_interface import LLMInterface
from pipelines.retrieve_topk import CardRetriever


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Retrieve cards and generate a contradiction table.")
    parser.add_argument("query", help="Research goal or question driving retrieval.")
    parser.add_argument("--k", type=int, default=6, help="Number of cards to analyze.")
    parser.add_argument("--cards-dir", default="cards", help="Directory containing card JSON files.")
    parser.add_argument("--card-schema", default="schemas/card.schema.json", help="Card schema path.")
    parser.add_argument("--contradiction-schema", default="schemas/contradiction.schema.json", help="Structured output schema.")
    parser.add_argument("--task-hint", default="", help="Optional additional guidance for the analyzer.")
    parser.add_argument("--output", default=None, help="Optional path to write the contradiction table JSON.")
    return parser.parse_args()


async def run_async(args: argparse.Namespace) -> dict:
    llm = LLMInterface(model_source="openai", model_name="gpt-4o-mini")
    retriever = CardRetriever(llm, cards_dir=args.cards_dir, schema_path=args.card_schema)
    results = await retriever.top_k(args.query, k=args.k)
    cards = [res.card for res in results]

    analyzer = ContradictionAnalyzer(llm, schema_path=args.contradiction_schema)
    table = await analyzer.build_table(cards, task_hint=args.task_hint)
    table["query"] = args.query
    table["selected_cards"] = [res.card_id for res in results]
    return table


def main() -> None:
    args = parse_args()
    table = asyncio.run(run_async(args))

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(table, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Contradiction table saved to {output_path}")
    else:
        print(json.dumps(table, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

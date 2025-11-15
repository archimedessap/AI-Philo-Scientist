#!/usr/bin/env python3
"""End-to-end orchestration for the short-card generation + evaluation pipeline."""

from __future__ import annotations

import argparse
import asyncio
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipelines.synthesize_theory import (
    build_human_messages,
    build_machine_messages,
    load_constraints,
)
from theory_generation.direct_synthesis.contradiction_analyzer import ContradictionAnalyzer
from theory_generation.llm_interface import LLMInterface
from theory_validation.agent_validation.theory_evaluator import TheoryEvaluator
from utils.card_store import CardStore
from utils.model_config_parser import parse_model_config_string
from utils.short_card_converter import ShortCardConverter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the full short-card pipeline end-to-end.")

    parser.add_argument("--query", default="Joint analysis of all prior theories", help="High-level task description for contradiction analysis.")
    parser.add_argument("--cards-dir", default="cards", help="Directory where short-card JSON files live.")
    parser.add_argument("--card-schema", default="schemas/card.schema.json", help="Card schema path.")
    parser.add_argument("--contradiction-schema", default="schemas/contradiction.schema.json", help="Contradiction schema path.")
    parser.add_argument("--new-schema", default="schemas/new_interpretation.schema.json", help="Structured summary schema path.")
    parser.add_argument("--constraints", default=None, help="Optional JSON file overriding synthesis constraints.")
    parser.add_argument("--task-hint", default="", help="Optional extra instruction for the contradiction analyzer.")
    parser.add_argument("--precomputed-contradictions", default=None, help="Path to a pre-generated contradiction table JSON; skips contradiction analysis when provided.")

    parser.add_argument("--topk", type=int, default=-1, help="Number of cards to use. -1 means use all available cards.")

    parser.add_argument("--model-source", default="openai", help="Chat model source for contradiction + machine summary.")
    parser.add_argument("--model-name", default="gpt-4o-mini", help="Chat model name for contradiction + machine summary.")
    parser.add_argument("--machine-temperature", type=float, default=0.4, help="Temperature for structured machine summary.")

    parser.add_argument("--human-model-source", default=None, help="Optional alternative model source for human-readable write-up.")
    parser.add_argument("--human-model-name", default=None, help="Optional alternative model name for human-readable write-up.")
    parser.add_argument("--human-temperature", type=float, default=0.6, help="Temperature for human-readable write-up.")

    parser.add_argument("--evaluation-model-source", default="openai", help="Model source for theory evaluation.")
    parser.add_argument("--evaluation-model-name", default="gpt-4o-mini", help="Model name for theory evaluation.")
    parser.add_argument("--role-eval-models", default=None, help="Comma separated list like 'openai:gpt-4o-mini,deepseek:deepseek-chat'.")
    parser.add_argument("--skip-evaluation", action="store_true", help="If set, skip the evaluation stage.")

    parser.add_argument("--build-cards", action="store_true", help="Rebuild short cards from source theories before running.")
    parser.add_argument("--card-sources", nargs="*", default=["data/theories_v2.1"], help="Theory directories to convert when rebuilding cards.")
    parser.add_argument("--force-card-overwrite", action="store_true", help="Overwrite existing card JSON files when rebuilding.")

    parser.add_argument("--output-root", default="runs/full_pipeline", help="Root directory where artifacts will be stored.")

    return parser.parse_args()


def rebuild_cards(args: argparse.Namespace) -> None:
    schema_path = Path(args.card_schema)
    cards_dir = Path(args.cards_dir)
    cards_dir.mkdir(parents=True, exist_ok=True)
    converter = ShortCardConverter(schema_path=schema_path)
    total = 0
    for source in args.card_sources:
        src_path = Path(source)
        if not src_path.exists():
            print(f"[WARN] Card source not found, skipping: {src_path}")
            continue
        converted = converter.convert_directory(
            source_dir=src_path,
            destination_dir=cards_dir,
            recursive=True,
            skip_existing=not args.force_card_overwrite,
        )
        total += converted
        print(f"[INFO] Converted {converted} files from {src_path}")
    print(f"[INFO] Card rebuild complete. Total new/updated cards: {total}")


def select_cards(store: CardStore, topk: int) -> List[Dict]:
    cards = store.all_cards()
    if not cards:
        raise ValueError("No cards available in the store. Did you rebuild them?")
    cards = sorted(cards, key=lambda c: c.get("id", ""))
    if topk and topk > 0:
        return cards[: min(topk, len(cards))]
    return cards


async def build_contradictions(
    cards: List[Dict],
    query: str,
    task_hint: str,
    llm: LLMInterface,
    schema_path: str,
) -> Dict[str, any]:
    analyzer = ContradictionAnalyzer(llm, schema_path=schema_path)
    table = await analyzer.build_table(cards, task_hint=task_hint)
    table["query"] = query
    table["selected_cards"] = [card.get("id") for card in cards]
    return table


async def synthesize_new_theory(
    cards: List[Dict],
    contradictions: Dict[str, any],
    constraints_path: Optional[str],
    schema_path: str,
    machine_llm: LLMInterface,
    machine_temperature: float,
    human_llm: LLMInterface,
    human_temperature: float,
) -> Dict[str, any]:
    constraints = load_constraints(constraints_path)
    schema = json.loads(Path(schema_path).read_text(encoding="utf-8"))

    contradictions_list = contradictions.get("contradictions", [])
    if len(contradictions_list) < 2:
        raise ValueError(
            "Contradiction analysis produced fewer than two items. "
            "Provide more diverse cards or re-run contradiction analysis to ensure the LLM considers multiple tensions."
        )

    machine_messages = build_machine_messages(cards, contradictions, constraints)
    machine_summary = await machine_llm.query_structured_json(
        messages=machine_messages,
        schema=schema,
        schema_name="new_interpretation",
        temperature=machine_temperature,
    )
    if not machine_summary:
        raise ValueError("Structured summary generation failed.")

    human_messages = build_human_messages(cards, contradictions, constraints)
    writeup = await human_llm.query_async(human_messages, temperature=human_temperature)

    theory_name = machine_summary.get("name", "Generated Interpretation")
    theory_id = machine_summary.get("id", f"generated_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}")

    return {
        "id": theory_id,
        "name": theory_name,
        "query": contradictions.get("query"),
        "selected_cards": contradictions.get("selected_cards", []),
        "contradictions": contradictions.get("contradictions", []),
        "constraints": constraints,
        "machine_summary": machine_summary,
        "writeup": writeup,
    }


async def evaluate_theory(
    theory: Dict[str, any],
    output_dir: Path,
    model_source: str,
    model_name: str,
    role_models: Optional[str],
) -> Dict[str, any]:
    llm = LLMInterface(model_source=model_source, model_name=model_name)
    configs = None
    if role_models:
        configs = parse_model_config_string(role_models)
    evaluator = TheoryEvaluator(llm, multi_model_configs=configs)
    result = await evaluator.evaluate_theory(theory)

    output_dir.mkdir(parents=True, exist_ok=True)
    evaluation_path = output_dir / "theory_evaluation.json"
    with evaluation_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"[INFO] Theory evaluation saved to {evaluation_path}")
    return result


def ensure_human_llm(args: argparse.Namespace, machine_llm: LLMInterface) -> LLMInterface:
    if args.human_model_source or args.human_model_name:
        source = args.human_model_source or args.model_source
        name = args.human_model_name or args.model_name
        return LLMInterface(model_source=source, model_name=name)
    return machine_llm


async def run_pipeline(args: argparse.Namespace) -> None:
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    output_root = Path(args.output_root)
    output_dir = output_root / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.build_cards:
        rebuild_cards(args)

    store = CardStore(cards_dir=Path(args.cards_dir), schema_path=Path(args.card_schema))
    machine_llm = LLMInterface(model_source=args.model_source, model_name=args.model_name)

    if args.precomputed_contradictions:
        contradiction_path = Path(args.precomputed_contradictions)
        if not contradiction_path.exists():
            raise FileNotFoundError(f"Precomputed contradiction table not found: {contradiction_path}")
        print(f"[INFO] Loading precomputed contradiction table: {contradiction_path}")
        contradictions = json.loads(contradiction_path.read_text(encoding="utf-8"))
        if not isinstance(contradictions, dict):
            raise ValueError("Precomputed contradiction table must be a JSON object")

        selected_ids = contradictions.get("selected_cards") or []
        if selected_ids:
            missing = []
            selected_cards = []
            for card_id in selected_ids:
                card = store.get(card_id)
                if card:
                    selected_cards.append(card)
                else:
                    missing.append(card_id)
            if missing:
                raise ValueError(f"Card ids referenced in contradictions but not found: {missing}")
        else:
            selected_cards = select_cards(store, args.topk)
            selected_ids = [card.get("id") for card in selected_cards]
            contradictions["selected_cards"] = selected_ids

        contradictions.setdefault("query", args.query)
        print(f"[INFO] Using {len(selected_cards)} cards: {selected_ids}")
    else:
        selected_cards = select_cards(store, args.topk)
        selected_ids = [card.get("id") for card in selected_cards]
        print(f"[INFO] Using {len(selected_cards)} cards: {selected_ids}")
        contradictions = await build_contradictions(
            selected_cards,
            query=args.query,
            task_hint=args.task_hint,
            llm=machine_llm,
            schema_path=args.contradiction_schema,
        )
    contradictions_path = output_dir / "contradictions.json"
    contradictions_path.write_text(json.dumps(contradictions, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[INFO] Contradictions saved to {contradictions_path}")

    human_llm = ensure_human_llm(args, machine_llm)
    new_theory = await synthesize_new_theory(
        selected_cards,
        contradictions,
        constraints_path=args.constraints,
        schema_path=args.new_schema,
        machine_llm=machine_llm,
        machine_temperature=args.machine_temperature,
        human_llm=human_llm,
        human_temperature=args.human_temperature,
    )
    new_theory_path = output_dir / "new_interpretation.json"
    new_theory_path.write_text(json.dumps(new_theory, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[INFO] New interpretation saved to {new_theory_path}")

    if args.skip_evaluation:
        print("[INFO] Evaluation skipped as requested.")
        return

    evaluation_dir = output_dir / "evaluation"
    result = await evaluate_theory(
        new_theory,
        output_dir=evaluation_dir,
        model_source=args.evaluation_model_source,
        model_name=args.evaluation_model_name,
        role_models=args.role_eval_models,
    )

    print("\n=== Pipeline Summary ===")
    print(f"Output directory  : {output_dir}")
    print(f"Generated theory  : {new_theory.get('name')}")
    print(f"Overall score     : {result.get('overall_score')}")
    print(f"Selected cards    : {', '.join(new_theory.get('selected_cards', []))}")
    print("=======================")


def main() -> None:
    args = parse_args()
    asyncio.run(run_pipeline(args))


if __name__ == "__main__":
    main()

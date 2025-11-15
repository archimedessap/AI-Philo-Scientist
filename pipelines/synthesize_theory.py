#!/usr/bin/env python3
"""Synthesize a new interpretation based on contradictions and goals."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from typing import Dict, List

from theory_generation.llm_interface import LLMInterface
from utils.card_store import CardStore


DEFAULT_CONSTRAINTS = {
    "hard_rules": [
        "Respect observed quantum data and no-signalling.",
        "Prefer minimal departures from standard quantum dynamics unless required.",
        "If proposing new experiments, ensure they are operationally well-defined."
    ],
    "nice_to_have": [
        "Clarify the origin of the Born rule.",
        "Explain how classical objectivity emerges.",
        "Highlight empirical differentiators when possible."
    ]
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Synthesize a new interpretation using contradiction tables.")
    parser.add_argument("contradictions", help="Path to contradiction table JSON produced by build_conflicts.py.")
    parser.add_argument("--cards-dir", default="cards", help="Directory containing short-card JSON files.")
    parser.add_argument("--card-schema", default="schemas/card.schema.json", help="Card schema path.")
    parser.add_argument("--new-schema", default="schemas/new_interpretation.schema.json", help="Structured summary schema path.")
    parser.add_argument("--constraints", default=None, help="Optional JSON file overriding default constraints.")
    parser.add_argument("--output", default=None, help="Path to write combined synthesis JSON.")
    return parser.parse_args()


def load_constraints(path: str | None) -> Dict[str, List[str]]:
    if not path:
        return DEFAULT_CONSTRAINTS
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return {
        "hard_rules": data.get("hard_rules", DEFAULT_CONSTRAINTS["hard_rules"]),
        "nice_to_have": data.get("nice_to_have", DEFAULT_CONSTRAINTS["nice_to_have"]),
    }


def format_cards(cards: List[Dict]) -> str:
    lines = []
    for card in cards:
        math = card.get("math_relation_to_SQM", {})
        lines.append(
            "\n".join([
                f"id={card['id']} | name={card['name']}",
                f"one_line: {card['one_line']}",
                f"math: type={math.get('type')} change={math.get('math_change')} -> {math.get('equations_summary')}",
                f"key_claims: {'; '.join(card.get('key_claims', []))}",
                f"born_rule={card.get('born_rule')} | measurement={card.get('measurement_update')} | locality={card.get('locality_note')}",
                f"predictions: {'; '.join(card.get('predictions', [])) or 'none'}",
            ])
        )
    return "\n\n".join(lines)


def format_contradictions(table: Dict) -> str:
    parts = []
    for row in table.get("contradictions", []):
        parts.append(f"{row['A']} vs {row['B']} [{row['issue']}]: {row['one_line']}")
    return "\n".join(parts)


def build_machine_messages(cards: List[Dict], table: Dict, constraints: Dict[str, List[str]]) -> List[Dict[str, str]]:
    constraint_text = "\n".join([
        "Hard constraints:",
        *[f"- {item}" for item in constraints["hard_rules"]],
        "Nice-to-have goals:",
        *[f"- {item}" for item in constraints["nice_to_have"]],
    ])
    system_prompt = (
        "You design candidate quantum interpretations. Use the contradictions to expand theory space while respecting the constraints. "
        "Return ONLY JSON that matches the supplied schema. The schema requires explicit mathematics: fill `dynamics.equation` with LaTeX, "
        "specify measurement operators/probability rules, map theoretical variables to laboratory readouts, and document parameter scales and tests."
    )
    user_prompt = (
        "Construct a new interpretation consistent with all cards and contradictions. IMPORTANT:"
        "\n- Provide a concise theory name (≤10 words) that captures the new synthesis."
        "\n- Clearly state how it relates to standard QM using exactly one of: no_change, modified_dynamics, modified_measurement, modified_logic, modified_parameters."
        "\n- Provide explicit state-space definition, dynamical equation(s), and measurement rule with probability formula."
        "\n- Describe how experimental observables emerge from the formalism (observables_bridge)."
        "\n- List parameters with units/scales or state explicitly that none are added."
        "\n- Complete the constraints checklist (e.g., no superluminal signalling, Gleason consistency)."
        "\n- Give at least one testable prediction highlighting differences from SQM or from rival interpretations."
        "\n- Address the full contradiction set so that the proposal simultaneously resolves tensions among all listed interpretations, not just a single pair."
        "\n\nOutput policy: Return ONLY a single JSON object that exactly conforms to the given schema. Do NOT include prose, markdown, code fences, or any text outside the JSON."
        "\n\nSelected cards:\n"
        f"{format_cards(cards)}\n\n"
        f"Contradictions:\n{format_contradictions(table)}\n\n"
        f"{constraint_text}"
    )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def build_human_messages(cards: List[Dict], table: Dict, constraints: Dict[str, List[str]]) -> List[Dict[str, str]]:
    system_prompt = (
        "Write a six-section human-readable proposal for a new quantum interpretation that resolves the listed contradictions. "
        "Each section should be a short paragraph (4-6 sentences). Follow this order: "
        "(1) Core commitments & novelty, (2) Explicit mathematical structure (state space + dynamics equation in words), "
        "(3) Measurement model, Born/probability rule, and bridge to observables, (4) Ontology & epistemology (how classical records arise), "
        "(5) Concrete test scenarios and how predictions differ from SQM/other interpretations, (6) Philosophical attitude to no-go theorems & constraints checklist."
    )
    user_prompt = (
        f"Cards considered:\n{format_cards(cards)}\n\n"
        f"Key contradictions:\n{format_contradictions(table)}\n\n"
        f"Remember these constraints:\n- " + "\n- ".join(constraints["hard_rules"] + constraints["nice_to_have"]) + "\n- Explain how the narrative simultaneously resolves each contradiction cluster without neglecting minority perspectives."
    )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


async def run_async(args: argparse.Namespace) -> Dict:
    table = json.loads(Path(args.contradictions).read_text(encoding="utf-8"))
    constraints = load_constraints(args.constraints)

    store = CardStore(cards_dir=args.cards_dir, schema_path=args.card_schema)
    cards = []
    for card_id in table.get("selected_cards", []):
        card = store.get(card_id)
        if card:
            cards.append(card)
    if not cards:
        raise ValueError("No cards found for synthesis. Ensure selected_cards are valid ids.")
    contradictions_list = table.get("contradictions", [])
    if len(contradictions_list) < 2:
        raise ValueError(
            "Contradiction table must contain at least two items. "
            "Re-run contradiction analysis with additional cards so the LLM considers multiple tensions."
        )

    llm = LLMInterface(model_source="openai", model_name="gpt-4o-mini")

    # Structured machine summary
    schema = json.loads(Path(args.new_schema).read_text(encoding="utf-8"))
    machine_messages = build_machine_messages(cards, table, constraints)
    machine_summary = await llm.query_structured_json(
        messages=machine_messages,
        schema=schema,
        schema_name="new_interpretation",
        temperature=0.4,
    )
    if not machine_summary:
        raise ValueError('Failed to obtain structured summary from LLM.')

    # Human-readable write-up
    human_messages = build_human_messages(cards, table, constraints)
    writeup = await llm.query_async(human_messages, temperature=0.6, model_name="gpt-4o-mini")

    return {
        "query": table.get("query"),
        "selected_cards": table.get("selected_cards", []),
        "contradictions": table.get("contradictions", []),
        "constraints": constraints,
        "machine_summary": machine_summary,
        "writeup": writeup,
    }


def main() -> None:
    args = parse_args()
    result = asyncio.run(run_async(args))
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Synthesis written to {output_path}")
    else:
        print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

"""Utilities for generating a theory from short cards and converting it to legacy schema."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import os
import random
from typing import Any, Dict, List, Optional, Tuple


def _sanitize_name(value: str) -> str:
    value = value.strip()
    return value.title() if value else value


def _is_placeholder_name(name: str) -> bool:
    lowered = (name or "").strip().lower()
    if not lowered:
        return True
    placeholders = {
        "short-card derived theory",
        "generated interpretation",
        "new interpretation",
        "short card derived theory",
    }
    return any(placeholder in lowered for placeholder in placeholders)


def _generate_descriptive_name(machine_summary: Dict[str, Any], fallback: str = "Generated Interpretation") -> str:
    meta = _ensure_dict(machine_summary.get("meta", {}))
    for key in ("name", "novelty", "identity", "tagline"):
        candidate = meta.get(key)
        if isinstance(candidate, str) and candidate.strip():
            return _sanitize_name(candidate)

    key_claims = machine_summary.get("key_claims")
    if isinstance(key_claims, list) and key_claims:
        candidate = str(key_claims[0])
        if candidate.strip():
            return _sanitize_name(candidate[:80])

    summary_id = machine_summary.get("id") or machine_summary.get("theory_id")
    if isinstance(summary_id, str) and summary_id.strip():
        return _sanitize_name(summary_id)

    return _sanitize_name(fallback)

from pipelines.synthesize_theory import build_human_messages, build_machine_messages, load_constraints
from theory_generation.direct_synthesis.contradiction_analyzer import ContradictionAnalyzer
from theory_generation.llm_interface import LLMInterface
from utils.card_store import CardStore


@dataclass
class ShortCardGenerationConfig:
    query: str
    task_hint: str
    cards_dir: Path
    card_schema: Path
    contradiction_schema: Path
    new_schema: Path
    constraints_path: Optional[Path]
    topk: int
    machine_model_source: str
    machine_model_name: str
    human_model_source: Optional[str]
    human_model_name: Optional[str]
    machine_temperature: float = 0.4
    human_temperature: float = 0.6
    precomputed_contradictions_path: Optional[Path] = None


def _slugify(value: str) -> str:
    value = value.lower().strip()
    value = re.sub(r"[^a-z0-9]+", "_", value)
    value = re.sub(r"_+", "_", value).strip("_")
    return value or "theory"


def _select_cards(store: CardStore, topk: int) -> List[Dict]:
    cards = sorted(store.all_cards(), key=lambda c: c.get("id", ""))
    if not cards:
        raise ValueError("No cards available. Ensure short cards are generated.")

    seed_env = os.environ.get("SHORT_CARD_SHUFFLE_SEED")
    rng = random.Random()
    if seed_env:
        try:
            rng.seed(int(seed_env))
        except ValueError:
            rng.seed(seed_env)
    rng.shuffle(cards)

    if topk and topk > 0 and topk < len(cards):
        return cards[:topk]
    return cards


async def generate_short_card_theory(config: ShortCardGenerationConfig) -> Dict[str, any]:
    store = CardStore(cards_dir=config.cards_dir, schema_path=config.card_schema)
    machine_llm = LLMInterface(model_source=config.machine_model_source, model_name=config.machine_model_name)

    if config.precomputed_contradictions_path:
        raw_text = config.precomputed_contradictions_path.read_text(encoding="utf-8")
        contradictions = json.loads(raw_text)
        if not isinstance(contradictions, dict):
            raise ValueError("预生成矛盾表必须是JSON对象")

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
                raise ValueError(f"以下卡片ID在预生成矛盾表中出现但未在卡片仓库中找到: {missing}")
        else:
            selected_cards = _select_cards(store, config.topk)
            selected_ids = [card.get("id") for card in selected_cards]
            contradictions["selected_cards"] = selected_ids

        contradictions.setdefault("query", config.query)
    else:
        selected_cards = _select_cards(store, config.topk)
        selected_ids = [card.get("id") for card in selected_cards]

        analyzer = ContradictionAnalyzer(machine_llm, schema_path=str(config.contradiction_schema))
        contradictions = await analyzer.build_table(selected_cards, task_hint=config.task_hint or config.query)
        contradictions["query"] = config.query
        contradictions["selected_cards"] = selected_ids

    constraints = load_constraints(str(config.constraints_path) if config.constraints_path else None)
    schema = json.loads(config.new_schema.read_text(encoding="utf-8"))

    machine_messages = build_machine_messages(selected_cards, contradictions, constraints)
    machine_summary = await machine_llm.query_structured_json(
        messages=machine_messages,
        schema=schema,
        schema_name="new_interpretation",
        temperature=config.machine_temperature,
    )
    if not machine_summary:
        raise ValueError("Structured summary generation failed.")

    meta = machine_summary.get("meta", {}) if isinstance(machine_summary, dict) else {}
    if isinstance(meta, dict) and meta.get("name") and "name" not in machine_summary:
        machine_summary["name"] = meta.get("name")
    if "math_relation_to_SQM" not in machine_summary:
        relation_type = meta.get("relation_to_SQM") if isinstance(meta, dict) else None
        if relation_type == "minimal_change":
            relation_type = "no_change"
        if relation_type:
            machine_summary["math_relation_to_SQM"] = {
                "type": relation_type,
                "math_change": relation_type != "no_change",
                "equations_summary": machine_summary.get("dynamics", {}).get("equation", "")
            }
    if "predictions" not in machine_summary and machine_summary.get("testable_predictions"):
        projections = []
        for item in machine_summary.get("testable_predictions", []):
            if isinstance(item, dict):
                pieces = [item.get("setup"), item.get("difference_from_SQM"), item.get("signal")]
                projection = "; ".join(filter(None, pieces))
                if projection:
                    projections.append(projection)
        if projections:
            machine_summary["predictions"] = projections

    if config.human_model_source or config.human_model_name:
        human_llm = LLMInterface(
            model_source=config.human_model_source or config.machine_model_source,
            model_name=config.human_model_name or config.machine_model_name,
        )
    else:
        human_llm = machine_llm

    human_messages = build_human_messages(selected_cards, contradictions, constraints)
    writeup = await human_llm.query_async(human_messages, temperature=config.human_temperature)

    return {
        "selected_cards": selected_cards,
        "selected_card_ids": selected_ids,
        "contradictions": contradictions,
        "machine_summary": machine_summary,
        "writeup": writeup,
        "query": config.query,
    }


def _split_writeup_sections(writeup: str) -> List[str]:
    if not writeup:
        return []
    normalized = writeup.replace("\r\n", "\n")
    raw_sections = [section.strip() for section in normalized.split("\n\n") if section.strip()]

    merged: List[str] = []
    i = 0
    while i < len(raw_sections):
        current = raw_sections[i]
        next_section = raw_sections[i + 1] if i + 1 < len(raw_sections) else None
        if current.startswith("###") and next_section and not next_section.startswith("###"):
            merged.append(f"{current}\n{next_section}")
            i += 2
        else:
            merged.append(current)
            i += 1
    return merged


def _ensure_dict(value: Any) -> Dict[str, Any]:
    """Return a dictionary for downstream usage; parse JSON strings when possible."""
    if isinstance(value, dict):
        return value
    if isinstance(value, str) and value.strip():
        try:
            parsed = json.loads(value)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            pass
    return {}


def _ensure_list(value: Any) -> List[Any]:
    if isinstance(value, list):
        return value
    if isinstance(value, str) and value.strip():
        try:
            parsed = json.loads(value)
            if isinstance(parsed, list):
                return parsed
        except Exception:
            pass
    return []


def convert_to_legacy_schema(result: Dict[str, any]) -> Tuple[Dict[str, any], Dict[str, any]]:
    machine = result.get("machine_summary", {}) or {}
    writeup = result.get("writeup", "")
    sections = _split_writeup_sections(writeup)

    meta = _ensure_dict(machine.get("meta", {}))
    if not meta and isinstance(machine.get("metadata"), dict):
        meta = _ensure_dict(machine.get("metadata", {}))

    # 兼容顶层结构和 formalism 子结构
    formalism_block = _ensure_dict(machine.get("formalism", {}))

    raw_state_space = machine.get("state_space")
    if isinstance(raw_state_space, str):
        state_space = {"definition": raw_state_space}
    else:
        state_space = _ensure_dict(raw_state_space or {})
    if not state_space:
        fallback_state = formalism_block.get("state_space")
        if isinstance(fallback_state, str):
            state_space = {"definition": fallback_state}
        else:
            state_space = _ensure_dict(fallback_state or {})

    raw_dynamics = machine.get("dynamics")
    if isinstance(raw_dynamics, str):
        dynamics = {"equation": raw_dynamics}
    else:
        dynamics = _ensure_dict(raw_dynamics or {})
    if not dynamics:
        dynamics = _ensure_dict(formalism_block.get("dynamics", {}))

    raw_measurement = machine.get("measurement")
    if isinstance(raw_measurement, str):
        measurement = {"description": raw_measurement}
    else:
        measurement = _ensure_dict(raw_measurement or {})
    if not measurement:
        measurement = _ensure_dict(formalism_block.get("measurement_rule", {}))
    if not measurement:
        measurement = _ensure_dict(formalism_block.get("measurement", {}))
    if not measurement and isinstance(formalism_block.get("measurement_rule"), str):
        measurement = {"description": formalism_block.get("measurement_rule")}

    observables = _ensure_dict(machine.get("observables_bridge", {}))
    if not observables:
        observables = _ensure_dict(formalism_block.get("observables_bridge", {}))

    params = _ensure_list(machine.get("parameters", []))
    constraints = _ensure_dict(machine.get("constraints_checklist", {}))
    philosophy = _ensure_dict(machine.get("philosophy_and_principles", {}))
    predictions = _ensure_list(machine.get("testable_predictions", []))
    if not predictions:
        predictions = _ensure_list(machine.get("testable_prediction", []))
    if not predictions and isinstance(machine.get("predictions"), list):
        predictions = machine.get("predictions")

    name = meta.get("name") or machine.get("name") or machine.get("theory_name") or "Short-Card Derived Theory"
    if _is_placeholder_name(name):
        name = _generate_descriptive_name(machine, fallback="Short-Card Derived Interpretation")
    slug = _slugify(name) or f"theory_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"

    summary = sections[0] if sections else meta.get("novelty", "")
    ontology_section = philosophy.get("ontology", sections[3] if len(sections) >= 4 else "")
    measurement_section = philosophy.get("measurement_postulate", sections[2] if len(sections) >= 3 else "")
    epistemology_section = philosophy.get("epistemology", sections[1] if len(sections) >= 2 else "")
    empirical_section = sections[4] if len(sections) >= 5 else ""
    attitude_section = philosophy.get("attitude_to_no_go_theorems", sections[5] if len(sections) >= 6 else "")

    relation = machine.get("relation_to_sqm") or machine.get("math_relation_to_SQM") or meta.get("relation_to_SQM")
    if isinstance(relation, dict):
        relation = relation.get("type")
    math_relation = relation or "unknown"
    if math_relation == "minimal_change":
        math_relation = "no_change"
    measurement_model = (
        measurement.get("model")
        or measurement.get("name")
        or measurement.get("type", "")
    )
    probability_rule = (
        measurement.get("probability_rule")
        or measurement.get("probability_formula", "")
        or measurement.get("probability_description", "")
    )

    parameter_block = {}
    for item in params:
        key = item.get("name")
        if key:
            parameter_block[key] = {
                "unit": item.get("unit"),
                "scale": item.get("scale"),
                "role": item.get("role"),
                "prior_or_constraint": item.get("prior_or_constraint"),
                "value": item.get("value")
            }

    legacy_theory = {
        "name": name,
        "summary": summary or meta.get("novelty", ""),
        "philosophy": {
            "ontology": ontology_section,
            "measurement": measurement_section,
        },
        "parameters": parameter_block,
        "formalism": {
            "math_relation": f"Relation to SQM: {math_relation}. Dynamics equation: {dynamics.get('equation', '')}",
            "state_space": f"Object: {state_space.get('object')} | Space: {state_space.get('space')} | Definition: {state_space.get('definition')}",
            "measurement_rule": f"Model: {measurement_model}. Update: {measurement.get('update_rule', measurement.get('description', ''))}",
            "born_rule": f"Probability rule: {probability_rule}",
        },
        "semantics": {
            "overview": empirical_section or observables.get("how_variables_map_to_readouts", ""),
            "attitude_to_no_go": attitude_section,
            "measurement_update": measurement_section,
        },
        "core_principles": {
            "ontological_commitments": ontology_section,
            "epistemological_stances": epistemology_section,
            "key_postulates": philosophy.get("core_principles", []) or [meta.get("novelty", "")],
        },
        "metadata": {
            "schema_version": meta.get("schema_version", "2.2"),
            "theory_id": slug,
            "status": "generated",
            "tags": [
                math_relation,
                measurement_model,
                dynamics.get("cp_tp_guarantee", ""),
            ],
            "lineage": {
                "method": "short_card_rag",
                "parents": result.get("selected_card_ids", []),
                "query": result.get("query"),
            },
        },
        "generated_assets": {
            "machine_summary": machine,
            "writeup": writeup,
            "contradictions": result.get("contradictions", {}),
        }
    }

    legacy_theory["observables_bridge"] = observables
    legacy_theory["constraints_checklist"] = constraints
    legacy_theory["testable_predictions"] = predictions

    return legacy_theory, machine


__all__ = ["ShortCardGenerationConfig", "generate_short_card_theory", "convert_to_legacy_schema"]

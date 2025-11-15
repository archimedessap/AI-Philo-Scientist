"""Analyze contradictions among short-form theory cards."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


DIRECT_CONTRADICTION_SCHEMA = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "title": "DirectTheoryContradictions",
    "type": "object",
    "additionalProperties": False,
    "required": ["theory1", "theory2", "contradictions", "summary"],
    "properties": {
        "theory1": {"type": "string"},
        "theory2": {"type": "string"},
        "summary": {"type": "string"},
        "contradictions": {
            "type": "array",
            "minItems": 1,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": [
                    "dimension",
                    "theory1_position",
                    "theory2_position",
                    "core_tension"
                ],
                "properties": {
                    "dimension": {"type": "string"},
                    "theory1_position": {"type": "string"},
                    "theory2_position": {"type": "string"},
                    "core_tension": {"type": "string"},
                    "importance_score": {"type": "number"},
                    "philosophical_implications": {"type": "string"}
                }
            }
        }
    }
}


class ContradictionAnalyzer:
    """Support contradiction analysis for both short-card and direct workflows."""

    def __init__(
        self,
        llm_interface,
        schema_path: str = "schemas/contradiction.schema.json",
    ) -> None:
        self.llm = llm_interface
        self.schema_path = Path(schema_path)
        if not self.schema_path.exists():
            raise FileNotFoundError(f"Contradiction schema not found: {self.schema_path}")
        self.schema = json.loads(self.schema_path.read_text(encoding="utf-8"))
        self.theories: Dict[str, Dict[str, Any]] = {}

    async def build_table(
        self,
        cards: Iterable[Dict[str, Any]],
        task_hint: str = "",
        temperature: float = 0.0,
    ) -> Dict[str, Any]:
        """Call the LLM to generate a structured contradiction table."""
        cards_list = list(cards)
        if len(cards_list) < 2:
            raise ValueError("At least two cards are required to analyze contradictions.")

        messages = self._build_messages(cards_list, task_hint)
        result = await self.llm.query_structured_json(
            messages=messages,
            schema=self.schema,
            schema_name="contradiction_table",
            temperature=temperature,
        )
        if isinstance(result, list):
            result = {"contradictions": result}
        elif isinstance(result, dict) and "items" in result and "contradictions" not in result:
            result = {"contradictions": result.get("items", [])}
        elif isinstance(result, dict) and {"A", "B", "issue", "one_line"}.issubset(result.keys()):
            result = {"contradictions": [result]}

        if not result or "contradictions" not in result:
            print(f"[ERROR] 结构化矛盾输出格式异常: {result}")
            raise ValueError("Structured contradiction result missing required field 'contradictions'.")
        return result

    # ------------------------------------------------------------------
    def _build_messages(self, cards: List[Dict[str, Any]], task_hint: str) -> List[Dict[str, str]]:
        system_prompt = (
            "You are a quantum foundations researcher. Analyse the supplied interpretation cards. "
            "Compare their commitments and produce concise contradiction items. "
            "Use the provided JSON schema. Prefer deep, substantive disagreements over terminology. "
            "Enumerate multiple contradictions (aim for five or more) spanning distinct issue categories whenever the cards allow; "
            "do not stop after a single conflict."
        )

        if task_hint:
            system_prompt += f" Task focus: {task_hint.strip()}"

        cards_text = "\n\n".join(self._format_card(card, index) for index, card in enumerate(cards, start=1))
        user_prompt = (
            "Interpretation cards:\n"
            f"{cards_text}\n\n"
            "Generate a contradiction table that captures the diverse conflicts among these cards. "
            "Cover every major incompatibility cluster (collapse, ontology, determinism, information, locality, etc.) when supported by the cards. "
            "Produce at least two contradictions and prefer five or more. "
            "For each contradiction: use card ids or names in fields 'A' and 'B'; pick the `issue` enum that fits best. "
            "The `one_line` text must be under 25 ASCII characters and explain the tension plainly."
        )

        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

    def _format_card(self, card: Dict[str, Any], index: int) -> str:
        predictions = card.get("predictions", [])
        predictions_text = "; ".join(predictions) if predictions else "No distinct empirical deviations." 
        key_claims = "; ".join(card.get("key_claims", []))
        return (
            f"[{index}] id={card.get('id')} | name={card.get('name')}\n"
            f"one_line: {card.get('one_line')}\n"
            f"math_relation: type={card.get('math_relation_to_SQM', {}).get('type')} | "
            f"math_change={card.get('math_relation_to_SQM', {}).get('math_change')} | "
            f"eq: {card.get('math_relation_to_SQM', {}).get('equations_summary')}\n"
            f"claims: {key_claims}\n"
            f"born_rule: {card.get('born_rule')} | measurement: {card.get('measurement_update')} | locality: {card.get('locality_note')}\n"
            f"predictions: {predictions_text}\n"
            f"tags: {', '.join(card.get('tags', []))}"
        )

    # ------------------------------------------------------------------
    def load_theories(self, theories_dir: str, schema_version: Optional[str] = None) -> None:
        """Load full theory files for direct-mode contradiction analysis."""
        theories_path = Path(theories_dir)
        if not theories_path.exists() or not theories_path.is_dir():
            raise FileNotFoundError(f"Theory directory not found: {theories_dir}")

        loaded: Dict[str, Dict[str, Any]] = {}

        for file_path in theories_path.glob("*.json"):
            try:
                data = json.loads(file_path.read_text(encoding="utf-8"))
            except Exception as exc:
                print(f"[WARN] Failed to load theory file {file_path}: {exc}")
                continue

            # Some files may contain a list of theories; support both structures.
            entries = []
            if isinstance(data, list):
                entries = [item for item in data if isinstance(item, dict)]
            elif isinstance(data, dict):
                entries = [data]
            else:
                continue

            for entry in entries:
                name = entry.get("name") or entry.get("theory_name") or file_path.stem
                if not isinstance(name, str):
                    continue
                if schema_version and schema_version.lower() != "any":
                    meta = entry.get("metadata", {})
                    version = meta.get("schema_version")
                    if version and str(version) != str(schema_version):
                        continue
                loaded[name] = entry

        self.theories = loaded

    async def find_contradictions(self, theory1: str, theory2: str) -> Dict[str, Any]:
        """Ask the LLM to summarise contradictions between two full theories."""
        data1 = self.theories.get(theory1)
        data2 = self.theories.get(theory2)
        if not data1 or not data2:
            return {"error": "Theory not found", "missing": [theory1, theory2]}

        def _compact(theory: Dict[str, Any]) -> Dict[str, Any]:
            """Extract key sections to keep prompts reasonably small."""
            return {
                "name": theory.get("name"),
                "summary": theory.get("summary"),
                "core_principles": theory.get("core_principles") or theory.get("core_assumptions"),
                "formalism": theory.get("formalism") or theory.get("mathematical_formalism"),
                "measurement": theory.get("measurement") or theory.get("measurement_model"),
                "predictions": theory.get("predictions_and_verifiability") or theory.get("empirical_predictions"),
            }

        theory1_text = json.dumps(_compact(data1), ensure_ascii=False, indent=2)
        theory2_text = json.dumps(_compact(data2), ensure_ascii=False, indent=2)

        system_prompt = (
            "You are a quantum foundations analyst. Compare two quantum theories and extract their core contradictions. "
            "Respond using the provided JSON schema. Highlight conceptual, mathematical, ontological, and operational tensions. "
            "Whenever possible, provide at least five high-value contradictions."
        )
        user_prompt = (
            f"Theory A:\n{theory1_text}\n\n"
            f"Theory B:\n{theory2_text}\n\n"
            "Identify the main contradictions between Theory A and Theory B. "
            "Focus on differences in ontology, measurement postulates, dynamics, probability interpretation, classical limit, and non-locality. "
            "For each contradiction, fill in the fields `dimension`, `theory1_position`, `theory2_position`, `core_tension`, "
            "and, when appropriate, `importance_score` (0-10) and `philosophical_implications`."
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        structured = await self.llm.query_structured_json(
            messages=messages,
            schema=DIRECT_CONTRADICTION_SCHEMA,
            schema_name="direct_contradiction_table",
            temperature=0.2,
        )

        if structured:
            result = structured
            if isinstance(result, list):
                result = {
                    "theory1": theory1,
                    "theory2": theory2,
                    "summary": "Auto converted from list output.",
                    "contradictions": result,
                }
            elif isinstance(result, dict) and "contradictions" not in result:
                # interpret as a single contradiction entry
                guessed_summary = result.pop("summary", "Summary unavailable.")
                contradiction_entry = {
                    "dimension": result.get("dimension", "Unknown"),
                    "theory1_position": result.get("theory1_position", ""),
                    "theory2_position": result.get("theory2_position", ""),
                    "core_tension": result.get("core_tension", ""),
                    "importance_score": result.get("importance_score"),
                    "philosophical_implications": result.get("philosophical_implications"),
                }
                result = {
                    "theory1": theory1,
                    "theory2": theory2,
                    "summary": guessed_summary,
                    "contradictions": [contradiction_entry],
                }
        else:
            text_path = Path("logs/last_structured_response.txt")
            text = text_path.read_text(encoding="utf-8") if text_path.exists() else ""
            try:
                as_list = json.loads(text)
                if isinstance(as_list, list):
                    result = {
                        "theory1": theory1,
                        "theory2": theory2,
                        "summary": "Auto recovered from list response.",
                        "contradictions": []
                    }
                    for item in as_list:
                        if isinstance(item, dict):
                            result["contradictions"].append({
                                "dimension": item.get("dimension", "Unknown"),
                                "theory1_position": item.get("theory1_position", ""),
                                "theory2_position": item.get("theory2_position", ""),
                                "core_tension": item.get("core_tension", ""),
                                "importance_score": item.get("importance_score"),
                                "philosophical_implications": item.get("philosophical_implications")
                            })
                    if not result["contradictions"]:
                        return {"error": "LLM_contradiction_analysis_failed", "theory1": theory1, "theory2": theory2}
            except Exception:
                return {"error": "LLM_contradiction_analysis_failed", "theory1": theory1, "theory2": theory2}
        if not result["contradictions"]:
            return {"error": "empty_contradiction_list", "theory1": theory1, "theory2": theory2}

        result.setdefault("theory1", theory1)
        result.setdefault("theory2", theory2)
        if "summary" not in result:
            result["summary"] = "Summary unavailable."
        return result


__all__ = ["ContradictionAnalyzer"]

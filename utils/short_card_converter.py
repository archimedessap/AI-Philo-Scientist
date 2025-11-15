"""Utility functions for converting rich theory records into short-card format."""

from __future__ import annotations

import json
import re
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from jsonschema import Draft7Validator


def _slugify(value: str) -> str:
    """Create a stable, lowercase identifier."""
    value = value.lower().strip()
    value = re.sub(r"[^a-z0-9]+", "_", value)
    value = re.sub(r"_+", "_", value).strip("_")
    return value or "theory"


def _truncate(text: str, limit: int) -> str:
    """Ensure text stays within a character budget."""
    text = " ".join(text.split())
    if len(text) <= limit:
        return text
    return textwrap.shorten(text, width=limit, placeholder='...')


def _first_sentence(text: str) -> str:
    parts = re.split(r"(?<=[.!?。！？])\s+", text.strip())
    return parts[0] if parts else text.strip()


def _collect_strings(value: Any) -> List[str]:
    """Flatten nested structures into a list of strings."""
    results: List[str] = []
    if isinstance(value, str):
        results.append(value)
    elif isinstance(value, dict):
        for item in value.values():
            results.extend(_collect_strings(item))
    elif isinstance(value, list):
        for item in value:
            results.extend(_collect_strings(item))
    return results


def _pick_unique(items: Iterable[str], limit: int) -> List[str]:
    seen = set()
    out: List[str] = []
    for item in items:
        cleaned = " ".join(item.split())
        if not cleaned or cleaned.lower() in seen:
            continue
        seen.add(cleaned.lower())
        out.append(cleaned)
        if len(out) >= limit:
            break
    return out


def _to_tag(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return slug[:40] if slug else "tag"


@dataclass
class ConversionContext:
    """Optional overrides keyed by card id."""
    overrides: Dict[str, Dict[str, Any]]

    @classmethod
    def from_path(cls, path: Optional[Path]) -> "ConversionContext":
        if path and path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(data, dict):
                raise ValueError("Overrides file must contain an object")
            return cls(overrides=data)
        return cls(overrides={})

    def get(self, card_id: str) -> Dict[str, Any]:
        return self.overrides.get(card_id, {})


class ShortCardConverter:
    """Convert long-form theory entries to the compact card schema."""

    def __init__(self, schema_path: Path, overrides: Optional[ConversionContext] = None):
        self.schema_path = schema_path
        self.schema = json.loads(schema_path.read_text(encoding="utf-8"))
        self.validator = Draft7Validator(self.schema)
        self.context = overrides or ConversionContext.from_path(None)

    # Public API ------------------------------------------------------------
    def convert_directory(
        self,
        source_dir: Path,
        destination_dir: Path,
        recursive: bool = False,
        skip_existing: bool = True,
    ) -> int:
        destination_dir.mkdir(parents=True, exist_ok=True)
        pattern = "**/*.json" if recursive else "*.json"
        converted = 0
        for file_path in sorted(source_dir.glob(pattern)):
            if file_path.is_dir():
                continue
            theory = json.loads(file_path.read_text(encoding="utf-8"))
            card = self.convert_theory(theory)
            overrides = self.context.get(card["id"])
            if overrides:
                card.update(overrides)
                self.validator.validate(card)
            card_path = destination_dir / f"{card['id']}.json"
            if skip_existing and card_path.exists():
                continue
            card_path.write_text(json.dumps(card, indent=2, ensure_ascii=False), encoding="utf-8")
            converted += 1
        return converted

    def convert_theory(self, theory: Dict[str, Any]) -> Dict[str, Any]:
        card = {
            "id": self._derive_id(theory),
            "name": theory.get("name") or theory.get("theory_name") or "Unnamed Theory",
            "one_line": self._derive_one_line(theory),
            "math_relation_to_SQM": self._derive_math_relation(theory),
            "key_claims": self._derive_key_claims(theory),
            "born_rule": self._derive_born_rule(theory),
            "measurement_update": self._derive_measurement_update(theory),
            "locality_note": self._derive_locality(theory),
            "predictions": self._derive_predictions(theory),
            "tags": self._derive_tags(theory),
        }
        self.validator.validate(card)
        return card

    # Derivation helpers ---------------------------------------------------
    def _derive_id(self, theory: Dict[str, Any]) -> str:
        meta = theory.get("metadata", {})
        theory_id = meta.get("theory_id") or theory.get("id") or theory.get("name")
        return _slugify(str(theory_id))

    def _derive_one_line(self, theory: Dict[str, Any]) -> str:
        summary = theory.get("one_line") or theory.get("summary")
        if isinstance(summary, str) and summary.strip():
            return _truncate(_first_sentence(summary), 180)
        description = theory.get("description") or theory.get("core_principles")
        texts = _collect_strings(description)
        for text in texts:
            if text.strip():
                return _truncate(_first_sentence(text), 180)
        return 'Summary TBD.'

    def _derive_math_relation(self, theory: Dict[str, Any]) -> Dict[str, Any]:
        meta = theory.get("metadata", {})
        math_meta = theory.get("mathematical_relation_to_sqm", {})
        classification = meta.get("mathematical_classification", {})
        relation_type = math_meta.get("type") or classification.get("type") or "interpretation"
        relation_type = relation_type.lower()
        mapped = {
            "interpretation": "interpretation",
            "extension": "modified_dynamics",
            "modification": "modified_dynamics",
            "modified_qm": "modified_dynamics",
            "retrocausal": "retrocausal",
        }.get(relation_type, "interpretation")
        uses_standard = classification.get("uses_standard_qm_math")
        if uses_standard is None:
            uses_standard = mapped != "modified_dynamics"
        summary = (
            math_meta.get("summary")
            or classification.get("analysis", {}).get("mathematical_relation", {}).get("summary")
            or theory.get("formalism", {}).get("comparison_with_sqm")
            or theory.get("summary")
            or "与SQM的数学关系未说明。"
        )
        return {
            "type": mapped,
            "math_change": bool(not uses_standard),
            "equations_summary": _truncate(summary, 280),
        }

    def _derive_key_claims(self, theory: Dict[str, Any]) -> List[str]:
        claims: List[str] = []
        core = theory.get("key_claims") or theory.get("core_principles") or theory.get("core_assumptions")
        if isinstance(core, list):
            for item in core:
                if isinstance(item, dict):
                    text = item.get("statement") or item.get("principle") or item.get("summary")
                    if text:
                        claims.append(text)
                elif isinstance(item, str):
                    claims.append(item)
        elif isinstance(core, dict):
            claims.extend(_collect_strings(core))
        if len(claims) < 3:
            summary = theory.get("summary") or theory.get("description")
            if isinstance(summary, str):
                sentences = re.split(r"(?<=[.!?。！？])\s+", summary)
                claims.extend(sentences)
        trimmed = [
            _truncate(text, 160)
            for text in claims
            if isinstance(text, str) and text.strip()
        ]
        trimmed = _pick_unique(trimmed, limit=6)
        while len(trimmed) < 3:
            trimmed.append("待补充关键主张。")
        return trimmed

    def _derive_born_rule(self, theory: Dict[str, Any]) -> str:
        text_blobs = _collect_strings(
            [
                theory.get("summary"),
                theory.get("formalism"),
                theory.get("philosophy"),
                theory.get("core_principles"),
            ]
        )
        lowered = " ".join(t.lower() for t in text_blobs)
        tags = {t.lower() for t in (theory.get("metadata", {}).get("tags") or [])}
        if any(key in lowered for key in ["bayes", "subjective", "agent", "belief"]):
            return "normative_bayes"
        if "decision" in lowered:
            return "decision_theory"
        if "envariance" in lowered:
            return "envariance"
        if "born" in lowered:
            return "postulate"
        if "frequentist" in lowered:
            return "other"
        if any("bayes" in tag for tag in tags):
            return "normative_bayes"
        return "postulate"

    def _derive_measurement_update(self, theory: Dict[str, Any]) -> str:
        tags = {t.lower() for t in (theory.get("metadata", {}).get("tags") or [])}
        text_blobs = " ".join(t.lower() for t in _collect_strings(theory.get("core_principles")))
        if any(keyword in tags for keyword in ["subjective-probability", "agent", "information", "bayesian"]):
            return "belief_update"
        if any(keyword in text_blobs for keyword in ["belief", "agent", "bayesian"]):
            return "belief_update"
        if any(keyword in tags for keyword in ["no-collapse", "many-worlds", "pilot-wave", "decoherence"]):
            return "no_collapse"
        if "collapse" in text_blobs or any("collapse" in tag for tag in tags):
            return "physical_collapse"
        summary = (theory.get("summary") or "").lower()
        if "collapse" in summary and "no" not in summary:
            return "physical_collapse"
        return "no_collapse"

    def _derive_locality(self, theory: Dict[str, Any]) -> str:
        tags = [t.lower() for t in (theory.get("metadata", {}).get("tags") or [])]
        summary = (theory.get("summary") or "").lower()
        if any("non-local" in tag or "nonlocal" in tag for tag in tags) or "nonlocal" in summary:
            return "explicitly_nonlocal"
        if any(tag in {"local", "locality"} for tag in tags):
            return "local"
        return "signal_local_only"

    def _derive_predictions(self, theory: Dict[str, Any]) -> List[str]:
        preds = theory.get("predictions") or theory.get("predictions_and_verifiability") or {}
        items = _collect_strings(preds)
        trimmed = [_truncate(it, 160) for it in items if isinstance(it, str) and it.strip()]
        return _pick_unique(trimmed, limit=3)

    def _derive_tags(self, theory: Dict[str, Any]) -> List[str]:
        tags = theory.get("tags") or theory.get("metadata", {}).get("tags") or []
        clean = []
        for tag in tags:
            if not isinstance(tag, str):
                continue
            clean.append(_to_tag(tag))
        if len(clean) < 3:
            extra_sources = [theory.get("math_relation_to_SQM", {}).get("type"), theory.get("mathematical_relation_to_sqm", {}).get("type")]
            for source in extra_sources:
                if isinstance(source, str):
                    clean.append(_to_tag(source))
        clean = _pick_unique(clean, limit=6)
        while len(clean) < 3:
            clean.append("quantum")
        return clean


__all__ = ["ShortCardConverter", "ConversionContext"]

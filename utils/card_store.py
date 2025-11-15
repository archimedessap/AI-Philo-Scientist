"""Simple loader/validator for short theory cards."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from jsonschema import Draft7Validator


class CardStore:
    """Load short cards from disk and provide simple lookup utilities."""

    def __init__(
        self,
        cards_dir: Path | str = "cards",
        schema_path: Path | str = "schemas/card.schema.json",
    ) -> None:
        self.cards_dir = Path(cards_dir)
        self.schema_path = Path(schema_path)
        if not self.cards_dir.exists():
            raise FileNotFoundError(f"Cards directory not found: {self.cards_dir}")
        if not self.schema_path.exists():
            raise FileNotFoundError(f"Schema not found: {self.schema_path}")
        self.validator = Draft7Validator(json.loads(self.schema_path.read_text(encoding="utf-8")))
        self._cards = self._load_cards()

    # ------------------------------------------------------------------
    def _load_cards(self) -> Dict[str, Dict]:
        cards: Dict[str, Dict] = {}
        for path in sorted(self.cards_dir.glob("*.json")):
            data = json.loads(path.read_text(encoding="utf-8"))
            self.validator.validate(data)
            card_id = data["id"]
            if card_id in cards:
                raise ValueError(f"Duplicate card id detected: {card_id}")
            cards[card_id] = data
        return cards

    # Public accessors -------------------------------------------------
    def all_cards(self) -> List[Dict]:
        return list(self._cards.values())

    def get(self, card_id: str) -> Optional[Dict]:
        return self._cards.get(card_id)

    def ids(self) -> Iterable[str]:
        return self._cards.keys()


__all__ = ["CardStore"]

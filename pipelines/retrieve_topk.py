#!/usr/bin/env python3
"""Retrieve top-k relevant theory cards for a given query."""

from __future__ import annotations

import argparse
import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from utils.card_store import CardStore


@dataclass
class RetrievalResult:
    card_id: str
    score: float
    card: Dict


class CardRetriever:
    """Hybrid retriever that prefers LLM embeddings but falls back to TF-IDF."""

    def __init__(
        self,
        llm_interface,
        cards_dir: str = "cards",
        schema_path: str = "schemas/card.schema.json",
        backend: str = "auto",
    ) -> None:
        self.llm = llm_interface
        self.store = CardStore(cards_dir=cards_dir, schema_path=schema_path)
        self.backend = backend
        self._ready = False
        self._card_ids: List[str] = []
        self._corpus: List[str] = []
        self._embedding_matrix: Optional[np.ndarray] = None
        self._embedding_norms: Optional[np.ndarray] = None
        self._vectorizer: Optional[TfidfVectorizer] = None
        self._tfidf_matrix = None

    async def ensure_index(self) -> None:
        if self._ready:
            return
        self._prepare_corpus()
        if self.backend in {"auto", "openai"}:
            try:
                await self._build_embedding_index()
                self.backend = "openai"
                self._ready = True
                return
            except Exception as exc:
                print(f"[WARN] OpenAI embedding backend unavailable, fallback to TF-IDF: {exc}")
                self.backend = "tfidf"
        # TF-IDF fallback
        self._build_tfidf_index()
        self.backend = "tfidf"
        self._ready = True

    async def top_k(self, query: str, k: int = 5, extra_context: Optional[str] = None) -> List[RetrievalResult]:
        await self.ensure_index()
        query_text = query.strip()
        if extra_context:
            query_text = f"{query_text}\n{extra_context.strip()}"
        if self.backend == "openai":
            return await self._top_k_embeddings(query_text, k)
        return self._top_k_tfidf(query_text, k)

    # ------------------------------------------------------------------
    def _prepare_corpus(self) -> None:
        cards = self.store.all_cards()
        self._card_ids = [card["id"] for card in cards]
        self._corpus = [self._card_to_text(card) for card in cards]

    async def _build_embedding_index(self) -> None:
        vectors = []
        for text in self._corpus:
            embedding = await self.llm.get_embedding(text)
            vectors.append(np.asarray(embedding, dtype=float))
        self._embedding_matrix = np.vstack(vectors)
        self._embedding_norms = np.linalg.norm(self._embedding_matrix, axis=1)

    def _build_tfidf_index(self) -> None:
        self._vectorizer = TfidfVectorizer(stop_words="english")
        self._tfidf_matrix = self._vectorizer.fit_transform(self._corpus)

    async def _top_k_embeddings(self, query: str, k: int) -> List[RetrievalResult]:
        query_vec = np.asarray(await self.llm.get_embedding(query), dtype=float)
        query_norm = np.linalg.norm(query_vec)
        denom = self._embedding_norms * (query_norm + 1e-9)
        scores = (self._embedding_matrix @ query_vec) / (denom + 1e-9)
        return self._package(scores, k)

    def _top_k_tfidf(self, query: str, k: int) -> List[RetrievalResult]:
        assert self._vectorizer is not None
        query_vec = self._vectorizer.transform([query])
        scores = (self._tfidf_matrix @ query_vec.T).toarray().ravel()
        # Normalize scores to [0,1] for downstream stability
        if scores.size:
            max_score = float(scores.max())
            if max_score > 0:
                scores = scores / max_score
        return self._package(scores, k)

    def _package(self, raw_scores: np.ndarray, k: int) -> List[RetrievalResult]:
        scores = np.asarray(raw_scores)
        limit = min(k, len(self._card_ids))
        top_indices = np.argsort(scores)[::-1][:limit]
        results: List[RetrievalResult] = []
        for idx in top_indices:
            card = self.store.get(self._card_ids[idx])
            if not card:
                continue
            results.append(
                RetrievalResult(
                    card_id=self._card_ids[idx],
                    score=float(scores[idx]),
                    card=card,
                )
            )
        return results

    def _card_to_text(self, card: Dict) -> str:
        math = card.get("math_relation_to_SQM", {})
        pieces = [
            card.get("name", ""),
            card.get("one_line", ""),
            math.get("type", ""),
            math.get("equations_summary", ""),
            " ".join(card.get("key_claims", [])),
            card.get("measurement_update", ""),
            card.get("born_rule", ""),
            card.get("locality_note", ""),
            " ".join(card.get("predictions", [])),
            " ".join(card.get("tags", [])),
        ]
        return "\n".join(piece for piece in pieces if piece)


# ---------------------------------------------------------------------------
# CLI (optional quick smoke test)
# ---------------------------------------------------------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Retrieve top-k cards for a query.")
    parser.add_argument("query", help="Task or question to retrieve cards for.")
    parser.add_argument("--k", type=int, default=5, help="Number of cards to return.")
    parser.add_argument("--cards-dir", default="cards", help="Directory containing card JSON files.")
    parser.add_argument("--schema", default="schemas/card.schema.json", help="Path to card schema.")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    from theory_generation.llm_interface import LLMInterface

    llm = LLMInterface(model_source="openai", model_name="text-embedding-3-small")
    retriever = CardRetriever(llm, cards_dir=args.cards_dir, schema_path=args.schema)

    async def _run():
        results = await retriever.top_k(args.query, k=args.k)
        for res in results:
            print(f"{res.card_id}\t{res.score:.3f}\t{res.card['name']}")

    asyncio.run(_run())


if __name__ == "__main__":
    main()

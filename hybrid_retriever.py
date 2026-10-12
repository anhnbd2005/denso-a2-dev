"""Experimental BM25 + semantic vector retrieval with rank-only RRF."""

import json
import math
import os
import re
from collections import Counter
from itertools import count
from typing import Any, Dict, List, Optional, Tuple

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

from schema import KnowledgeItem


def tokenize(text: str) -> List[str]:
    return re.findall(
        r"\w+",
        text.lower(),
    )


class HybridKnowledgeRetriever:
    """Field-weighted BM25 and cosine-vector rankings fused with RRF."""

    _embedding_models: Dict[str, SentenceTransformer] = {}
    _collection_sequence = count()

    def __init__(
        self,
        kb_path: str = "data/mock_knowledge.json",
        items: Optional[List[KnowledgeItem]] = None,
        rrf_k: int = 60,
        oos_distance_threshold: float = 0.55,
    ):
        self.rrf_k = rrf_k
        self.oos_distance_threshold = oos_distance_threshold
        self.bm25_k1 = 1.5
        self.bm25_b = 0.75
        self.embedding_backend = "explicit SentenceTransformer"
        self.embedding_model_name = os.getenv(
            "DENSO_EMBEDDING_MODEL",
            "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        )
        self.collection_name = "denso_hybrid_kb"
        self.storage_collection_name = (
            f"{self.collection_name}_{next(self._collection_sequence)}"
        )
        self.collection_lifecycle = "Chroma ephemeral/in-memory"

        if items is None:
            with open(kb_path, "r", encoding="utf-8") as file:
                initial_items = [
                    KnowledgeItem(**item) for item in json.load(file)
                ]
        else:
            initial_items = [
                item
                if isinstance(item, KnowledgeItem)
                else KnowledgeItem(**item)
                for item in items
            ]
        self.kb_items: List[KnowledgeItem] = []
        self.items_by_id: Dict[str, KnowledgeItem] = {}
        self.embedding_model = self._get_embedding_model()
        self.client = chromadb.EphemeralClient(
            settings=Settings(
                anonymized_telemetry=False,
                is_persistent=False,
            )
        )
        self.collection = self.client.get_or_create_collection(
            name=self.storage_collection_name,
            metadata={"hnsw:space": "cosine"},
            embedding_function=None,
        )
        self.reindex_all(initial_items)

    def _get_embedding_model(self) -> SentenceTransformer:
        if self.embedding_model_name not in self._embedding_models:
            self._embedding_models[self.embedding_model_name] = (
                SentenceTransformer(self.embedding_model_name)
            )
        return self._embedding_models[self.embedding_model_name]

    @staticmethod
    def _extra_metadata_to_text(item: KnowledgeItem) -> str:
        parts = []
        for key in sorted(item.extra_metadata, key=str):
            value = item.extra_metadata[key]
            if isinstance(value, (dict, list)):
                rendered = json.dumps(
                    value,
                    ensure_ascii=False,
                    sort_keys=True,
                )
            else:
                rendered = str(value)
            parts.append(f"{key}: {rendered}")
        return " ".join(parts)

    @staticmethod
    def _item_to_document(item: KnowledgeItem) -> str:
        fields = [
                f"Title: {item.title}",
                f"Domain: {item.domain.value}",
                f"Equipment: {item.equipment_or_station}",
                f"Symptom: {item.symptom_trigger}",
                f"Expert Observation: {item.expert_observation}",
                f"Root Cause: {item.root_cause_insight}",
                f"Action: {item.action_solution}",
                "Contraindication: "
                f"{item.contraindications or 'Không có.'}",
                f"Tags: {', '.join(item.tags)}",
            ]
        extra_text = HybridKnowledgeRetriever._extra_metadata_to_text(item)
        if extra_text:
            fields.append(f"Extra Metadata: {extra_text}")
        return "\n".join(fields)

    def _bm25_document(self, item: KnowledgeItem) -> str:
        # Field-weighted corpus construction, not a canonical BM25 field model.
        extra_text = self._extra_metadata_to_text(item)
        return (
            (item.title + " ") * 3
            + (item.equipment_or_station + " ") * 2
            + (item.symptom_trigger + " ") * 2
            + (item.root_cause_insight + " ")
            + (" ".join(item.tags) + " ") * 4
            + (extra_text + " ") * 2
        )

    def _build_bm25_index(self) -> None:
        self.bm25_tokens = [
            tokenize(self._bm25_document(item)) for item in self.kb_items
        ]
        self.bm25_term_frequencies = [
            Counter(tokens) for tokens in self.bm25_tokens
        ]
        self.bm25_document_lengths = [
            len(tokens) for tokens in self.bm25_tokens
        ]
        self.bm25_average_document_length = (
            sum(self.bm25_document_lengths) / len(self.bm25_document_lengths)
            if self.bm25_document_lengths
            else 0.0
        )
        self.bm25_document_frequencies = Counter()
        for tokens in self.bm25_tokens:
            self.bm25_document_frequencies.update(set(tokens))

    def _encode(self, texts: List[str]) -> List[List[float]]:
        embeddings = self.embedding_model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return embeddings.tolist()

    def _upsert_vector_items(self, items: List[KnowledgeItem]) -> None:
        if not items:
            return
        documents = [self._item_to_document(item) for item in items]
        self.collection.upsert(
            ids=[item.knowledge_id for item in items],
            documents=documents,
            embeddings=self._encode(documents),
            metadatas=[
                {"domain": item.domain.value} for item in items
            ],
        )

    def indexed_ids(self) -> set[str]:
        return set(self.collection.get(include=[])["ids"])

    def _assert_index_consistency(self) -> None:
        expected_ids = {item.knowledge_id for item in self.kb_items}
        actual_ids = self.indexed_ids()
        if actual_ids != expected_ids:
            raise RuntimeError(
                "VECTOR_INDEX_ERROR: vector IDs do not match KnowledgeItems"
            )

    def add_item(self, new_item: KnowledgeItem) -> None:
        item = (
            new_item
            if isinstance(new_item, KnowledgeItem)
            else KnowledgeItem(**new_item)
        )
        updated_items = list(self.kb_items)
        for index, existing in enumerate(updated_items):
            if existing.knowledge_id == item.knowledge_id:
                updated_items[index] = item
                break
        else:
            updated_items.append(item)

        self._upsert_vector_items([item])
        self.kb_items = updated_items
        self.items_by_id = {
            current.knowledge_id: current for current in self.kb_items
        }
        self._build_bm25_index()
        self._assert_index_consistency()

    def reindex_all(self, items: List[KnowledgeItem]) -> None:
        new_items = [
            item
            if isinstance(item, KnowledgeItem)
            else KnowledgeItem(**item)
            for item in items
        ]
        new_ids = {item.knowledge_id for item in new_items}
        if len(new_ids) != len(new_items):
            raise ValueError("Duplicate knowledge IDs are not allowed")
        stale_ids = self.indexed_ids() - new_ids
        if stale_ids:
            self.collection.delete(ids=sorted(stale_ids))
        self._upsert_vector_items(new_items)
        self.kb_items = new_items
        self.items_by_id = {
            item.knowledge_id: item for item in self.kb_items
        }
        self._build_bm25_index()
        self._assert_index_consistency()

    def _score_bm25(self, query: str) -> List[Tuple[str, float]]:
        query_tokens = tokenize(query)
        document_count = len(self.kb_items)
        ranking = []
        for item, frequencies, document_length in zip(
            self.kb_items,
            self.bm25_term_frequencies,
            self.bm25_document_lengths,
        ):
            score = 0.0
            for token in query_tokens:
                term_frequency = frequencies[token]
                if term_frequency == 0:
                    continue
                document_frequency = self.bm25_document_frequencies[token]
                inverse_document_frequency = math.log(
                    (document_count - document_frequency + 0.5)
                    / (document_frequency + 0.5)
                    + 1
                )
                denominator = term_frequency + self.bm25_k1 * (
                    1
                    - self.bm25_b
                    + self.bm25_b
                    * document_length
                    / self.bm25_average_document_length
                )
                score += inverse_document_frequency * (
                    term_frequency * (self.bm25_k1 + 1) / denominator
                )
            ranking.append((item.knowledge_id, float(score)))
        return sorted(ranking, key=lambda pair: (-pair[1], pair[0]))

    def _vector_ranking(self, query: str) -> List[Dict[str, Any]]:
        if not self.kb_items:
            return []
        result = self.collection.query(
            query_embeddings=self._encode([query]),
            n_results=len(self.kb_items),
            include=["distances"],
        )
        ids = result.get("ids") or []
        distances = result.get("distances") or []
        if not ids or not distances:
            return []
        return [
            {"id": knowledge_id, "distance": float(distance)}
            for knowledge_id, distance in zip(ids[0], distances[0])
        ]

    def search(
        self,
        query: str,
    ) -> Tuple[Optional[KnowledgeItem], Dict[str, Any]]:
        empty_metadata = {
            "matched_id": None,
            "vector_distance": None,
            "top_bm25_score": 0.0,
            "matched_bm25_score": None,
            "rrf_score": None,
            "vector_ranking": [],
            "bm25_ranking": [],
            "rrf_ranking": [],
        }
        if not query.strip():
            return None, {
                "reason": "EMPTY_QUERY",
                **empty_metadata,
                "rrf_score": 0.0,
            }
        if not self.kb_items:
            return None, {"reason": "EMPTY_KB", **empty_metadata}

        vector_ranking = self._vector_ranking(query)
        bm25_pairs = self._score_bm25(query)
        bm25_ranking = [
            {"id": knowledge_id, "score": score}
            for knowledge_id, score in bm25_pairs
        ]
        top_vector_distance = (
            vector_ranking[0]["distance"] if vector_ranking else None
        )
        top_bm25_score = bm25_pairs[0][1] if bm25_pairs else 0.0

        base_metadata = {
            "vector_distance": top_vector_distance,
            "top_bm25_score": top_bm25_score,
            "vector_ranking": vector_ranking,
            "bm25_ranking": bm25_ranking,
        }
        if (
            top_vector_distance is None
            or top_vector_distance > self.oos_distance_threshold
        ):
            return None, {
                "matched_id": None,
                "reason": "OUT_OF_SCOPE_REJECTED",
                "matched_bm25_score": None,
                "rrf_score": None,
                "rrf_ranking": [],
                **base_metadata,
            }

        rrf_scores = {item.knowledge_id: 0.0 for item in self.kb_items}
        contributions = {
            item.knowledge_id: {
                "vector_rank": None,
                "vector_contribution": 0.0,
                "bm25_rank": None,
                "bm25_contribution": 0.0,
            }
            for item in self.kb_items
        }
        for rank, candidate in enumerate(vector_ranking, start=1):
            contribution = 1.0 / (self.rrf_k + rank)
            knowledge_id = candidate["id"]
            rrf_scores[knowledge_id] += contribution
            contributions[knowledge_id]["vector_rank"] = rank
            contributions[knowledge_id]["vector_contribution"] = contribution
        for rank, (knowledge_id, score) in enumerate(bm25_pairs, start=1):
            if score <= 0.0:
                continue
            contribution = 1.0 / (self.rrf_k + rank)
            rrf_scores[knowledge_id] += contribution
            contributions[knowledge_id]["bm25_rank"] = rank
            contributions[knowledge_id]["bm25_contribution"] = contribution

        rrf_ranking = [
            {
                "id": knowledge_id,
                "rrf_score": score,
                **contributions[knowledge_id],
            }
            for knowledge_id, score in sorted(
                rrf_scores.items(), key=lambda pair: (-pair[1], pair[0])
            )
        ]
        winner = rrf_ranking[0]
        matched_id = winner["id"]
        bm25_scores = dict(bm25_pairs)
        vector_distances = {
            candidate["id"]: candidate["distance"]
            for candidate in vector_ranking
        }
        return self.items_by_id[matched_id], {
            "matched_id": matched_id,
            "reason": "MATCHED",
            "rrf_score": winner["rrf_score"],
            "matched_bm25_score": bm25_scores.get(matched_id, 0.0),
            "matched_vector_distance": vector_distances.get(matched_id),
            "rrf_ranking": rrf_ranking,
            **base_metadata,
        }

import json
import math
import re
from typing import List, Optional, Tuple

from schema import KnowledgeItem


def tokenize(text: str) -> List[str]:
    """
    Tokenizer baseline.

    - lowercase
    - giữ Unicode tiếng Việt
    - tách bằng regex \\w+
    - không stemming
    - không stopword removal
    - không embedding
    """
    text = text.lower()
    return re.findall(r"\w+", text)


class LocalKnowledgeRetriever:
    def __init__(
        self,
        kb_path: str = "data/mock_knowledge.json",
        similarity_threshold: float = 0.20,
    ):
        self.similarity_threshold = similarity_threshold

        with open(kb_path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)

        self.kb_items: List[KnowledgeItem] = [
            KnowledgeItem(**it)
            for it in raw_data
        ]

        self._build_index()

    def _build_index(self):
        self.doc_tokens = []

        for it in self.kb_items:
            combined = (
                (it.title + " ") * 2
                + (it.equipment_or_station + " ")
                + (it.symptom_trigger + " ") * 2
                + (it.root_cause_insight + " ")
                + (" ".join(it.tags) + " ") * 3
            )

            tokens = set(tokenize(combined))
            self.doc_tokens.append(tokens)

    def search(
        self,
        query: str,
    ) -> Tuple[
        Optional[KnowledgeItem],
        float,
        List[Tuple[str, float]],
    ]:
        q_tokens = set(tokenize(query))

        if not q_tokens:
            return None, 0.0, []

        scores = []

        for idx, d_tokens in enumerate(self.doc_tokens):
            intersection = q_tokens.intersection(d_tokens)

            if not intersection:
                score = 0.0
            else:
                score = len(intersection) / math.sqrt(
                    len(q_tokens) * len(d_tokens)
                )

            scores.append(
                (self.kb_items[idx], score)
            )

        scores.sort(
            key=lambda x: x[1],
            reverse=True,
        )

        top_item, top_score = scores[0]

        all_scores_summary = [
            (it.knowledge_id, round(sc, 3))
            for it, sc in scores
        ]

        if top_score >= self.similarity_threshold:
            return (
                top_item,
                top_score,
                all_scores_summary,
            )

        return (
            None,
            top_score,
            all_scores_summary,
        )

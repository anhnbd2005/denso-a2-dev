import json
import os
from typing import Any, Dict, List

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

from llm_client import UniversalLLMClient
from schema import KnowledgeItem


OUT_OF_SCOPE_RESPONSE = (
    "Hiện tại trong Kho tri thức DENSO chưa có "
    "bài học hoặc kinh nghiệm ghi nhận về vấn đề này. "
    "Vui lòng tham khảo chuyên gia trực tiếp hoặc "
    "ghi nhận thêm trường hợp mới."
)


RAG_SYSTEM_PROMPT = """
Bạn là trợ lý kỹ thuật sử dụng Knowledge Base.

BẮT BUỘC:
1. Chỉ dùng context được retrieve.
2. Không tự thêm thông số, nguyên nhân, quy trình hoặc cảnh báo
   nếu context không chứa.
3. Phải ghi knowledge_id.
4. Phải phân tách hiện tượng/quan sát, nguyên nhân, hành động,
   contraindication và verification nếu có.
5. Nếu không có matched knowledge, trả đúng phản hồi out-of-scope.
6. Không mô tả dữ liệu synthetic/mock là production DENSO knowledge.
""".strip()


class DensoRAGEngine:
    def __init__(
        self,
        kb_path: str = "data/mock_knowledge.json",
        provider: str | None = None,
        distance_threshold: float = 0.65,
        embedding_model_name: str | None = None,
    ):
        self.provider = (
            provider
            or os.getenv("DENSO_LLM_PROVIDER", "mock")
        ).lower()
        self.distance_threshold = distance_threshold
        self.embedding_backend = "explicit SentenceTransformer"
        self.embedding_model_name = (
            embedding_model_name
            or os.getenv(
                "DENSO_EMBEDDING_MODEL",
                "sentence-transformers/"
                "paraphrase-multilingual-MiniLM-L12-v2",
            )
        )
        if self.provider == "mock":
            self.llm_client = None
        elif self.provider in {"openai", "openai_compatible"}:
            self.llm_client = UniversalLLMClient()
        else:
            raise ValueError(
                "Unsupported provider. Configure the explicit "
                "OpenAI-compatible client or use mock."
            )

        with open(kb_path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)

        self.kb_items: List[KnowledgeItem] = [
            KnowledgeItem(**item)
            for item in raw_data
        ]
        self.items_by_id = {
            item.knowledge_id: item
            for item in self.kb_items
        }

        self.embedding_model = SentenceTransformer(
            self.embedding_model_name
        )
        self.client = chromadb.EphemeralClient(
            settings=Settings(
                anonymized_telemetry=False,
                is_persistent=False,
            )
        )
        self.collection = self.client.get_or_create_collection(
            name="denso_master_knowhow",
            metadata={"hnsw:space": "cosine"},
            embedding_function=None,
        )
        self._build_index()

    @staticmethod
    def _document_text(item: KnowledgeItem) -> str:
        return "\n".join(
            [
                f"Title: {item.title}",
                f"Domain: {item.domain.value}",
                f"Equipment: {item.equipment_or_station}",
                f"Symptom: {item.symptom_trigger}",
                f"Expert Observation: {item.expert_observation}",
                f"Root Cause: {item.root_cause_insight}",
                f"Action: {item.action_solution}",
                "Contraindication: "
                f"{item.contraindications or 'Không có.'}",
                "Verification: "
                f"{item.verification_standard or 'Chưa có thông tin.'}",
                f"Tags: {', '.join(item.tags)}",
            ]
        )

    def _encode(self, texts: List[str]) -> List[List[float]]:
        embeddings = self.embedding_model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return embeddings.tolist()

    def _build_index(self) -> None:
        documents = [
            self._document_text(item)
            for item in self.kb_items
        ]
        self.collection.upsert(
            ids=[item.knowledge_id for item in self.kb_items],
            documents=documents,
            embeddings=self._encode(documents),
            metadatas=[
                {"domain": item.domain.value}
                for item in self.kb_items
            ],
        )

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
    ) -> Dict[str, Any]:
        if not query.strip():
            return {
                "matched_item": None,
                "distance": None,
                "candidates": [],
            }

        count = min(top_k, len(self.kb_items))
        query_embedding = self._encode([query])
        result = self.collection.query(
            query_embeddings=query_embedding,
            n_results=count,
            include=["distances"],
        )

        ids = result["ids"][0]
        distances = result["distances"][0]
        candidates = [
            {
                "id": knowledge_id,
                "distance": float(distance),
            }
            for knowledge_id, distance in zip(ids, distances)
        ]

        top_id = ids[0]
        top_distance = float(distances[0])
        matched_item = (
            self.items_by_id[top_id]
            if top_distance <= self.distance_threshold
            else None
        )
        return {
            "matched_item": matched_item,
            "distance": top_distance,
            "candidates": candidates,
        }

    @staticmethod
    def _format_grounded_answer(item: KnowledgeItem) -> str:
        return f"""
[TRÍCH NGUỒN: {item.knowledge_id} - {item.title}]

Hiện tượng:
{item.symptom_trigger}

Quan sát:
{item.expert_observation}

Nguyên nhân:
{item.root_cause_insight}

Hành động:
{item.action_solution}

Điều kiêng kỵ:
{item.contraindications or "Không có."}

Tiêu chuẩn nghiệm thu:
{item.verification_standard or "Chưa có thông tin."}
""".strip()

    def _build_llm_prompt(
        self,
        query: str,
        item: KnowledgeItem,
    ) -> str:
        return (
            f"Câu hỏi người dùng:\n{query}\n\n"
            "Context duy nhất được phép sử dụng:\n"
            f"knowledge_id: {item.knowledge_id}\n"
            f"{self._document_text(item)}"
        )

    def ask(self, query: str) -> Dict[str, Any]:
        retrieval = self.retrieve(query)
        matched_item = retrieval["matched_item"]

        if matched_item is None:
            return {
                "matched_kb_id": None,
                "is_out_of_scope": True,
                "response_text": OUT_OF_SCOPE_RESPONSE,
                "distance": retrieval["distance"],
                "provider": self.provider,
                "candidates": retrieval["candidates"],
            }

        if self.provider == "mock":
            response_text = self._format_grounded_answer(
                matched_item
            )
        else:
            if self.llm_client is None:
                raise RuntimeError("OpenAI-compatible API is not configured.")
            response_text = self.llm_client.generate(
                prompt=self._build_llm_prompt(
                    query,
                    matched_item,
                ),
                system_prompt=RAG_SYSTEM_PROMPT,
            )

        return {
            "matched_kb_id": matched_item.knowledge_id,
            "is_out_of_scope": False,
            "response_text": response_text,
            "distance": retrieval["distance"],
            "provider": self.provider,
            "candidates": retrieval["candidates"],
        }

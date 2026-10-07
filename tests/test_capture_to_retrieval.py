"""Offline integration test for capture, save, indexing, retrieval, and reset."""

from hybrid_retriever import HybridKnowledgeRetriever
from workflow_capture import WorkflowCaptureEngine


SOURCE_NOTE = (
    "Sau khi thay bạc đạn motor băng tải số 2, máy rung mạnh. "
    "Kiểm tra thấy hai bulông chân đế phía trái bị lỏng. "
    "Siết chéo bốn bulông với lực 32 Nm thì hết rung. "
    "Không được siết hết lực từng con một vì dễ làm lệch chân đế."
)


class SourceGroundedClient:
    def __init__(self) -> None:
        self.prompt = ""

    def generate_json(self, prompt: str, system_prompt: str) -> dict:
        self.prompt = prompt
        assert "Không được phát minh" in system_prompt
        assert SOURCE_NOTE in prompt
        return {
            "extracted_fields": {
                "title": "Khắc phục rung motor băng tải số 2 sau thay bạc đạn",
                "domain": "Bảo dưỡng",
                "equipment_or_station": "Motor băng tải số 2",
                "symptom_trigger": "Máy rung mạnh sau khi thay bạc đạn.",
                "expert_observation": (
                    "Hai bulông chân đế phía trái bị lỏng; sau xử lý máy hết rung."
                ),
                "root_cause_insight": "Hai bulông chân đế phía trái bị lỏng.",
                "action_solution": "Siết chéo bốn bulông với lực 32 Nm.",
                "contraindications": (
                    "Không siết hết lực từng bulông riêng lẻ vì dễ lệch chân đế."
                ),
                "verification_standard": None,
                "tags": ["motor", "băng tải", "rung", "bulông"],
            }
        }


def run_capture_to_retrieval_test() -> None:
    client = SourceGroundedClient()
    engine = WorkflowCaptureEngine(mode="llm", llm_client=client)
    result = engine.extract_any_text(
        SOURCE_NOTE,
        next_id="KB-NEW-001",
        author="Synthetic Integration Expert",
        created_date="2026-10-07",
    )
    assert result.status == "DRAFT"
    assert not result.missing_required_fields
    assert result.knowledge_item is not None
    item = result.knowledge_item
    assert item.verification_standard is None
    assert "32 Nm" in item.action_solution
    assert "hết rung" in item.expert_observation

    retriever = HybridKnowledgeRetriever()
    frozen_items = list(retriever.kb_items)
    frozen_ids = retriever.indexed_ids()
    retriever.add_item(item)
    assert item.knowledge_id in retriever.indexed_ids()
    assert item.knowledge_id in {row.knowledge_id for row in retriever.kb_items}

    matched, metadata = retriever.search(
        "Motor băng tải số 2 rung mạnh sau thay bạc đạn thì kiểm tra gì?"
    )
    assert matched is not None, metadata
    assert matched.knowledge_id == item.knowledge_id, metadata
    assert any(
        row["id"] == item.knowledge_id for row in metadata["bm25_ranking"]
    )
    assert any(
        row["id"] == item.knowledge_id for row in metadata["vector_ranking"]
    )

    retriever.reindex_all(frozen_items)
    assert retriever.indexed_ids() == frozen_ids
    assert item.knowledge_id not in retriever.indexed_ids()
    print("[PASS] CAPTURE -> SESSION KB -> BM25/VECTOR -> HYBRID -> RESET")


if __name__ == "__main__":
    run_capture_to_retrieval_test()

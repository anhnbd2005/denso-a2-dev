"""Dynamic add, replace, and complete reset tests for Session Hybrid."""

from hybrid_retriever import HybridKnowledgeRetriever
from schema import DomainEnum, KnowledgeItem


def dynamic_item(action: str = "Thay bộ lọc heli chuyên dụng.") -> KnowledgeItem:
    return KnowledgeItem(
        knowledge_id="KB-DYN-999",
        title="Xử lý bộ lọc heli cryogenic Zeta-999",
        domain=DomainEnum.MAINTENANCE,
        equipment_or_station="Trạm Zeta-999",
        symptom_trigger="Báo động ZETA_HELI_999 khi khởi động trạm.",
        expert_observation="Áp suất heli giảm và đèn tím Zeta nhấp nháy.",
        root_cause_insight="Bộ lọc heli cryogenic Zeta-999 bị bão hòa.",
        action_solution=action,
        contraindications="Không bypass liên động ZETA_HELI_999.",
        verification_standard="Áp suất heli ổn định sau ba chu kỳ.",
        tags=["zeta_999", "heli", "cryogenic"],
        source_author="Synthetic Phase 9 Test",
        created_date="2026-10-03",
    )


def run_dynamic_index_tests() -> None:
    retriever = HybridKnowledgeRetriever()
    base_items = list(retriever.kb_items)
    base_ids = {item.knowledge_id for item in base_items}
    assert len(base_items) == 5
    assert retriever.indexed_ids() == base_ids

    new_item = dynamic_item()
    retriever.add_item(new_item)
    assert len(retriever.kb_items) == 6
    assert len(retriever.bm25_tokens) == 6
    assert retriever.indexed_ids() == base_ids | {new_item.knowledge_id}
    matched, metadata = retriever.search(
        "Zeta-999 báo ZETA_HELI_999, bộ lọc heli cryogenic bị bão hòa"
    )
    assert any(
        row["id"] == new_item.knowledge_id
        for row in metadata["vector_ranking"]
    )
    assert any(
        row["id"] == new_item.knowledge_id
        for row in metadata["bm25_ranking"]
    )

    replacement = dynamic_item("Thay cartridge heli Zeta phiên bản B.")
    retriever.add_item(replacement)
    assert len(retriever.kb_items) == 6
    assert retriever.items_by_id[new_item.knowledge_id].action_solution == (
        replacement.action_solution
    )
    document = retriever.collection.get(
        ids=[new_item.knowledge_id], include=["documents"]
    )["documents"][0]
    assert replacement.action_solution in document
    assert len(retriever.bm25_tokens) == 6

    retriever.reindex_all(base_items)
    assert len(retriever.kb_items) == 5
    assert len(retriever.bm25_tokens) == 5
    assert retriever.indexed_ids() == base_ids
    _, after_reset = retriever.search("ZETA_HELI_999 cryogenic")
    assert all(
        row["id"] != new_item.knowledge_id
        for row in after_reset["vector_ranking"]
    )
    assert all(
        row["id"] != new_item.knowledge_id
        for row in after_reset["bm25_ranking"]
    )

    before_duplicate_attempt = retriever.indexed_ids()
    try:
        retriever.reindex_all(base_items + [base_items[0]])
    except ValueError as exc:
        assert "Duplicate knowledge IDs" in str(exc)
    else:
        raise AssertionError("Duplicate ID batch was accepted")
    assert retriever.indexed_ids() == before_duplicate_attempt
    assert len(retriever.kb_items) == len(base_items)

    print("[PASS] DYNAMIC SESSION INDEX TESTS")
    print(
        "Initial: 5 | After add: 6 | Replace: PASS | "
        "After reset: 5 | Stale vector IDs: NO"
    )
    print(
        "New item matched: "
        f"{matched.knowledge_id if matched is not None else None}"
    )


if __name__ == "__main__":
    run_dynamic_index_tests()

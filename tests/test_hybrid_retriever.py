"""Acceptance and audit tests for the Phase 8 hybrid retriever."""

import json
import math

from hybrid_retriever import HybridKnowledgeRetriever


def run_hybrid_tests() -> None:
    with open("data/test_benchmark.json", "r", encoding="utf-8") as file:
        cases = {case["test_id"]: case for case in json.load(file)}

    first = HybridKnowledgeRetriever()
    second = HybridKnowledgeRetriever()
    assert len(first.kb_items) == 5
    assert len(second.kb_items) == 5
    assert first.storage_collection_name != second.storage_collection_name
    assert first.indexed_ids() == second.indexed_ids()

    empty_item, empty_meta = first.search("   ")
    assert empty_item is None
    assert empty_meta["reason"] == "EMPTY_QUERY"

    audited = {}
    for test_id in ["TC-001", "TC-006", "TC-009", "TC-010"]:
        item, metadata = first.search(cases[test_id]["query"])
        audited[test_id] = (item, metadata)
        assert metadata["reason"] in {
            "MATCHED",
            "OUT_OF_SCOPE_REJECTED",
        }
        assert metadata["vector_ranking"]
        assert metadata["bm25_ranking"]
        scores = [row["score"] for row in metadata["bm25_ranking"]]
        assert all(math.isfinite(score) for score in scores)
        assert scores == sorted(scores, reverse=True)
        if item is not None:
            assert metadata["rrf_ranking"]
            assert metadata["rrf_score"] > 0.0
        else:
            assert metadata["rrf_ranking"] == []
            assert metadata["rrf_score"] is None

    query = cases["TC-006"]["query"]
    run_one = first.search(query)
    run_two = first.search(query)
    assert (run_one[0].knowledge_id if run_one[0] else None) == (
        run_two[0].knowledge_id if run_two[0] else None
    )
    assert run_one[1]["rrf_ranking"] == run_two[1]["rrf_ranking"]

    gibberish_item, gibberish_meta = first.search(
        "xyzqwerty abcnonexistent"
    )
    assert gibberish_item is None
    assert gibberish_meta["reason"] == "OUT_OF_SCOPE_REJECTED"

    print("[PASS] HYBRID RETRIEVER TESTS")
    print("KB items: 5 | Two instances: PASS | Duplicate ID error: NO")
    for test_id, (item, metadata) in audited.items():
        matched = item.knowledge_id if item else None
        print(
            f"{test_id}: matched={matched} reason={metadata['reason']} "
            f"bm25={metadata['top_bm25_score']:.6f} "
            f"distance={metadata['vector_distance']:.6f}"
        )


if __name__ == "__main__":
    run_hybrid_tests()

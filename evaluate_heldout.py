"""Retrieval-only evaluation on the extended KB and held-out benchmark.

Loads the frozen base KB plus data/extended_knowledge.json (30 items) and runs
data/heldout_benchmark.json (30 in-scope + 8 out-of-scope questions) through
HybridKnowledgeRetriever without any threshold tuning. The frozen 10-question
benchmark and historical snapshot are not touched.
"""

import json
import statistics
import time
from typing import Any, Dict, List

from hybrid_retriever import HybridKnowledgeRetriever
from schema import KnowledgeItem, TestCase


def load_items() -> List[KnowledgeItem]:
    items = []
    for path in ("data/mock_knowledge.json", "data/extended_knowledge.json"):
        with open(path, "r", encoding="utf-8") as file:
            items.extend(KnowledgeItem(**entry) for entry in json.load(file))
    return items


def run_heldout_evaluation() -> List[Dict[str, Any]]:
    items = load_items()
    retriever = HybridKnowledgeRetriever(items=items)
    with open("data/heldout_benchmark.json", "r", encoding="utf-8") as file:
        cases = [TestCase(**entry) for entry in json.load(file)]

    print("=" * 100)
    print(f"HELD-OUT RETRIEVAL EVALUATION | KB items: {len(items)} | questions: {len(cases)}")
    print(f"OOS distance threshold (not re-tuned): {retriever.oos_distance_threshold}")
    print("=" * 100)

    results = []
    for case in cases:
        started = time.perf_counter()
        matched, metadata = retriever.search(case.query)
        latency_ms = (time.perf_counter() - started) * 1000
        retrieved = matched.knowledge_id if matched is not None else None
        correct = retrieved == case.expected_kb_id
        results.append(
            {
                "test_id": case.test_id,
                "out_of_scope": case.is_out_of_scope,
                "expected": case.expected_kb_id,
                "retrieved": retrieved,
                "correct": correct,
                "distance": metadata["vector_distance"],
                "latency_ms": latency_ms,
            }
        )
        distance = metadata["vector_distance"]
        print(
            f"[{case.test_id}] {'OOS' if case.is_out_of_scope else 'IN '} | "
            f"expected={case.expected_kb_id} | retrieved={retrieved} | "
            f"distance={'N/A' if distance is None else f'{distance:.4f}'} | "
            f"{'PASS' if correct else 'FAIL'}"
        )

    in_scope = [r for r in results if not r["out_of_scope"]]
    oos = [r for r in results if r["out_of_scope"]]
    print("=" * 100)
    print(f"In-scope top-1 retrieval: {sum(r['correct'] for r in in_scope)}/{len(in_scope)}")
    print(f"OOS rejection:            {sum(r['correct'] for r in oos)}/{len(oos)}")
    print(f"Overall:                  {sum(r['correct'] for r in results)}/{len(results)}")
    in_distances = [r["distance"] for r in in_scope if r["distance"] is not None]
    oos_distances = [r["distance"] for r in oos if r["distance"] is not None]
    print(
        f"Distance: max_in_scope={max(in_distances):.4f} | "
        f"min_oos={min(oos_distances):.4f}"
    )
    print(f"Retrieval latency ms: median={statistics.median(r['latency_ms'] for r in results):.1f}")
    return results


if __name__ == "__main__":
    run_heldout_evaluation()

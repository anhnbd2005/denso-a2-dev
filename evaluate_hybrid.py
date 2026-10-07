"""Evaluate Phase 8 hybrid retrieval with the frozen Phase 2 rubric."""

import json
import statistics
import time
from typing import Any, Dict, List

from evaluate_usefulness import contains_concept
from hybrid_retriever import HybridKnowledgeRetriever
from operational_metrics import load_operational_metric
from rag_engine import OUT_OF_SCOPE_RESPONSE
from schema import KnowledgeItem, TestCase


def format_knowledge_item(item: KnowledgeItem) -> str:
    return f"""
[TRÍCH NGUỒN: {item.knowledge_id} - {item.title}]

Observation:
{item.expert_observation}

Root Cause:
{item.root_cause_insight}

Action:
{item.action_solution}

Contraindication:
{item.contraindications or "Không có."}

Verification:
{item.verification_standard or "Chưa có thông tin."}
""".strip()


def classify_failures(
    test_case: TestCase,
    retrieved_id: str | None,
    scores: Dict[str, float],
) -> List[str]:
    failures = []
    if test_case.is_out_of_scope and retrieved_id is not None:
        failures.append("HYBRID_OOS_FALSE_POSITIVE")
    elif not test_case.is_out_of_scope and retrieved_id is None:
        failures.append("HYBRID_OOS_FALSE_NEGATIVE")
    elif (
        not test_case.is_out_of_scope
        and retrieved_id != test_case.expected_kb_id
    ):
        failures.append("HYBRID_WRONG_RETRIEVAL")
    if scores["cause"] == 0.0:
        failures.append("HYBRID_CAUSE_MISS")
    if scores["action"] == 0.0:
        failures.append("HYBRID_ACTION_MISS")
    if scores["safety"] == 0.0:
        failures.append("HYBRID_SAFETY_MISS")
    return failures


def run_hybrid_evaluation() -> List[Dict[str, Any]]:
    retriever = HybridKnowledgeRetriever(
        rrf_k=60,
        oos_distance_threshold=0.55,
    )
    with open("data/test_benchmark.json", "r", encoding="utf-8") as file:
        test_cases = [TestCase(**item) for item in json.load(file)]

    print("=" * 100)
    print("DENSO A2 — HYBRID V2 CALIBRATED BENCHMARK")
    print("Development benchmark: 10 questions")
    print("Threshold selected using the same benchmark: YES")
    print("Independent held-out evaluation: NO")
    print("BM25 k1=1.5 b=0.75 | RRF k=60 | OOS cosine distance threshold=0.55")
    print(f"Embedding backend: {retriever.embedding_backend}")
    print(f"Embedding model: {retriever.embedding_model_name}")
    print("Collection: denso_hybrid_kb | Chroma ephemeral/in-memory")
    print("=" * 100)

    results = []
    for test_case in test_cases:
        started = time.perf_counter()
        matched_item, metadata = retriever.search(test_case.query)
        latency_ms = (time.perf_counter() - started) * 1000
        retrieved_id = (
            matched_item.knowledge_id if matched_item is not None else None
        )
        if test_case.is_out_of_scope and matched_item is None:
            response = OUT_OF_SCOPE_RESPONSE
            retrieval_score = 1.0
            cause_score = 1.0
            action_score = 1.0
            safety_score = 1.0
        elif matched_item is None:
            # An in-scope false rejection must score zero without crashing.
            response = OUT_OF_SCOPE_RESPONSE
            retrieval_score = 0.0
            cause_score = 0.0
            action_score = 0.0
            safety_score = 0.0
        else:
            response = format_knowledge_item(matched_item)
            retrieval_score = float(
                retrieved_id == test_case.expected_kb_id
            )
            cause_score = float(
                contains_concept(
                    response, test_case.rubric.must_include_root_cause
                )
            )
            action_score = float(
                contains_concept(
                    response, test_case.rubric.must_include_action
                )
            )
            safety_score = float(
                contains_concept(
                    response,
                    test_case.rubric.must_include_contraindication or "",
                )
            )
        scores = {
            "retrieval": retrieval_score,
            "cause": cause_score,
            "action": action_score,
            "safety": safety_score,
        }
        total = sum(scores.values()) / 4.0
        failures = classify_failures(test_case, retrieved_id, scores)
        result = {
            "test_id": test_case.test_id,
            "scope": (
                "out-of-scope"
                if test_case.is_out_of_scope
                else "in-scope"
            ),
            "expected": test_case.expected_kb_id,
            "retrieved": retrieved_id,
            "top_bm25_score": metadata["top_bm25_score"],
            "vector_distance": metadata["vector_distance"],
            "threshold": retriever.oos_distance_threshold,
            "rrf_score": metadata["rrf_score"],
            "scores": scores,
            "total": total,
            "is_useful": total >= 0.75,
            "failures": failures,
            "metadata": metadata,
            "latency_ms": latency_ms,
        }
        results.append(result)
        distance = metadata["vector_distance"]
        distance_text = "N/A" if distance is None else f"{distance:.6f}"
        rrf_text = (
            "N/A"
            if metadata["rrf_score"] is None
            else f"{metadata['rrf_score']:.6f}"
        )
        print(
            f"[{test_case.test_id}] scope={result['scope']} | "
            f"expected={test_case.expected_kb_id} | retrieved={retrieved_id} | "
            f"BM25={metadata['top_bm25_score']:.6f} | "
            f"distance={distance_text} | threshold=0.55 | RRF={rrf_text} | "
            f"R={retrieval_score:.0f} C={cause_score:.0f} "
            f"A={action_score:.0f} S={safety_score:.0f} | "
            f"total={total:.2f} | {'PASS' if total >= 0.75 else 'FAIL'}"
        )
        if failures:
            print(f"  Failures: {', '.join(failures)}")
            for ranking_name, score_name in [
                ("bm25_ranking", "score"),
                ("vector_ranking", "distance"),
                ("rrf_ranking", "rrf_score"),
            ]:
                top_three = metadata[ranking_name][:3]
                print(
                    f"  {ranking_name} top-3: "
                    + ", ".join(
                        f"{row['id']}={row[score_name]:.6f}"
                        for row in top_three
                    )
                )
            print("  RRF contributions:")
            for row in metadata["rrf_ranking"][:3]:
                print(
                    f"    {row['id']}: vector rank={row['vector_rank']} "
                    f"({row['vector_contribution']:.6f}), "
                    f"BM25 rank={row['bm25_rank']} "
                    f"({row['bm25_contribution']:.6f}), "
                    f"total={row['rrf_score']:.6f}"
                )

    useful = sum(result["is_useful"] for result in results)
    in_scope = [result for result in results if result["scope"] == "in-scope"]
    out_of_scope = [
        result for result in results if result["scope"] == "out-of-scope"
    ]
    in_scope_correct = sum(
        result["scores"]["retrieval"] == 1.0 for result in in_scope
    )
    oos_rejected = sum(
        result["scores"]["retrieval"] == 1.0 for result in out_of_scope
    )
    overall_correct = sum(
        result["scores"]["retrieval"] == 1.0 for result in results
    )
    latencies = [result["latency_ms"] for result in results]
    in_scope_distances = [
        result["vector_distance"] for result in in_scope
    ]
    oos_distances = [
        result["vector_distance"] for result in out_of_scope
    ]
    max_in_scope_distance = max(in_scope_distances)
    min_oos_distance = min(oos_distances)
    separation_gap = min_oos_distance - max_in_scope_distance
    midpoint = (min_oos_distance + max_in_scope_distance) / 2.0
    print("=" * 100)
    print(
        f"Calibrated usefulness: {useful}/{len(results)} "
        f"({100 * useful / len(results):.1f}%)"
    )
    print(f"In-scope retrieval: {in_scope_correct}/{len(in_scope)}")
    print(f"OOS rejection: {oos_rejected}/{len(out_of_scope)}")
    print(f"Overall scope+retrieval: {overall_correct}/{len(results)}")
    print("HYBRID RETRIEVAL LATENCY (MEASURED THIS RUN)")
    print(
        f"Latency ms: mean={statistics.mean(latencies):.3f} | "
        f"median={statistics.median(latencies):.3f} | "
        f"min={min(latencies):.3f} | max={max(latencies):.3f}"
    )
    print(
        f"Distance margin: max_in_scope={max_in_scope_distance:.6f} | "
        f"min_oos={min_oos_distance:.6f} | gap={separation_gap:.6f} | "
        f"midpoint={midpoint:.6f}"
    )
    print("Correct OOS 4/4 means correct abstention, not an answered cause/action.")
    print("-" * 100)
    print("OPERATIONAL KPI SCENARIO")
    operational_metric = load_operational_metric()
    if operational_metric is None:
        print("Status: N/A | No operational metric configuration supplied")
    else:
        ratio = operational_metric.improvement_ratio
        ratio_text = "N/A" if ratio is None else f"{ratio:g}x"
        print(f"Status: {operational_metric.status.upper()}")
        print(
            f"T_before: {operational_metric.t_before_minutes} min | "
            f"T_after: {operational_metric.t_after_minutes} min | "
            f"{operational_metric.claim_type.title()} improvement: {ratio_text}"
        )
        print(f"Source note: {operational_metric.source_note}")
        print("Independent real timing study: NO")
    print("=" * 100)
    return results


if __name__ == "__main__":
    run_hybrid_evaluation()

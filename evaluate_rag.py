import json
import os
import time

from evaluate_usefulness import contains_concept
from rag_engine import DensoRAGEngine
from schema import TestCase


def classify_failures(
    tc: TestCase,
    retrieved_id,
    scores,
):
    failures = []
    if tc.is_out_of_scope and retrieved_id is not None:
        failures.append("RAG_OOS_FALSE_POSITIVE")
    elif not tc.is_out_of_scope and retrieved_id is None:
        failures.append("RAG_OOS_FALSE_NEGATIVE")
    elif (
        not tc.is_out_of_scope
        and retrieved_id != tc.expected_kb_id
    ):
        failures.append("RAG_WRONG_RETRIEVAL")

    if scores["cause"] == 0.0:
        failures.append("RAG_CAUSE_MISS")
    if scores["action"] == 0.0:
        failures.append("RAG_ACTION_MISS")
    if scores["safety"] == 0.0:
        failures.append("RAG_SAFETY_MISS")
    return failures


def run_rag_evaluation():
    provider = os.getenv("DENSO_LLM_PROVIDER", "mock").lower()
    engine = DensoRAGEngine(
        provider=provider,
        distance_threshold=0.65,
    )

    with open(
        "data/test_benchmark.json",
        "r",
        encoding="utf-8",
    ) as f:
        test_cases = [
            TestCase(**item)
            for item in json.load(f)
        ]

    results = []
    passed_count = 0

    print("=" * 80)
    print("DENSO A2 — SEMANTIC RAG MOCK BENCHMARK")
    print(f"Provider: {provider}")
    print(f"Embedding backend: {engine.embedding_backend}")
    print(f"Embedding model: {engine.embedding_model_name}")
    print("Distance: cosine | Scope threshold: 0.65")
    print("Collection: Chroma ephemeral/in-memory")
    print("=" * 80)

    for tc in test_cases:
        start = time.perf_counter()
        try:
            output = engine.ask(tc.query)
        except Exception as exc:
            raise RuntimeError(
                f"LLM_PROVIDER_ERROR tại {tc.test_id}: {exc}"
            ) from exc
        latency_ms = (time.perf_counter() - start) * 1000

        response_text = output["response_text"]
        retrieved_id = output["matched_kb_id"]

        if tc.is_out_of_scope:
            s_retrieval = 1.0 if retrieved_id is None else 0.0
        else:
            s_retrieval = (
                1.0
                if retrieved_id == tc.expected_kb_id
                else 0.0
            )

        s_cause = (
            1.0
            if contains_concept(
                response_text,
                tc.rubric.must_include_root_cause,
            )
            else 0.0
        )
        s_action = (
            1.0
            if contains_concept(
                response_text,
                tc.rubric.must_include_action,
            )
            else 0.0
        )

        if tc.is_out_of_scope:
            response_lower = response_text.lower()
            s_safety = (
                1.0
                if (
                    "chưa có dữ liệu" in response_lower
                    or "chưa có kinh nghiệm" in response_lower
                )
                else 0.0
            )
        else:
            expected_contra = (
                tc.rubric.must_include_contraindication
                or ""
            )
            s_safety = (
                1.0
                if contains_concept(
                    response_text,
                    expected_contra,
                )
                else 0.0
            )

        scores = {
            "retrieval": s_retrieval,
            "cause": s_cause,
            "action": s_action,
            "safety": s_safety,
        }
        total = sum(scores.values()) / 4.0
        is_useful = total >= 0.75
        if is_useful:
            passed_count += 1

        failures = classify_failures(
            tc,
            retrieved_id,
            scores,
        )
        result = {
            "test_id": tc.test_id,
            "scope": (
                "out-of-scope"
                if tc.is_out_of_scope
                else "in-scope"
            ),
            "expected": tc.expected_kb_id,
            "retrieved": retrieved_id,
            "distance": output["distance"],
            "provider": output["provider"],
            "scores": scores,
            "total": total,
            "is_useful": is_useful,
            "failures": failures,
            "candidates": output["candidates"],
            "latency_ms": latency_ms,
        }
        results.append(result)

        distance_text = (
            "N/A"
            if output["distance"] is None
            else f"{output['distance']:.6f}"
        )
        status = "PASS" if is_useful else "FAIL"
        print(
            f"[{tc.test_id}] scope={result['scope']} | "
            f"expected={tc.expected_kb_id} | "
            f"retrieved={retrieved_id} | "
            f"distance={distance_text} | provider={provider} | "
            f"R={s_retrieval:.0f} C={s_cause:.0f} "
            f"A={s_action:.0f} S={s_safety:.0f} | "
            f"total={total:.2f} | {status}"
        )
        if failures:
            print(f"  Failures: {', '.join(failures)}")
            print(
                "  Candidates: "
                + ", ".join(
                    f"{candidate['id']}="
                    f"{candidate['distance']:.6f}"
                    for candidate in output["candidates"]
                )
            )

    usefulness_pct = passed_count / len(test_cases) * 100
    print("=" * 80)
    print(
        f"Useful: {passed_count}/{len(test_cases)} "
        f"({usefulness_pct:.1f}%)"
    )
    print("=" * 80)
    return results


if __name__ == "__main__":
    run_rag_evaluation()

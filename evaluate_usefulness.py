import json
import re

from assistant import KnowledgeAssistant
from retriever import LocalKnowledgeRetriever
from schema import TestCase


def contains_concept(
    text: str,
    concept: str,
) -> bool:
    """
    Baseline lexical rubric matcher.

    Đây KHÔNG phải semantic judge.

    - tokenize rubric
    - bỏ token <= 2 chars
    - kiểm tra token xuất hiện trong answer
    - PASS component nếu >= 40%
    """
    text_clean = text.lower()

    concept_tokens = [
        word
        for word in re.findall(
            r"\w+",
            concept.lower(),
        )
        if len(word) > 2
    ]

    if not concept_tokens:
        return True

    match_count = sum(
        1
        for token in concept_tokens
        if token in text_clean
    )

    return (
        match_count / len(concept_tokens)
    ) >= 0.40


def run_evaluation():
    retriever = LocalKnowledgeRetriever(
        similarity_threshold=0.18
    )

    assistant = KnowledgeAssistant(
        retriever
    )

    with open(
        "data/test_benchmark.json",
        "r",
        encoding="utf-8",
    ) as f:
        test_data = json.load(f)

    test_cases = [
        TestCase(**it)
        for it in test_data
    ]

    results = []
    passed_count = 0

    print("=" * 80)
    print(
        "BẮT ĐẦU CHẠY BENCHMARK "
        "ĐÁNH GIÁ % HỮU ÍCH (DENSO A2)"
    )
    print("=" * 80)

    for tc in test_cases:
        output = assistant.answer_query(
            tc.query
        )

        resp = output["response_text"]
        retrieved_id = output["matched_kb_id"]

        # --------------------------------
        # 1. RETRIEVAL SCORE
        # --------------------------------
        if tc.is_out_of_scope:
            s_retrieval = (
                1.0
                if retrieved_id is None
                else 0.0
            )
        else:
            s_retrieval = (
                1.0
                if retrieved_id
                == tc.expected_kb_id
                else 0.0
            )

        # --------------------------------
        # 2. ROOT CAUSE
        # --------------------------------
        s_cause = (
            1.0
            if contains_concept(
                resp,
                tc.rubric.must_include_root_cause,
            )
            else 0.0
        )

        # --------------------------------
        # 3. ACTION
        # --------------------------------
        s_action = (
            1.0
            if contains_concept(
                resp,
                tc.rubric.must_include_action,
            )
            else 0.0
        )

        # --------------------------------
        # 4. SAFETY
        # --------------------------------
        if tc.is_out_of_scope:
            response_lower = resp.lower()

            s_safety = (
                1.0
                if (
                    "chưa có dữ liệu"
                    in response_lower
                    or
                    "chưa có kinh nghiệm"
                    in response_lower
                )
                else 0.0
            )

        else:
            expected_contra = (
                tc.rubric
                .must_include_contraindication
                or ""
            )

            s_safety = (
                1.0
                if contains_concept(
                    resp,
                    expected_contra,
                )
                else 0.0
            )

        # --------------------------------
        # TOTAL
        # --------------------------------
        total_score = (
            s_retrieval
            + s_cause
            + s_action
            + s_safety
        ) / 4.0

        is_useful = (
            total_score >= 0.75
        )

        if is_useful:
            passed_count += 1

        result = {
            "test_id": tc.test_id,
            "role": tc.user_role,
            "is_out_of_scope": (
                tc.is_out_of_scope
            ),
            "expected": tc.expected_kb_id,
            "retrieved": retrieved_id,
            "scores": {
                "retrieval": s_retrieval,
                "cause": s_cause,
                "action": s_action,
                "safety": s_safety,
            },
            "total": total_score,
            "is_useful": is_useful,
            "retrieval_score": output[
                "confidence_score"
            ],
        }

        results.append(result)

        status_icon = (
            "✅ ĐẠT"
            if is_useful
            else "❌ CHƯA ĐẠT"
        )

        print(
            f"[{tc.test_id}] "
            f"{status_icon} | "
            f"Điểm: {total_score:.2f} | "
            f"Kỳ vọng: {tc.expected_kb_id} "
            f"-> Lấy được: {retrieved_id}"
        )

    usefulness_pct = (
        passed_count
        / len(test_cases)
    ) * 100.0

    print("=" * 80)
    print(
        "KẾT QUẢ ĐÁNH GIÁ "
        "CHỈ SỐ % HỮU ÍCH"
    )
    print(
        f"Tổng số câu test: "
        f"{len(test_cases)}"
    )
    print(
        "Số câu đạt tiêu chuẩn hữu ích "
        "(score >= 0.75): "
        f"{passed_count}/"
        f"{len(test_cases)}"
    )
    print(
        "👉 CHỈ SỐ % HỮU ÍCH "
        "TOÀN HỆ THỐNG: "
        f"{usefulness_pct:.1f}%"
    )
    print("=" * 80)

    return results


if __name__ == "__main__":
    run_evaluation()

import json
from schema import KnowledgeItem, TestCase


# Benchmark contract for a future evaluation phase (not executed here):
# Usefulness Score = (S_retrieval + S_cause + S_action + S_safety) / 4
# PASS / USEFUL when Usefulness Score >= 0.75
# Useful Percentage = (number_of_passed_cases / 10) * 100


def validate_benchmark():
    # 1. Load Knowledge Base
    with open("data/mock_knowledge.json", "r", encoding="utf-8") as f:
        kb_data = json.load(f)

    kb_ids = {
        KnowledgeItem(**item).knowledge_id
        for item in kb_data
    }

    # 2. Load Benchmark
    with open("data/test_benchmark.json", "r", encoding="utf-8") as f:
        test_data = json.load(f)

    validated_tests = []
    in_scope_count = 0
    out_scope_count = 0

    for entry in test_data:
        tc = TestCase(**entry)
        validated_tests.append(tc)

        if tc.is_out_of_scope:
            out_scope_count += 1

            assert (
                tc.expected_kb_id is None
            ), (
                f"TC {tc.test_id} là out-of-scope "
                "nhưng lại có expected_kb_id!"
            )

        else:
            in_scope_count += 1

            assert (
                tc.expected_kb_id in kb_ids
            ), (
                f"TC {tc.test_id} tham chiếu ID "
                f"{tc.expected_kb_id} không tồn tại trong KB!"
            )

    print(
        f"✅ Kiểm tra Benchmark thành công! "
        f"Tổng cộng {len(validated_tests)} câu test."
    )
    print(
        f" - Câu hỏi trong phạm vi tri thức "
        f"(In-scope): {in_scope_count}"
    )
    print(
        f" - Câu hỏi bẫy / ngoài dữ liệu "
        f"(Out-of-scope): {out_scope_count}"
    )


if __name__ == "__main__":
    validate_benchmark()

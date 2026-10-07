import json

from schema import KnowledgeItem
from workflow_capture import WorkflowCaptureEngine


def run_capture_test():
    with open(
        "data/raw_expert_sessions.json",
        "r",
        encoding="utf-8",
    ) as f:
        raw_sessions = json.load(f)

    assert (
        len(raw_sessions) == 2
    ), "Phase 4 fixture phải có đúng 2 raw sessions"

    engine = WorkflowCaptureEngine(
        mode="deterministic"
    )

    extracted_items = []

    print("=" * 80)
    print(
        "BẮT ĐẦU KIỂM THỬ WORKFLOW CAPTURE "
        "(RAW TEXT -> STANDARDIZED SCHEMA)"
    )
    print("=" * 80)

    for idx, raw in enumerate(
        raw_sessions,
        start=6,
    ):
        next_id = (
            f"KB-CAP-{idx:03d}"
        )

        item: KnowledgeItem = (
            engine.extract(
                raw,
                next_id=next_id,
            )
        )

        extracted_items.append(
            item
        )

        print(
            f"\n[SESSION: {raw['session_id']}] "
            f"Tác giả: {raw['author']}"
        )

        print(
            "👉 VĂN BẢN THÔ (RAW INPUT):\n"
            f"\"{raw['raw_content'][:130]}...\""
        )

        print(
            "\n👉 TRI THỨC ĐÃ CHUẨN HÓA "
            "(EXTRACTED KNOWLEDGE):"
        )

        print(
            f"  - Mã ID: "
            f"{item.knowledge_id}"
        )

        print(
            f"  - Tiêu đề: "
            f"{item.title}"
        )

        print(
            f"  - Bộ phận: "
            f"{item.domain.value}"
        )

        print(
            "  - Nguyên nhân ngầm "
            "(Root Cause): "
            f"{item.root_cause_insight}"
        )

        print(
            "  - Hành động xử lý "
            "(Action): "
            f"{item.action_solution}"
        )

        print(
            "  - Điều kiêng kỵ "
            "(Contraindications): "
            f"{item.contraindications}"
        )

        print(
            f"  - Tags: "
            f"{', '.join(item.tags)}"
        )

        assert (
            item.root_cause_insight
        ), "Thiếu nguyên nhân ngầm!"

        assert (
            item.contraindications
        ), (
            "Thiếu điều kiêng kỵ / "
            "cảnh báo rủi ro!"
        )

        assert (
            len(item.tags) >= 3
        ), "Thiếu tags phân loại!"

        assert (
            item.source_author
            == raw["author"]
        ), (
            "source_author không giữ đúng "
            "tác giả từ raw session"
        )

        assert (
            item.created_date
            == raw["date"]
        ), (
            "created_date không giữ đúng "
            "metadata từ raw session"
        )

    assert (
        len(extracted_items) == 2
    )

    assert (
        extracted_items[0].knowledge_id
        != extracted_items[1].knowledge_id
    )

    print(
        "\n" + "=" * 80
    )

    print(
        "✅ WORKFLOW CAPTURE HOÀN TẤT "
        "THÀNH CÔNG: "
        f"{len(extracted_items)}/"
        f"{len(raw_sessions)} phiên."
    )

    print(
        "Các KnowledgeItem đều hợp lệ "
        "theo Pydantic schema."
    )

    print(
        "=" * 80
    )


if __name__ == "__main__":
    run_capture_test()

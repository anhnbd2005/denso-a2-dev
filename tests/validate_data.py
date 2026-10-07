import json
from schema import KnowledgeItem


def test_validate():
    with open("data/mock_knowledge.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    validated_items = []
    for entry in data:
        item = KnowledgeItem(**entry)
        validated_items.append(item)

    print(
        f"✅ Kiểm tra thành công! Đã nạp và chuẩn hóa đúng {len(validated_items)} case tri thức."
    )
    for it in validated_items:
        print(f" - [{it.knowledge_id}] {it.domain}: {it.title}")


if __name__ == "__main__":
    test_validate()

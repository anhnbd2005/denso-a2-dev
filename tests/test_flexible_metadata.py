"""Flexible metadata schema, import, search, and dynamic-index tests."""

import io
import json

import pandas as pd

from batch_importer import BatchKnowledgeImporter
from hybrid_retriever import HybridKnowledgeRetriever
from schema import KnowledgeItem


def metadata_dataframe() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "id": "KB-META-999",
                "title": "Khắc phục PLC E-402 tại line L04",
                "domain": "Bảo dưỡng",
                "equipment": "PLC-controlled feeder line L04",
                "symptom": "PLC báo E-402 trong ca 2.",
                "observation": "Feeder dừng khi mã E-402 xuất hiện.",
                "root_cause": "Giắc tín hiệu PLC tại line L04 bị lỏng.",
                "action": "Cô lập nguồn rồi gắn lại giắc tín hiệu PLC.",
                "verification": "Chạy 20 chu kỳ không tái xuất hiện E-402.",
                "author": "Synthetic Metadata Expert",
                "date": "2026-10-05",
                "PLC Error Code": "E-402",
                "Ca sản xuất": "Ca 2",
                "Nhiệt độ dầu": 47.5,
                "Line Code": "L04",
                "Optional Empty": None,
            }
        ]
    )


def run_flexible_metadata_tests() -> None:
    with open("data/mock_knowledge.json", encoding="utf-8") as file:
        base_items = [KnowledgeItem(**row) for row in json.load(file)]
    assert len(base_items) == 5
    assert all(item.extra_metadata == {} for item in base_items)

    future = base_items[0].model_copy(
        update={"custom_future_field": "abc"}
    )
    # model_copy does not validate extras, so validate a full dump explicitly.
    future_data = base_items[0].model_dump()
    future_data["custom_future_field"] = "abc"
    future = KnowledgeItem(**future_data)
    assert future.custom_future_field == "abc"
    assert future.extra_metadata == {}

    importer = BatchKnowledgeImporter()
    dataframe = metadata_dataframe()
    csv_bytes = dataframe.to_csv(index=False).encode("utf-8")
    csv_items, csv_errors, csv_report = importer.parse_file(
        csv_bytes,
        "metadata.csv",
        {item.knowledge_id for item in base_items},
    )
    assert not csv_errors
    assert csv_report["extra_metadata_columns"] == [
        "PLC Error Code",
        "Ca sản xuất",
        "Nhiệt độ dầu",
        "Line Code",
        "Optional Empty",
    ]
    expected_metadata = {
        "PLC Error Code": "E-402",
        "Ca sản xuất": "Ca 2",
        "Nhiệt độ dầu": 47.5,
        "Line Code": "L04",
    }
    item = csv_items[0]
    assert item.extra_metadata == expected_metadata

    excel_buffer = io.BytesIO()
    dataframe.to_excel(excel_buffer, index=False, engine="openpyxl")
    xlsx_items, xlsx_errors, _ = importer.parse_file(
        excel_buffer.getvalue(),
        "metadata.xlsx",
        {base.knowledge_id for base in base_items},
    )
    assert not xlsx_errors
    assert xlsx_items[0].extra_metadata == expected_metadata

    dumped = item.model_dump()
    serialized = json.dumps(dumped, ensure_ascii=False)
    reloaded = KnowledgeItem(**json.loads(serialized))
    assert reloaded.extra_metadata == expected_metadata

    different_order = item.model_copy(
        update={"extra_metadata": dict(reversed(list(expected_metadata.items())))}
    )
    assert HybridKnowledgeRetriever._extra_metadata_to_text(item) == (
        HybridKnowledgeRetriever._extra_metadata_to_text(different_order)
    )
    assert HybridKnowledgeRetriever._item_to_document(item) == (
        HybridKnowledgeRetriever._item_to_document(different_order)
    )

    retriever = HybridKnowledgeRetriever(items=base_items)
    retriever.add_item(item)
    assert len(retriever.kb_items) == 6
    ranking_a = retriever.search("PLC E-402 ở line L04")
    ranking_b = retriever.search("sự cố ca 2 line L04")
    for matched, metadata in (ranking_a, ranking_b):
        assert any(
            row["id"] == item.knowledge_id
            for row in metadata["bm25_ranking"]
        )
        assert any(
            row["id"] == item.knowledge_id
            for row in metadata["vector_ranking"]
        )
        assert any(
            row["id"] == item.knowledge_id
            for row in metadata["rrf_ranking"]
        )
        assert matched is not None and matched.knowledge_id == item.knowledge_id

    retriever.reindex_all(base_items)
    assert len(retriever.kb_items) == 5
    assert item.knowledge_id not in retriever.indexed_ids()

    first_meta = ranking_a[1]
    def rank_of(key):
        return next(
            index
            for index, row in enumerate(first_meta[key], start=1)
            if row["id"] == item.knowledge_id
        )

    print("[PASS] FLEXIBLE METADATA TESTS")
    print("CSV/XLSX metadata identical | NaN omitted | JSON roundtrip PASS")
    print(
        "E-402 ranks: "
        f"BM25={rank_of('bm25_ranking')} "
        f"Vector={rank_of('vector_ranking')} "
        f"RRF={rank_of('rrf_ranking')} "
        f"Retrieved={ranking_a[0].knowledge_id}"
    )


if __name__ == "__main__":
    run_flexible_metadata_tests()

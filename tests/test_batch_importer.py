"""Happy-path CSV/XLSX, alias, indexing, search, and graph tests."""

import io
import json
import time

import pandas as pd

from batch_importer import BatchKnowledgeImporter
from hybrid_retriever import HybridKnowledgeRetriever
from knowledge_graph import build_selected_knowledge_map
from schema import KnowledgeItem


def load_base_items() -> list[KnowledgeItem]:
    with open("data/mock_knowledge.json", "r", encoding="utf-8") as file:
        return [KnowledgeItem(**row) for row in json.load(file)]


def run_batch_importer_tests() -> None:
    importer = BatchKnowledgeImporter()
    base_items = load_base_items()
    existing_ids = {item.knowledge_id for item in base_items}
    with open("data/sample_maintenance_logs.csv", "rb") as file:
        csv_bytes = file.read()

    imported, errors, report = importer.parse_file(
        csv_bytes, "sample_maintenance_logs.csv", existing_ids
    )
    assert not errors
    assert report["total_rows"] == 3
    assert report["valid_rows"] == 3
    assert report["invalid_rows"] == 0
    assert report["duplicate_rows"] == 0
    assert [item.knowledge_id for item in imported] == [
        "KB-IMP-001",
        "KB-IMP-002",
        "KB-IMP-003",
    ]

    bom_items, bom_errors, _ = importer.parse_file(
        b"\xef\xbb\xbf" + csv_bytes,
        "bom.CSV",
        existing_ids,
    )
    assert not bom_errors
    assert [item.model_dump() for item in bom_items] == [
        item.model_dump() for item in imported
    ]

    dataframe = pd.read_csv(io.BytesIO(csv_bytes))
    excel_buffer = io.BytesIO()
    dataframe.to_excel(excel_buffer, index=False, engine="openpyxl")
    excel_items, excel_errors, excel_report = importer.parse_file(
        excel_buffer.getvalue(), "sample.XLSX", existing_ids
    )
    assert not excel_errors
    assert excel_report["file_type"] == "xlsx"
    assert [item.model_dump() for item in excel_items] == [
        item.model_dump() for item in imported
    ]

    vietnamese = dataframe.iloc[[0]].rename(
        columns={
            "id": "Mã",
            "title": "Tiêu đề",
            "domain": "Bộ phận",
            "equipment": "Thiết bị",
            "symptom": "Hiện tượng",
            "observation": "Quan sát",
            "root_cause": "Nguyên nhân",
            "action": "Giải pháp",
            "contraindication": "Điều cấm",
            "verification": "Nghiệm thu",
            "tags": "Từ khóa",
            "author": "Chuyên gia",
            "date": "Ngày",
        }
    )
    vietnamese_buffer = io.BytesIO()
    vietnamese.to_csv(vietnamese_buffer, index=False)
    alias_items, alias_errors, _ = importer.parse_file(
        vietnamese_buffer.getvalue(), "aliases.csv", existing_ids
    )
    assert not alias_errors
    assert alias_items[0].model_dump() == imported[0].model_dump()

    retriever = HybridKnowledgeRetriever(items=base_items)
    combined = base_items + imported
    retriever.reindex_all(combined)
    assert len(retriever.kb_items) == 8
    assert retriever.collection.count() == 8
    assert len(retriever.bm25_tokens) == 8
    query = (
        "Cụm van khí nén bị tụt áp từ 0.6 xuống 0.35 MPa "
        "sau ca chạy thì xử lý thế nào?"
    )
    started = time.perf_counter()
    matched, metadata = retriever.search(query)
    latency_ms = (time.perf_counter() - started) * 1000
    assert matched is not None
    assert matched.knowledge_id == "KB-IMP-001", metadata

    imported_maps = [build_selected_knowledge_map(item) for item in imported]
    assert {
        model["case"]["knowledge_id"] for model in imported_maps
    } == {item.knowledge_id for item in imported}

    second_items, second_errors, second_report = importer.parse_file(
        csv_bytes,
        "sample_maintenance_logs.csv",
        existing_ids | {item.knowledge_id for item in imported},
    )
    assert not second_items
    assert len(second_errors) == 3
    assert second_report["duplicate_rows"] == 3

    print("[PASS] SAFE BATCH IMPORT HAPPY PATH")
    print("CSV/BOM/XLSX/aliases: PASS | 5 + 3 = 8 synchronized items")
    print(
        f"Imported search: expected=KB-IMP-001 "
        f"retrieved={matched.knowledge_id} "
        f"distance={metadata['vector_distance']:.6f} "
        f"rrf={metadata['rrf_score']:.6f} latency_ms={latency_ms:.3f}"
    )
    print("Duplicate re-upload: 3/3 rejected | Graph imported nodes: YES")


if __name__ == "__main__":
    run_batch_importer_tests()

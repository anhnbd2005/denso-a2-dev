"""Validation and safety tests for malformed batch import data."""

import io
import json

import pandas as pd

from batch_importer import BatchKnowledgeImporter, MAX_ROWS


def to_csv_bytes(rows: list[dict]) -> bytes:
    return pd.DataFrame(rows).to_csv(index=False).encode("utf-8")


def parse_rows(importer, rows, existing_ids=None):
    return importer.parse_file(
        to_csv_bytes(rows), "test.csv", existing_ids or set()
    )


def run_invalid_tests() -> None:
    importer = BatchKnowledgeImporter()
    with open("data/sample_maintenance_logs.csv", encoding="utf-8") as file:
        sample_rows = pd.read_csv(file).to_dict(orient="records")
    with open("data/mock_knowledge.json", encoding="utf-8") as file:
        existing_ids = {row["knowledge_id"] for row in json.load(file)}

    unknown_domain = dict(sample_rows[0], id="KB-TEST-101")
    unknown_domain["domain"] = "Phòng kỹ thuật đặc biệt"
    items, errors, _ = parse_rows(importer, [unknown_domain])
    assert not items
    assert any(error["code"] == "ROW_INVALID_DOMAIN" for error in errors)

    missing_cause = dict(sample_rows[0], id="KB-TEST-102", root_cause=None)
    items, errors, _ = parse_rows(importer, [missing_cause])
    assert not items
    assert any(
        error["code"] == "MISSING_FIELD"
        and error["field"] == "root_cause_insight"
        for error in errors
    )

    duplicate_existing = dict(sample_rows[0], id=next(iter(existing_ids)))
    items, errors, report = parse_rows(
        importer, [duplicate_existing], existing_ids
    )
    assert not items and report["duplicate_rows"] == 1
    assert errors[0]["code"] == "ROW_DUPLICATE_ID"

    duplicate_one = dict(sample_rows[0], id="KB-TEST-103")
    duplicate_two = dict(sample_rows[1], id="KB-TEST-103")
    items, errors, report = parse_rows(
        importer, [duplicate_one, duplicate_two]
    )
    assert len(items) == 1 and report["duplicate_rows"] == 1
    assert any(error["code"] == "ROW_DUPLICATE_ID" for error in errors)
    assert not importer.transaction_ready(items, errors)

    items, errors, report = importer.parse_file(b"", "empty.csv", set())
    assert not items and report["total_rows"] == 0
    assert errors[0]["code"] == "EMPTY_FILE"

    items, errors, _ = importer.parse_file(b"text", "unsafe.txt", set())
    assert not items and errors[0]["code"] == "UNSUPPORTED_FILE_TYPE"

    no_warning = dict(
        sample_rows[0], id="KB-TEST-104", contraindication=None
    )
    items, errors, _ = parse_rows(importer, [no_warning])
    assert not errors and items[0].contraindications is None
    assert importer.transaction_ready(items, errors)

    extra_column = dict(
        sample_rows[0], id="KB-TEST-105", ca_sản_xuất="Ca 1"
    )
    items, errors, report = parse_rows(importer, [extra_column])
    assert not errors and len(items) == 1
    assert "ca_sản_xuất" in report["extra_metadata_columns"]
    assert items[0].extra_metadata["ca_sản_xuất"] == "Ca 1"
    assert report["ignored_columns"] == []

    collision = pd.DataFrame(
        [dict(sample_rows[0], id="KB-TEST-106", nguyên_nhân="collision")]
    ).rename(columns={"nguyên_nhân": "Nguyên nhân"})
    items, errors, _ = importer.parse_file(
        collision.to_csv(index=False).encode("utf-8"),
        "collision.csv",
        set(),
    )
    assert not items
    assert errors[0]["code"] == "DUPLICATE_MAPPED_COLUMN"

    generated = dict(sample_rows[0], id=None)
    items, errors, _ = parse_rows(
        importer,
        [generated],
        {"KB-IMP-001", "KB-IMP-100"},
    )
    assert not errors and items[0].knowledge_id == "KB-IMP-002"

    oversized = [
        dict(sample_rows[0], id=f"KB-LARGE-{number:04d}")
        for number in range(MAX_ROWS + 1)
    ]
    items, errors, _ = parse_rows(importer, oversized)
    assert not items and errors[0]["code"] == "FILE_TOO_LARGE"

    print("[PASS] SAFE BATCH IMPORT INVALID CASES")
    print(
        "Unknown domain, missing field, duplicates, empty/unsupported file, "
        "column collision and row limit: correctly rejected"
    )
    print(
        "Optional contraindication: accepted | Unknown column: reported | "
        "Generated ID collision-safe: KB-IMP-002"
    )


if __name__ == "__main__":
    run_invalid_tests()

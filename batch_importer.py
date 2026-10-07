"""Pure preview/validation pipeline for CSV and XLSX knowledge imports."""

import io
import json
import re
from datetime import date, datetime
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import numpy as np
from pydantic import ValidationError

from schema import DomainEnum, KnowledgeItem


MAX_ROWS = 1000


def normalize_column_name(value: str) -> str:
    return " ".join(str(value).strip().lower().split())


def clean_cell(value: Any) -> Optional[str]:
    if pd.isna(value):
        return None
    if isinstance(value, (datetime, date)):
        return value.date().isoformat() if isinstance(value, datetime) else value.isoformat()
    text = str(value).strip()
    return text if text else None


def normalize_extra_value(value: Any) -> Any:
    """Convert spreadsheet scalars without inventing or stringifying NaN."""
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, (dict, list)):
        return value
    if pd.isna(value):
        return None
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, str):
        text = value.strip()
        return text if text else None
    if isinstance(value, (int, float, bool)):
        return value
    return str(value).strip() or None


def generate_unique_import_id(used_ids: set[str]) -> str:
    number = 1
    while True:
        candidate = f"KB-IMP-{number:03d}"
        if candidate not in used_ids:
            return candidate
        number += 1


COLUMN_ALIASES = {
    "id": "knowledge_id",
    "mã": "knowledge_id",
    "mã sự cố": "knowledge_id",
    "knowledge_id": "knowledge_id",
    "title": "title",
    "tiêu đề": "title",
    "tên sự cố": "title",
    "domain": "domain",
    "bộ phận": "domain",
    "phân xưởng": "domain",
    "equipment": "equipment_or_station",
    "thiết bị": "equipment_or_station",
    "trạm máy": "equipment_or_station",
    "equipment_or_station": "equipment_or_station",
    "symptom": "symptom_trigger",
    "hiện tượng": "symptom_trigger",
    "mô tả lỗi": "symptom_trigger",
    "symptom_trigger": "symptom_trigger",
    "observation": "expert_observation",
    "nhận định": "expert_observation",
    "quan sát": "expert_observation",
    "expert_observation": "expert_observation",
    "root_cause": "root_cause_insight",
    "nguyên nhân": "root_cause_insight",
    "bí quyết": "root_cause_insight",
    "nguyên nhân ngầm": "root_cause_insight",
    "root_cause_insight": "root_cause_insight",
    "action": "action_solution",
    "giải pháp": "action_solution",
    "hành động": "action_solution",
    "action_solution": "action_solution",
    "contraindication": "contraindications",
    "kiêng kỵ": "contraindications",
    "điều cấm": "contraindications",
    "cảnh báo": "contraindications",
    "contraindications": "contraindications",
    "verification": "verification_standard",
    "nghiệm thu": "verification_standard",
    "tiêu chuẩn": "verification_standard",
    "verification_standard": "verification_standard",
    "tags": "tags",
    "từ khóa": "tags",
    "author": "source_author",
    "người thực hiện": "source_author",
    "chuyên gia": "source_author",
    "source_author": "source_author",
    "date": "created_date",
    "ngày": "created_date",
    "created_date": "created_date",
    "extra_metadata": "extra_metadata",
}

DOMAIN_MAP = {
    "bảo dưỡng": DomainEnum.MAINTENANCE,
    "maintenance": DomainEnum.MAINTENANCE,
    "chế tạo máy": DomainEnum.MACHINE_BUILDING,
    "machine building": DomainEnum.MACHINE_BUILDING,
    "thiết kế công đoạn": DomainEnum.PROCESS_DESIGN,
    "process design": DomainEnum.PROCESS_DESIGN,
    "hiện trường": DomainEnum.GEMBA,
    "gemba": DomainEnum.GEMBA,
    "chất lượng": DomainEnum.QUALITY,
    "quality": DomainEnum.QUALITY,
}

REQUIRED_FIELDS = [
    "title",
    "domain",
    "equipment_or_station",
    "symptom_trigger",
    "expert_observation",
    "root_cause_insight",
    "action_solution",
    "verification_standard",
    "source_author",
    "created_date",
]


class BatchKnowledgeImporter:
    @staticmethod
    def transaction_ready(
        items: List[KnowledgeItem], errors: List[Dict[str, Any]]
    ) -> bool:
        """Allow commit only when the complete uploaded batch is valid."""
        return bool(items) and not errors

    def read_file(self, file_bytes: bytes, filename: str) -> Tuple[pd.DataFrame, str]:
        lowered = filename.lower()
        if lowered.endswith(".csv"):
            try:
                text = file_bytes.decode("utf-8-sig")
            except UnicodeDecodeError:
                text = file_bytes.decode("utf-8")
            try:
                return pd.read_csv(io.StringIO(text)), "csv"
            except pd.errors.EmptyDataError:
                return pd.DataFrame(), "csv"
        if lowered.endswith(".xlsx"):
            return (
                pd.read_excel(io.BytesIO(file_bytes), engine="openpyxl"),
                "xlsx",
            )
        raise ValueError("UNSUPPORTED_FILE_TYPE")

    def normalize_columns(
        self,
        dataframe: pd.DataFrame,
    ) -> Tuple[pd.DataFrame, List[str], Dict[str, str], List[Dict[str, Any]]]:
        rename_map: Dict[Any, str] = {}
        extra_columns = []
        mapped_columns: Dict[str, str] = {}
        collisions = []
        for original in dataframe.columns:
            normalized = normalize_column_name(original)
            canonical = COLUMN_ALIASES.get(normalized)
            if canonical is None:
                stable_key = " ".join(str(original).strip().split())
                rename_map[original] = stable_key
                extra_columns.append(stable_key)
                continue
            if canonical in mapped_columns:
                collisions.append(
                    {
                        "row": None,
                        "code": "DUPLICATE_MAPPED_COLUMN",
                        "field": canonical,
                        "message": (
                            f"Columns '{mapped_columns[canonical]}' and "
                            f"'{original}' both map to '{canonical}'."
                        ),
                    }
                )
                continue
            rename_map[original] = canonical
            mapped_columns[canonical] = str(original)
        normalized = dataframe[list(rename_map)].rename(columns=rename_map)
        return normalized, extra_columns, mapped_columns, collisions

    @staticmethod
    def _parse_tags(raw_tags: Optional[str]) -> List[str]:
        if raw_tags is None:
            return []
        result = []
        seen = set()
        for value in re.split(r"[,;]", raw_tags):
            tag = value.strip()
            key = tag.casefold()
            if tag and key not in seen:
                seen.add(key)
                result.append(tag)
        return result

    def parse_row(
        self,
        row: pd.Series,
        extra_columns: List[str],
    ) -> Dict[str, Any]:
        parsed = {field: clean_cell(row.get(field)) for field in REQUIRED_FIELDS}
        parsed["knowledge_id"] = clean_cell(row.get("knowledge_id"))
        parsed["contraindications"] = clean_cell(row.get("contraindications"))
        parsed["tags"] = self._parse_tags(clean_cell(row.get("tags")))
        explicit_metadata = row.get("extra_metadata")
        if isinstance(explicit_metadata, dict):
            extra_metadata = dict(explicit_metadata)
        elif isinstance(explicit_metadata, str) and explicit_metadata.strip():
            try:
                loaded = json.loads(explicit_metadata)
                extra_metadata = loaded if isinstance(loaded, dict) else {}
            except json.JSONDecodeError:
                extra_metadata = {}
        else:
            extra_metadata = {}
        for column in extra_columns:
            value = normalize_extra_value(row.get(column))
            if value is not None:
                extra_metadata[column] = value
        parsed["extra_metadata"] = extra_metadata
        return parsed

    def validate_row(
        self,
        parsed: Dict[str, Any],
        row_number: int,
    ) -> Tuple[Optional[KnowledgeItem], List[Dict[str, Any]]]:
        errors = []
        for field in REQUIRED_FIELDS:
            if parsed.get(field) is None:
                errors.append(
                    {
                        "row": row_number,
                        "code": "MISSING_FIELD",
                        "field": field,
                        "message": f"Missing required field: {field}",
                    }
                )
        raw_domain = parsed.get("domain")
        if raw_domain is not None:
            domain = DOMAIN_MAP.get(normalize_column_name(raw_domain))
            if domain is None:
                errors.append(
                    {
                        "row": row_number,
                        "code": "ROW_INVALID_DOMAIN",
                        "field": "domain",
                        "message": f"Unknown domain: {raw_domain}",
                    }
                )
            else:
                parsed["domain"] = domain
        if errors:
            return None, errors
        try:
            return KnowledgeItem(**parsed), []
        except ValidationError as exc:
            for detail in exc.errors():
                field = ".".join(str(part) for part in detail["loc"])
                errors.append(
                    {
                        "row": row_number,
                        "code": "ROW_INVALID",
                        "field": field,
                        "message": detail["msg"],
                    }
                )
            return None, errors

    @staticmethod
    def check_duplicates(
        knowledge_id: str,
        existing_ids: set[str],
        batch_seen_ids: set[str],
        row_number: int,
    ) -> Optional[Dict[str, Any]]:
        if knowledge_id in existing_ids or knowledge_id in batch_seen_ids:
            return {
                "row": row_number,
                "code": "ROW_DUPLICATE_ID",
                "field": "knowledge_id",
                "message": f"Duplicate knowledge ID: {knowledge_id}",
            }
        return None

    def parse_file(
        self,
        file_bytes: bytes,
        filename: str,
        existing_ids: set[str],
    ) -> Tuple[List[KnowledgeItem], List[Dict[str, Any]], Dict[str, Any]]:
        base_report = {
            "total_rows": 0,
            "valid_rows": 0,
            "invalid_rows": 0,
            "duplicate_rows": 0,
            "ignored_columns": [],
            "mapped_columns": {},
            "mapped_core_columns": [],
            "extra_metadata_columns": [],
            "duplicate_mapped_columns": [],
            "file_type": None,
        }
        try:
            dataframe, file_type = self.read_file(file_bytes, filename)
        except ValueError as exc:
            error = {
                "row": None,
                "code": str(exc),
                "field": None,
                "message": f"Unsupported file type: {filename}",
            }
            return [], [error], base_report
        except Exception as exc:
            error = {
                "row": None,
                "code": "FILE_READ_ERROR",
                "field": None,
                "message": str(exc),
            }
            return [], [error], {**base_report, "file_type": file_type if 'file_type' in locals() else None}

        report = {**base_report, "total_rows": len(dataframe), "file_type": file_type}
        if dataframe.empty:
            return [], [{
                "row": None,
                "code": "EMPTY_FILE",
                "field": None,
                "message": "The uploaded file contains no data rows.",
            }], report
        if len(dataframe) > MAX_ROWS:
            return [], [{
                "row": None,
                "code": "FILE_TOO_LARGE",
                "field": None,
                "message": f"Prototype limit is {MAX_ROWS} rows.",
            }], {**report, "invalid_rows": len(dataframe)}

        normalized, extra_columns, mapped, collisions = self.normalize_columns(dataframe)
        report["mapped_columns"] = mapped
        report["mapped_core_columns"] = sorted(mapped)
        report["extra_metadata_columns"] = extra_columns
        report["duplicate_mapped_columns"] = [
            error["field"] for error in collisions
        ]
        if collisions:
            report["invalid_rows"] = len(dataframe)
            return [], collisions, report

        valid_items = []
        errors = []
        used_ids = set(existing_ids)
        batch_seen_ids: set[str] = set()
        invalid_row_numbers = set()
        duplicate_row_numbers = set()
        for offset, (_, row) in enumerate(normalized.iterrows(), start=2):
            parsed = self.parse_row(row, extra_columns)
            supplied_id = parsed["knowledge_id"]
            if supplied_id is None:
                supplied_id = generate_unique_import_id(used_ids | batch_seen_ids)
                parsed["knowledge_id"] = supplied_id
            duplicate = self.check_duplicates(
                supplied_id, existing_ids, batch_seen_ids, offset
            )
            batch_seen_ids.add(supplied_id)
            if duplicate is not None:
                errors.append(duplicate)
                invalid_row_numbers.add(offset)
                duplicate_row_numbers.add(offset)
                continue
            item, row_errors = self.validate_row(parsed, offset)
            if row_errors:
                errors.extend(row_errors)
                invalid_row_numbers.add(offset)
                continue
            valid_items.append(item)
            used_ids.add(supplied_id)

        report.update(
            {
                "valid_rows": len(valid_items),
                "invalid_rows": len(invalid_row_numbers),
                "duplicate_rows": len(duplicate_row_numbers),
            }
        )
        return valid_items, errors, report

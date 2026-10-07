"""Regression tests for the deterministic selected-item knowledge map."""

import json
import time
from html import escape

from knowledge_graph import (
    MISSING_TEXT,
    build_selected_knowledge_map,
    render_knowledge_map,
)
from schema import DomainEnum, KnowledgeItem


EXPECTED_CATEGORIES = {
    "domain",
    "equipment",
    "case",
    "symptom",
    "root_cause",
    "action",
    "contraindication",
    "verification",
}


def run_knowledge_graph_tests() -> None:
    with open("data/mock_knowledge.json", encoding="utf-8") as file:
        item = KnowledgeItem(**json.load(file)[0])

    started = time.perf_counter()
    model = build_selected_knowledge_map(item)
    html = render_knowledge_map(model)
    elapsed_ms = (time.perf_counter() - started) * 1000

    assert set(model) == EXPECTED_CATEGORIES
    assert model["case"]["knowledge_id"] == item.knowledge_id
    assert item.knowledge_id in html
    assert "Lĩnh vực" in html
    assert "Nguyên nhân / Bí quyết" in html
    assert "HAS_ROOT_CAUSE" not in html
    assert "VERIFIED_BY" not in html
    assert "vis-network" not in html
    assert "forceAtlas" not in html
    assert "@media (max-width:900px)" in html
    assert "overflow-x:hidden" in html
    assert "text-overflow:ellipsis" not in html
    assert "white-space:nowrap" not in html
    for field in (
        "symptom",
        "root_cause",
        "action",
        "contraindication",
        "verification",
    ):
        assert escape(model[field]["text"]) in html
    assert " title=" not in html
    assert elapsed_ms < 100

    hostile = KnowledgeItem(
        knowledge_id="KB-SESSION-XSS",
        title='<script>alert("x")</script> ' + "Tiêu đề rất dài " * 20,
        domain=DomainEnum.MAINTENANCE,
        equipment_or_station='Trạm <img src=x onerror="alert(1)">',
        symptom_trigger="Tín hiệu chập chờn " * 20,
        expert_observation="Quan sát tại hiện trường " * 20,
        root_cause_insight="Bụi tích tụ " * 20,
        action_solution="Vệ sinh cảm biến " * 20,
        contraindications=None,
        verification_standard=None,
        source_author="Session Expert",
        created_date="2026-10-07",
    )
    hostile_model = build_selected_knowledge_map(hostile)
    hostile_html = render_knowledge_map(hostile_model)
    assert hostile_model["verification"]["missing"] is True
    assert hostile_model["verification"]["text"] == MISSING_TEXT
    assert hostile_model["contraindication"]["missing"] is True
    assert hostile_model["case"]["text"] == " ".join(hostile.title.split())
    assert hostile_model["symptom"]["text"].endswith(
        "Quan sát tại hiện trường"
    )
    assert "…" not in hostile_html
    assert "<script>alert" not in hostile_html
    assert "&lt;script&gt;" in hostile_html
    assert "<img src=x" not in hostile_html
    assert "&lt;img src=x" in hostile_html
    assert hostile.knowledge_id in hostile_html

    print("[PASS] DETERMINISTIC SELECTED-ITEM KNOWLEDGE MAP")
    print(f"Categories: {len(model)} | render_ms={elapsed_ms:.3f}")
    print("Missing data, full-text wrapping, responsive CSS and HTML escaping: PASS")


if __name__ == "__main__":
    run_knowledge_graph_tests()

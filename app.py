"""Streamlit interface for the four core DENSO A2 workflows."""

import json
import time
from datetime import date
from html import escape
from pathlib import Path

import pandas as pd
import streamlit as st

from assistant import GroundedHybridAssistant
from batch_importer import BatchKnowledgeImporter
from hybrid_retriever import HybridKnowledgeRetriever
from knowledge_graph import build_selected_knowledge_map, render_knowledge_map
from llm_client import UniversalLLMClient
from operational_metrics import load_operational_metric
from reporting import (
    load_heldout_manifest,
    load_historical_benchmark,
    measure_frozen_hybrid_latency,
    run_frozen_benchmark,
    run_heldout_benchmark,
)
from schema import CaptureExtractionResult, DomainEnum, KnowledgeItem
from ui_components import inject_styles, render_hero, render_sidebar, section_header
from workflow_capture import WorkflowCaptureEngine


APP_DIR = Path(__file__).resolve().parent
DATA_DIR = APP_DIR / "data"

st.set_page_config(
    page_title="DENSO A2 — Master Know-How Accumulator",
    page_icon="🔴",
    layout="wide",
    initial_sidebar_state="expanded",
)
inject_styles()


@st.cache_data
def load_initial_items() -> list[KnowledgeItem]:
    items = []
    for filename in ("mock_knowledge.json", "extended_knowledge.json"):
        with (DATA_DIR / filename).open(encoding="utf-8") as file:
            items.extend(KnowledgeItem(**row) for row in json.load(file))
    return items


def initialize_state() -> None:
    defaults = {
        "current_page": "search",
        "navigation_sequence": 0,
        "kb_items": load_initial_items(),
        "hybrid_retriever": None,
        "capture_result": None,
        "batch_preview_items": [],
        "batch_preview_errors": [],
        "batch_preview_report": None,
        "batch_file_signature": None,
        "last_search_latency": None,
        "search_result": None,
        "search_llm_client": None,
        "knowledge_map_item_id": None,
        "report_benchmark_results": None,
        "report_heldout_results": None,
        "report_latency": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value
    if st.session_state.hybrid_retriever is None:
        st.session_state.hybrid_retriever = HybridKnowledgeRetriever(
            items=st.session_state.kb_items
        )


def next_knowledge_id() -> str:
    used = {item.knowledge_id for item in st.session_state.kb_items}
    number = 1
    while f"KB-NEW-{number:03d}" in used:
        number += 1
    return f"KB-NEW-{number:03d}"


def commit_items(items: list[KnowledgeItem]) -> None:
    previous_items = list(st.session_state.kb_items)
    new_items = previous_items + list(items)
    try:
        st.session_state.hybrid_retriever.reindex_all(new_items)
    except Exception:
        st.session_state.hybrid_retriever.reindex_all(previous_items)
        raise
    st.session_state.kb_items = new_items


def show_knowledge_map(knowledge_id: str) -> None:
    st.session_state.knowledge_map_item_id = knowledge_id


def hide_knowledge_map() -> None:
    st.session_state.knowledge_map_item_id = None


def clear_search_result() -> None:
    st.session_state.search_result = None


def visible(value: object) -> str:
    if value is None or value == "":
        return "Chưa có thông tin"
    return str(value)


def render_item(item: KnowledgeItem, score: float | None = None) -> None:
    score_html = "" if score is None else f"<b>Điểm RRF:</b> {score:.4f}"
    st.markdown(
        f"""
        <div class="denso-card">
          <div class="denso-eyebrow">{escape(item.knowledge_id)} · {escape(item.domain.value)}</div>
          <h3>{escape(item.title)}</h3>
          <p>{score_html}</p>
          <div class="denso-result-grid">
            <div class="denso-result-section"><strong>HIỆN TƯỢNG</strong><p>{escape(item.symptom_trigger)}</p></div>
            <div class="denso-result-section"><strong>NHẬN ĐỊNH</strong><p>{escape(item.expert_observation)}</p></div>
            <div class="denso-result-section"><strong>NGUYÊN NHÂN</strong><p>{escape(item.root_cause_insight)}</p></div>
            <div class="denso-result-section"><strong>GIẢI PHÁP</strong><p>{escape(item.action_solution)}</p></div>
            <div class="denso-result-section"><strong>CẢNH BÁO</strong><p>{escape(visible(item.contraindications))}</p></div>
          </div>
          <div class="denso-result-section"><strong>NGHIỆM THU</strong><p>{escape(visible(item.verification_standard))}</p></div>
          <div class="denso-result-section"><strong>NGUỒN</strong><p>{escape(item.source_author)} · {escape(item.created_date)} · {escape(item.equipment_or_station)}</p></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_retrieval_summary(
    result: dict,
    *,
    expanded: bool = True,
) -> None:
    latency_ms = result["retrieval_latency_seconds"] * 1000
    sources = ", ".join(result["source_ids"])
    st.success(
        f"Đã tìm thấy tri thức phù hợp · Retrieval: {latency_ms:.1f} ms · "
        f"Nguồn: {sources}"
    )
    with st.expander("Tri thức gốc được truy xuất", expanded=expanded):
        for source_item in result["source_items"]:
            render_item(source_item)


def retrieve_for_search_ui(
    assistant: GroundedHybridAssistant,
    query: str,
) -> dict:
    retrieve_method = getattr(assistant, "retrieve", None)
    if callable(retrieve_method):
        return retrieve_method(query)

    retrieval_started = time.perf_counter()
    matched, metadata = assistant.retriever.search(query)
    retrieval_latency = time.perf_counter() - retrieval_started
    if matched is None:
        return {
            "query": query,
            "is_out_of_scope": True,
            "response_text": "Chưa có tri thức phù hợp trong kho hiện tại.",
            "source_ids": [],
            "source_items": [],
            "retrieval_metadata": metadata,
            "retrieval_latency_seconds": retrieval_latency,
            "generation_latency_seconds": None,
            "generation_error": False,
            "generation_pending": False,
        }

    context_items = assistant._select_context_items(matched, metadata)
    return {
        "query": query,
        "is_out_of_scope": False,
        "response_text": "",
        "source_ids": [item.knowledge_id for item in context_items],
        "source_items": context_items,
        "retrieval_metadata": metadata,
        "retrieval_latency_seconds": retrieval_latency,
        "generation_latency_seconds": None,
        "generation_error": False,
        "generation_pending": True,
    }


def generate_for_search_ui(
    assistant: GroundedHybridAssistant,
    preliminary: dict,
    on_preview,
) -> dict:
    generate_method = getattr(assistant, "generate_from_retrieval", None)
    if callable(generate_method):
        return generate_method(preliminary, on_preview=on_preview)

    completed = assistant.ask(preliminary["query"])
    completed["retrieval_latency_seconds"] = preliminary[
        "retrieval_latency_seconds"
    ]
    completed["retrieval_metadata"] = preliminary["retrieval_metadata"]
    completed["generation_pending"] = False
    return completed


def render_search_page() -> None:
    section_header(
        "Tra cứu bí quyết chuyên gia",
        "Tìm giải pháp bằng Hybrid RAG: BM25 kết hợp semantic retrieval.",
    )
    query_col, button_col = st.columns([8, 1.4], vertical_alignment="bottom")
    with query_col:
        query = st.text_input(
            "Nội dung cần tra cứu",
            placeholder="Ví dụ: motor rung mạnh sau khi thay bạc đạn...",
            label_visibility="collapsed",
            key="search_query",
            on_change=clear_search_result,
        )
    with button_col:
        search_clicked = st.button(
            "Tra cứu", type="primary", use_container_width=True, key="search_submit"
        )
    st.caption("Gợi ý: máy rung sau bảo trì · sensor báo sai · CNC xước bề mặt")
    if search_clicked:
        if not query.strip():
            st.warning("Vui lòng nhập nội dung cần tra cứu.")
            return
        client = st.session_state.search_llm_client or UniversalLLMClient()
        assistant = GroundedHybridAssistant(
            retriever=st.session_state.hybrid_retriever,
            llm_client=client,
        )
        preliminary = retrieve_for_search_ui(assistant, query)
        st.session_state.search_result = preliminary
        st.session_state.last_search_latency = preliminary[
            "retrieval_latency_seconds"
        ]

        if preliminary["is_out_of_scope"]:
            st.info(preliminary["response_text"])
            return

        render_retrieval_summary(preliminary, expanded=True)
        st.markdown("### Câu trả lời từ trợ lý tri thức")
        generation_status = st.info("Trợ lý AI đang tổng hợp câu trả lời…")
        answer_placeholder = st.empty()

        def show_preview(preview: str) -> None:
            answer_placeholder.markdown(preview + "  ▌")

        result = generate_for_search_ui(assistant, preliminary, show_preview)
        st.session_state.search_result = result
        generation_status.empty()
        if result["generation_error"]:
            answer_placeholder.error(result["response_text"])
        else:
            answer_placeholder.markdown(result["response_text"])
            st.caption(
                "LLM generation: "
                f"{result['generation_latency_seconds']:.2f} s · "
                "đo riêng, không thuộc retrieval latency"
            )
        return

    result = st.session_state.search_result
    if result is None:
        return
    if result["is_out_of_scope"]:
        st.info(result["response_text"])
        return
    render_retrieval_summary(result, expanded=True)
    if result.get("generation_pending"):
        st.warning("Phiên tạo câu trả lời trước đó đã bị gián đoạn. Vui lòng tra cứu lại.")
        return
    if result["generation_error"]:
        st.error(result["response_text"])
        return

    st.markdown("### Câu trả lời từ trợ lý tri thức")
    st.markdown(result["response_text"])
    st.caption(
        "LLM generation: "
        f"{result['generation_latency_seconds']:.2f} s · "
        "đo riêng, không thuộc retrieval latency"
    )


def capture_fields(result: CaptureExtractionResult) -> dict:
    fields = result.extracted_fields
    domains = [domain.value for domain in DomainEnum]
    current_domain = fields.get("domain")
    domain_index = domains.index(current_domain) if current_domain in domains else 0
    return {
        "title": st.text_input("Tiêu đề *", fields.get("title") or "", key="draft_title"),
        "domain": st.selectbox("Lĩnh vực *", domains, index=domain_index, key="draft_domain"),
        "equipment_or_station": st.text_input(
            "Thiết bị / trạm *", fields.get("equipment_or_station") or "", key="draft_equipment"
        ),
        "symptom_trigger": st.text_area(
            "Hiện tượng / điều kiện kích hoạt *", fields.get("symptom_trigger") or "", key="draft_symptom"
        ),
        "expert_observation": st.text_area(
            "Nhận định chuyên gia *", fields.get("expert_observation") or "", key="draft_observation"
        ),
        "root_cause_insight": st.text_area(
            "Nguyên nhân / bí quyết *", fields.get("root_cause_insight") or "", key="draft_root_cause"
        ),
        "action_solution": st.text_area(
            "Giải pháp *", fields.get("action_solution") or "", key="draft_action"
        ),
        "contraindications": st.text_area(
            "Cảnh báo / điều cấm", fields.get("contraindications") or "", key="draft_contraindications"
        ),
        "verification_standard": st.text_area(
            "Tiêu chuẩn nghiệm thu", fields.get("verification_standard") or "", key="draft_verification"
        ),
        "tags": [
            tag.strip()
            for tag in st.text_input(
                "Từ khóa (phân tách bằng dấu phẩy)",
                ", ".join(fields.get("tags") or []),
                key="draft_tags",
            ).split(",")
            if tag.strip()
        ],
    }


def render_ai_capture() -> None:
    left, right = st.columns([43, 57], gap="large")
    with left:
        with st.container(border=True):
            st.markdown("#### Nội dung nguồn")
            author = st.text_input("Chuyên gia / tác giả", key="capture_author")
            captured_date = st.date_input("Ngày ghi nhận", value=date.today(), key="capture_date")
            raw_text = st.text_area(
                "Biên bản, ghi chú hoặc transcript",
                height=300,
                placeholder="Dán nguyên văn nội dung chuyên gia tại đây...",
                key="capture_raw_text",
            )
            if st.button(
                "Trích xuất tri thức", type="primary", use_container_width=True, key="extract_knowledge"
            ):
                try:
                    engine = WorkflowCaptureEngine(
                        mode="llm", llm_client=UniversalLLMClient()
                    )
                    st.session_state.capture_result = engine.extract_any_text(
                        raw_text,
                        next_id=next_knowledge_id(),
                        author=author,
                        created_date=captured_date.isoformat(),
                    )
                except Exception as exc:
                    st.error(f"Không thể trích xuất: {exc}")
    with right:
        with st.container(border=True):
            st.markdown("#### Bản nháp có cấu trúc")
            result = st.session_state.capture_result
            if result is None:
                st.info("Bản nháp sẽ xuất hiện tại đây sau khi trích xuất.")
                return
            edited = capture_fields(result)
            if result.missing_required_fields:
                st.warning("AI còn thiếu: " + ", ".join(result.missing_required_fields))
            if st.button(
                "Lưu vào kho tri thức", type="primary", use_container_width=True, key="save_capture"
            ):
                try:
                    if not author.strip():
                        raise ValueError("Vui lòng nhập chuyên gia / tác giả.")
                    engine = WorkflowCaptureEngine(mode="llm")
                    missing = engine._missing_fields(edited)
                    updated = CaptureExtractionResult(
                        status="NEEDS_EXPERT_INPUT" if missing else "DRAFT",
                        extracted_fields=edited,
                        missing_required_fields=missing,
                    )
                    item = engine.finalize_draft(
                        updated,
                        next_id=next_knowledge_id(),
                        author=author.strip(),
                        created_date=captured_date.isoformat(),
                    )
                    commit_items([item])
                    st.session_state.capture_result = None
                    st.success(f"Đã lưu {item.knowledge_id} và cập nhật Hybrid index.")
                except Exception as exc:
                    st.error(f"Chưa thể lưu: {exc}")


def render_batch_capture() -> None:
    importer = BatchKnowledgeImporter()
    with st.container(border=True):
        st.markdown("#### Nhập CSV / Excel")
        st.caption("Quy trình: tải tệp → chuẩn hóa → kiểm tra → xem trước → xác nhận.")
        uploaded = st.file_uploader(
            "Tệp dữ liệu", type=["csv", "xlsx"], key="batch_upload"
        )
        if uploaded is not None:
            file_bytes = uploaded.getvalue()
            signature = (uploaded.name, len(file_bytes), hash(file_bytes))
            if signature != st.session_state.batch_file_signature:
                items, errors, report = importer.parse_file(
                    file_bytes,
                    uploaded.name,
                    {item.knowledge_id for item in st.session_state.kb_items},
                )
                st.session_state.batch_preview_items = items
                st.session_state.batch_preview_errors = errors
                st.session_state.batch_preview_report = report
                st.session_state.batch_file_signature = signature
        report = st.session_state.batch_preview_report
        if report is None:
            return
        metrics = st.columns(4)
        metrics[0].metric("Tổng dòng", report["total_rows"])
        metrics[1].metric("Hợp lệ", report["valid_rows"])
        metrics[2].metric("Cần sửa", report["invalid_rows"])
        metrics[3].metric("Trùng ID", report["duplicate_rows"])
        items = st.session_state.batch_preview_items
        errors = st.session_state.batch_preview_errors
        if items:
            st.dataframe(
                pd.DataFrame(
                    {
                        "ID": item.knowledge_id,
                        "Tiêu đề": item.title,
                        "Lĩnh vực": item.domain.value,
                        "Thiết bị": item.equipment_or_station,
                    }
                    for item in items
                ),
                use_container_width=True,
                hide_index=True,
            )
        if errors:
            with st.expander(f"Xem {len(errors)} lỗi"):
                st.dataframe(pd.DataFrame(errors), use_container_width=True, hide_index=True)
            st.warning(
                "Tệp chưa được nhập. Hãy sửa toàn bộ lỗi rồi tải lại để bảo đảm "
                "giao dịch nguyên khối."
            )
        transaction_ready = importer.transaction_ready(items, errors)
        if st.button(
            f"Nhập {len(items)} bản ghi hợp lệ",
            type="primary",
            disabled=not transaction_ready,
            key="confirm_batch_import",
        ):
            commit_items(items)
            st.success(f"Đã nhập {len(items)} bản ghi và cập nhật Hybrid index.")
            st.session_state.batch_preview_items = []
            st.session_state.batch_preview_errors = []
            st.session_state.batch_preview_report = None
            st.session_state.batch_file_signature = None


def render_capture_page() -> None:
    section_header(
        "Thu thập tri thức chuyên gia",
        "Trích xuất từ ghi chú bằng AI hoặc nhập dữ liệu hàng loạt.",
    )
    mode = st.radio(
        "Phương thức",
        ["AI ghi nhận", "Nhập CSV / Excel"],
        horizontal=True,
        key="capture_mode",
    )
    if mode == "AI ghi nhận":
        render_ai_capture()
    else:
        render_batch_capture()


def render_knowledge_page() -> None:
    section_header(
        "Kho tri thức",
        "Duyệt các bí quyết đã lưu và quan sát liên kết tri thức.",
    )
    domains = ["Tất cả"] + [domain.value for domain in DomainEnum]
    filter_col, search_col = st.columns(2)
    domain_filter = filter_col.selectbox("Lĩnh vực", domains, key="knowledge_domain")
    knowledge_query = search_col.text_input(
        "Tìm theo mã, tiêu đề hoặc thiết bị / trạm", key="knowledge_query"
    )
    normalized_query = knowledge_query.strip().casefold()
    items = [
        item
        for item in st.session_state.kb_items
        if (domain_filter == "Tất cả" or item.domain.value == domain_filter)
        and (
            not normalized_query
            or normalized_query in item.knowledge_id.casefold()
            or normalized_query in item.title.casefold()
            or normalized_query in item.equipment_or_station.casefold()
        )
    ]
    visible_ids = {item.knowledge_id for item in items}
    if st.session_state.knowledge_map_item_id not in visible_ids:
        st.session_state.knowledge_map_item_id = None
    st.caption(f"{len(items)} / {len(st.session_state.kb_items)} bản ghi")
    for item in items:
        with st.expander(f"{item.knowledge_id} — {item.title}"):
            render_item(item)
            if item.extra_metadata:
                with st.expander("Thông tin mở rộng"):
                    st.dataframe(
                        pd.DataFrame(
                            [
                                {"Thuộc tính": key, "Giá trị": visible(value)}
                                for key, value in sorted(item.extra_metadata.items())
                            ]
                        ),
                        use_container_width=True,
                        hide_index=True,
                    )
            selected = st.session_state.knowledge_map_item_id == item.knowledge_id
            st.button(
                "Ẩn sơ đồ" if selected else "Xem sơ đồ tri thức",
                key=f"hide_knowledge_map_{item.knowledge_id}" if selected else f"show_knowledge_map_{item.knowledge_id}",
                on_click=hide_knowledge_map if selected else show_knowledge_map,
                args=() if selected else (item.knowledge_id,),
            )
        if st.session_state.knowledge_map_item_id == item.knowledge_id:
            st.markdown("### Sơ đồ tri thức")
            st.caption(
                "Quan hệ giữa bối cảnh, nguyên nhân và cách xử lý của tri thức đã chọn."
            )
            st.html(
                render_knowledge_map(build_selected_knowledge_map(item))
            )


REPORT_PAGE_CSS = """
<style>
.report-head { display:flex; align-items:center; gap:10px; flex-wrap:wrap; margin:22px 0 5px; }
.report-head h3 { margin:0; color:#101828; font-size:22px; }
.report-copy { margin:0 0 15px; color:#667085; font-size:14px; }
.report-badge { display:inline-flex; align-items:center; width:max-content; padding:5px 9px;
  border:1px solid #D0D5DD; border-radius:999px; background:#F9FAFB; color:#475467;
  font-size:12px; font-weight:700; }
.report-badge--red { border-color:#FECDD3; background:#FFF1F2; color:#C50011; }
.report-grid { display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:16px;
  width:100%; margin:12px 0 14px; }
.report-metric { min-width:0; padding:18px 20px; border:1px solid #E6E8EC;
  border-radius:15px; background:#FFFFFF; box-shadow:0 4px 14px rgba(16,24,40,.04); }
.report-metric__label { color:#475467; font-size:14px; font-weight:700; }
.report-metric__value { margin:5px 0 2px; color:#101828; font-size:34px; line-height:1.1;
  font-weight:800; letter-spacing:-.02em; }
.report-metric--hybrid { border-top:3px solid #E60012; }
.report-metric__note { color:#667085; font-size:13px; }
.report-disclosure { padding:13px 15px; margin:10px 0 14px; border-left:3px solid #E60012;
  border-radius:0 10px 10px 0; background:#FFF7F8; color:#475467; font-size:13px; line-height:1.55; }
.report-disclosure strong { color:#1D2939; }
.report-table-wrap { width:100%; overflow-x:hidden; margin:10px 0 18px; border:1px solid #E6E8EC;
  border-radius:14px; background:#FFFFFF; }
.report-table { width:100%; table-layout:fixed; border-collapse:collapse; }
.report-table th { padding:11px 12px; background:#F9FAFB; color:#475467; font-size:12px;
  text-align:left; font-weight:750; border-bottom:1px solid #E6E8EC; }
.report-table td { padding:11px 12px; color:#344054; font-size:13px; overflow-wrap:break-word;
  border-bottom:1px solid #EEF0F3; }
.report-table tr:last-child td { border-bottom:0; }
.report-status { display:inline-flex; padding:4px 8px; border-radius:999px; font-size:12px;
  line-height:1.2; font-weight:700; }
.report-status--ok { background:#ECFDF3; color:#027A48; }
.report-status--bad { background:#FFF1F2; color:#C50011; }
.report-empty { padding:18px; margin:10px 0 18px; border:1px dashed #D0D5DD;
  border-radius:14px; background:#F9FAFB; color:#667085; font-size:14px; }
.report-perf-grid { display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:14px;
  margin:12px 0; }
.report-perf-card { min-width:0; padding:15px 17px; border:1px solid #E6E8EC;
  border-radius:14px; background:#FFFFFF; }
.report-perf-card span { display:block; color:#667085; font-size:13px; }
.report-perf-card strong { display:block; margin-top:4px; color:#101828; font-size:25px; }
.report-scenario { padding:17px 18px; border:1px solid #E6E8EC; border-radius:15px;
  background:#FFFFFF; }
.report-scenario-grid { display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:12px;
  margin-top:13px; }
.report-scenario-item { padding:11px 13px; border-radius:10px; background:#F9FAFB; }
.report-scenario-item span { display:block; color:#667085; font-size:12px; }
.report-scenario-item strong { display:block; margin-top:3px; color:#344054; font-size:20px; }
.report-scenario p { margin:13px 0 0; color:#667085; font-size:13px; line-height:1.5; }
@media (max-width:900px) {
  .report-grid, .report-perf-grid, .report-scenario-grid { grid-template-columns:1fr; }
  .report-table th, .report-table td { padding:9px 7px; font-size:11px; }
}
</style>
"""


def _report_status_badge(passed: bool) -> str:
    tone = "ok" if passed else "bad"
    label = "Đạt" if passed else "Không đạt"
    return f'<span class="report-status report-status--{tone}">{label}</span>'


def _report_summary_html(results: dict) -> str:
    labels = {"lexical": "Lexical", "semantic": "Semantic", "hybrid": "Hybrid"}
    cards = []
    for mode_name in ("lexical", "semantic", "hybrid"):
        mode = results["modes"][mode_name]
        modifier = " report-metric--hybrid" if mode_name == "hybrid" else ""
        cards.append(
            f'<div class="report-metric{modifier}">'
            f'<div class="report-metric__label">{labels[mode_name]}</div>'
            f'<div class="report-metric__value">{mode["percentage"]:.0f}%</div>'
            f'<div class="report-metric__note">{mode["passed"]} / {mode["total"]} câu đạt</div>'
            "</div>"
        )
    return '<div class="report-grid">' + "".join(cards) + "</div>"


def _report_detail_table_html(rows: list[dict]) -> str:
    rendered_rows = []
    for row in rows:
        rendered_rows.append(
            "<tr>"
            f'<td><strong>{escape(row["test_id"])}</strong></td>'
            f'<td>{escape(row["question_type"])}</td>'
            f'<td>{_report_status_badge(row["lexical_passed"])}</td>'
            f'<td>{_report_status_badge(row["semantic_passed"])}</td>'
            f'<td>{_report_status_badge(row["hybrid_passed"])}</td>'
            "</tr>"
        )
    return (
        '<div class="report-table-wrap"><table class="report-table">'
        "<thead><tr><th>Mã câu hỏi</th><th>Loại câu hỏi</th>"
        "<th>Lexical</th><th>Semantic</th><th>Hybrid</th></tr></thead>"
        f'<tbody>{"".join(rendered_rows)}</tbody></table></div>'
    )


def _heldout_summary_html(results: dict) -> str:
    cards = (
        ("Top-1 trong phạm vi", results["in_scope_passed"], results["in_scope"]),
        ("Từ chối ngoài phạm vi", results["oos_passed"], results["oos"]),
        ("Kết quả chung", results["overall_passed"], results["questions"]),
    )
    rendered = []
    for index, (label, passed, total) in enumerate(cards):
        modifier = " report-metric--hybrid" if index == 2 else ""
        percentage = 100.0 * passed / total
        rendered.append(
            f'<div class="report-metric{modifier}">'
            f'<div class="report-metric__label">{escape(label)}</div>'
            f'<div class="report-metric__value">{percentage:.1f}%</div>'
            f'<div class="report-metric__note">{passed} / {total} câu đạt</div>'
            "</div>"
        )
    return '<div class="report-grid">' + "".join(rendered) + "</div>"


def _heldout_failure_table_html(rows: list[dict]) -> str:
    rendered_rows = []
    for row in rows:
        expected = visible(row.get("expected"))
        retrieved = visible(row.get("retrieved"))
        distance = row.get("distance")
        distance_text = "N/A" if distance is None else f"{distance:.4f}"
        rendered_rows.append(
            "<tr>"
            f'<td><strong>{escape(row["test_id"])}</strong></td>'
            f"<td>{escape(expected)}</td>"
            f"<td>{escape(retrieved)}</td>"
            f"<td>{distance_text}</td>"
            "</tr>"
        )
    return (
        '<div class="report-table-wrap"><table class="report-table">'
        "<thead><tr><th>Mã câu hỏi</th><th>Kỳ vọng</th>"
        "<th>Kết quả</th><th>Cosine distance</th></tr></thead>"
        f'<tbody>{"".join(rendered_rows)}</tbody></table></div>'
    )


def _vi_number(value: float) -> str:
    text = f"{value:g}"
    return text.replace(".", ",")


def render_report_page() -> None:
    st.markdown(REPORT_PAGE_CSS, unsafe_allow_html=True)
    section_header(
        "Báo cáo đánh giá",
        "Đánh giá chất lượng tra cứu, khả năng trả lời và hiệu năng của hệ thống.",
    )

    historical = load_historical_benchmark()
    fresh = st.session_state.report_benchmark_results
    benchmark_results = fresh or historical
    source_label = (
        "Kết quả vừa chạy trong phiên"
        if fresh is not None
        else "Kết quả lịch sử đã xác thực"
    )

    st.markdown(
        '<div class="report-head"><h3>Kết quả benchmark</h3>'
        '<span class="report-badge">Benchmark phát triển</span></div>'
        '<p class="report-copy">Đánh giá trên 10 câu hỏi do nhóm xây dựng.</p>',
        unsafe_allow_html=True,
    )
    st.caption(source_label)
    if st.button("Chạy lại đánh giá", type="primary", key="run_report_benchmark"):
        try:
            with st.spinner("Đang đánh giá hệ thống..."):
                fresh = run_frozen_benchmark()
            st.session_state.report_benchmark_results = fresh
            st.rerun()
        except Exception as exc:
            st.error(f"Không thể hoàn tất đánh giá: {exc}")

    st.markdown(_report_summary_html(benchmark_results), unsafe_allow_html=True)
    st.markdown(
        '<div class="report-disclosure"><strong>Kết quả trên benchmark phát triển đã hiệu chỉnh.</strong> '
        'Ngưỡng nhận diện ngoài phạm vi của Hybrid được hiệu chỉnh trên cùng bộ kiểm tra. '
        'Kết quả phản ánh bộ kiểm tra phát triển hiện tại. '
        'Hiệu quả trên câu hỏi mới cần được đánh giá độc lập.</div>',
        unsafe_allow_html=True,
    )
    oos = benchmark_results["oos"]
    st.caption(
        f'Nhận diện câu ngoài phạm vi: {oos["passed"]} / {oos["total"]} '
        f'· {source_label}'
    )
    with st.expander("Cách tính kết quả"):
        st.markdown(
            """
            Mỗi câu hỏi được đánh giá theo bốn thành phần: **đúng kết quả tra cứu**,
            **đúng nguyên nhân**, **đúng hành động** và **đúng nội dung an toàn**.
            Mỗi thành phần đóng góp vào điểm cuối; một câu đạt khi **điểm ≥ 0,75**.
            Tỷ lệ chung được tính bằng **số câu đạt / tổng số câu × 100%**.

            Đây là rubric đánh giá do nhóm đề xuất, không phải công thức chấm điểm chính thức của DENSO.
            """
        )

    st.markdown('<div class="report-head"><h3>Chi tiết câu hỏi kiểm tra</h3></div>', unsafe_allow_html=True)
    if fresh is None:
        st.markdown(
            '<div class="report-empty">Chưa có kết quả chi tiết trong phiên. '
            'Chọn “Chạy lại đánh giá” để tạo bảng từ đầu ra thực tế của các evaluator.</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(_report_detail_table_html(fresh["rows"]), unsafe_allow_html=True)

    heldout_manifest = load_heldout_manifest()
    heldout = st.session_state.report_heldout_results
    st.markdown(
        '<div class="report-head"><h3>Kiểm tra trên câu hỏi mới</h3>'
        '<span class="report-badge">Held-out nội bộ</span></div>'
        f'<p class="report-copy">{heldout_manifest["questions"]} câu hỏi chưa dùng để hiệu chỉnh '
        f'ngưỡng, chạy trên {heldout_manifest["kb_items"]} thẻ tri thức synthetic.</p>',
        unsafe_allow_html=True,
    )
    if st.button(
        "Chạy lại held-out" if heldout else "Chạy held-out",
        key="run_report_heldout",
    ):
        try:
            with st.spinner("Đang chạy benchmark held-out..."):
                heldout = run_heldout_benchmark()
            st.session_state.report_heldout_results = heldout
            st.rerun()
        except Exception as exc:
            st.error(f"Không thể hoàn tất held-out benchmark: {exc}")
    if heldout is None:
        st.markdown(
            '<div class="report-empty">Bộ đánh giá đã sẵn sàng: '
            f'{heldout_manifest["in_scope"]} câu trong phạm vi và '
            f'{heldout_manifest["oos"]} câu ngoài phạm vi. '
            'Chọn “Chạy held-out” để đo lại bằng code hiện tại.</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(_heldout_summary_html(heldout), unsafe_allow_html=True)
        st.caption(
            f'Độ trễ retrieval-only: trung bình {heldout["latency"]["mean_ms"]:.1f} ms · '
            f'trung vị {heldout["latency"]["median_ms"]:.1f} ms · '
            f'ngưỡng OOS không hiệu chỉnh lại: 0,55'
        )
        if heldout["failures"]:
            st.markdown(
                '<div class="report-head"><h3>Các trường hợp chưa đạt</h3></div>',
                unsafe_allow_html=True,
            )
            st.markdown(
                _heldout_failure_table_html(heldout["failures"]),
                unsafe_allow_html=True,
            )
        st.markdown(
            '<div class="report-disclosure"><strong>Phạm vi bằng chứng:</strong> '
            'Đây là benchmark nội bộ trên dữ liệu synthetic, không phải benchmark chính thức '
            'của DENSO hay accuracy trong môi trường sản xuất.</div>',
            unsafe_allow_html=True,
        )

    st.markdown(
        '<div class="report-head"><h3>Hiệu năng tra cứu</h3></div>'
        '<p class="report-copy">Đo thời gian thực hiện thao tác tìm kiếm trên bộ dữ liệu và câu hỏi cố định.</p>',
        unsafe_allow_html=True,
    )
    latency = st.session_state.report_latency
    if st.button(
        "Đo lại độ trễ" if latency else "Đo ngay",
        key="measure_report_latency",
    ):
        try:
            with st.spinner("Đang đo hiệu năng tra cứu..."):
                latency = measure_frozen_hybrid_latency()
            st.session_state.report_latency = latency
            st.rerun()
        except Exception as exc:
            st.error(f"Không thể hoàn tất phép đo: {exc}")
    if latency is None:
        st.markdown('<div class="report-empty">Chưa có phép đo.</div>', unsafe_allow_html=True)
    else:
        st.markdown(
            '<div class="report-perf-grid">'
            f'<div class="report-perf-card"><span>Độ trễ trung bình</span><strong>{latency["mean_ms"]:.1f} ms</strong></div>'
            f'<div class="report-perf-card"><span>Trung vị</span><strong>{latency["median_ms"]:.1f} ms</strong></div>'
            f'<div class="report-perf-card"><span>Số lượt đo</span><strong>{latency["sample_count"]}</strong></div>'
            "</div>",
            unsafe_allow_html=True,
        )
        st.caption(
            f'Mức nhỏ nhất: {latency["minimum_ms"]:.1f} ms · '
            f'Mức lớn nhất: {latency["maximum_ms"]:.1f} ms · Đo trong phiên hiện tại'
        )
    st.caption(
        "Độ trễ này chỉ đo thời gian tìm kiếm. Không bao gồm thời gian làm việc của chuyên gia, "
        "thời gian xử lý sự cố, thời gian sinh câu trả lời đầy đủ hoặc chu kỳ sản xuất thực tế."
    )

    st.markdown(
        '<div class="report-head"><h3>Kịch bản hiệu quả dự kiến</h3>'
        '<span class="report-badge report-badge--red">GIẢ ĐỊNH</span></div>',
        unsafe_allow_html=True,
    )
    metric = load_operational_metric(str(DATA_DIR / "operational_metrics.json"))
    if metric is not None:
        ratio = metric.improvement_ratio
        st.markdown(
            '<div class="report-scenario"><div class="report-scenario-grid">'
            f'<div class="report-scenario-item"><span>Trước áp dụng</span><strong>{_vi_number(metric.t_before_minutes)} phút</strong></div>'
            f'<div class="report-scenario-item"><span>Sau áp dụng</span><strong>{_vi_number(metric.t_after_minutes)} phút</strong></div>'
            f'<div class="report-scenario-item"><span>Tỷ lệ</span><strong>{_vi_number(ratio)}×</strong></div>'
            '</div><p>Đây là kịch bản giả định do nhóm đề xuất, chưa được đo kiểm trong môi trường thực tế tại DENSO.</p></div>',
            unsafe_allow_html=True,
        )


def render_settings_page() -> None:
    section_header("Cài đặt", "Trạng thái kết nối OpenAI-compatible.")
    client = UniversalLLMClient()
    st.text_input("Base URL", client.base_url, disabled=True)
    st.text_input("Model", client.model_name, disabled=True)
    st.text_input("API key", "Đã cấu hình" if client.api_key else "Chưa cấu hình", disabled=True)
    if st.button("Kiểm tra kết nối", key="health_check"):
        result = client.health_check()
        if result["ok"]:
            st.success(f"Kết nối thành công · phản hồi: {result.get('response', '')}")
        else:
            st.error(result["error"])


initialize_state()
current_page = st.session_state.current_page
render_sidebar(current_page)
render_hero()

if current_page == "search":
    render_search_page()
elif current_page == "capture":
    render_capture_page()
elif current_page == "knowledge":
    render_knowledge_page()
elif current_page == "report":
    render_report_page()
elif current_page == "settings":
    render_settings_page()
else:
    st.session_state.current_page = "search"
    st.rerun()

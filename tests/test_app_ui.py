"""Smoke test for the final four-page Streamlit application."""

from pathlib import Path

from streamlit.testing.v1 import AppTest


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class CountingGenerationClient:
    def __init__(self):
        self.calls = []

    def generate(self, **kwargs):
        self.calls.append(kwargs)
        return (
            "Nên kiểm tra vị trí và tình trạng bám bụi của cảm biến quang, "
            "sau đó xử lý theo tri thức đã ghi nhận."
        )


def assert_clean(app: AppTest) -> None:
    assert not app.exception, [exception.message for exception in app.exception]


def test_four_page_navigation() -> None:
    app = AppTest.from_file(
        str(PROJECT_ROOT / "app.py"), default_timeout=120
    ).run()
    assert_clean(app)

    expected = {
        "nav_search": "Tra cứu",
        "nav_capture": "Thu thập",
        "nav_knowledge": "Kho tri thức",
        "nav_report": "Báo cáo",
    }
    for key, label in expected.items():
        assert app.button(key=key).label == label
    assert {button.key for button in app.button} == {
        *expected,
        "nav_settings",
        "search_submit",
    }

    app.session_state.search_llm_client = CountingGenerationClient()
    app.text_input(key="search_query").set_value(
        "Cảm biến quang báo sai tín hiệu thì cần kiểm tra những gì?"
    ).run(timeout=120)
    app.button(key="search_submit").click().run(timeout=120)
    assert_clean(app)
    assert not app.session_state.search_result["generation_error"]
    assert app.session_state.search_result["source_ids"] == ["KB-MNT-002"]
    assert len(app.session_state.search_llm_client.calls) == 1
    app.run(timeout=120)
    assert len(app.session_state.search_llm_client.calls) == 1

    app.text_input(key="search_query").set_value(
        "Dầu thủy lực của trạm nguồn ép cao áp bị tăng nhiệt độ lên trên "
        "75 độ C sau 3 tiếng vận hành. Cần kiểm tra van điều áp hay bơm?"
    ).run(timeout=120)
    app.button(key="search_submit").click().run(timeout=120)
    assert_clean(app)
    assert app.session_state.search_result["is_out_of_scope"]
    assert len(app.session_state.search_llm_client.calls) == 1

    for key in ("nav_capture", "nav_knowledge", "nav_report", "nav_search"):
        app.button(key=key).click().run(timeout=120)
        assert_clean(app)

    app.button(key="nav_report").click().run(timeout=120)
    assert_clean(app)
    report_markup = "\n".join(markdown.value for markdown in app.markdown)
    report_captions = "\n".join(caption.value for caption in app.caption)
    assert "Báo cáo đánh giá" in report_markup
    assert "Benchmark phát triển" in report_markup
    assert "Kịch bản hiệu quả dự kiến" in report_markup
    assert "GIẢ ĐỊNH" in report_markup
    assert "Kết quả lịch sử đã xác thực" in report_captions
    assert app.button(key="run_report_benchmark").label == "Chạy lại đánh giá"
    assert app.button(key="measure_report_latency").label == "Đo ngay"

    app.button(key="nav_knowledge").click().run(timeout=120)
    assert_clean(app)
    first_item = app.session_state.kb_items[0]
    app.text_input(key="knowledge_query").set_value(first_item.knowledge_id).run(
        timeout=120
    )
    assert_clean(app)
    assert len(app.expander) == 1
    assert first_item.knowledge_id in app.expander[0].label
    app.text_input(key="knowledge_query").set_value(first_item.title).run(timeout=120)
    assert_clean(app)
    assert len(app.expander) == 1
    app.text_input(key="knowledge_query").set_value(
        first_item.equipment_or_station
    ).run(timeout=120)
    assert_clean(app)
    assert len(app.expander) == 1
    app.text_input(key="knowledge_query").set_value("").run(timeout=120)
    app.button(key=f"show_knowledge_map_{first_item.knowledge_id}").click().run(
        timeout=120
    )
    assert_clean(app)
    assert app.session_state.knowledge_map_item_id == first_item.knowledge_id
    hidden_domain = next(
        domain
        for domain in app.selectbox(key="knowledge_domain").options
        if domain not in {"Tất cả", first_item.domain.value}
    )
    app.selectbox(key="knowledge_domain").set_value(hidden_domain).run(timeout=120)
    assert_clean(app)
    assert app.session_state.knowledge_map_item_id is None
    app.selectbox(key="knowledge_domain").set_value("Tất cả").run(timeout=120)
    app.button(key=f"show_knowledge_map_{first_item.knowledge_id}").click().run(
        timeout=120
    )
    app.button(key=f"hide_knowledge_map_{first_item.knowledge_id}").click().run(
        timeout=120
    )
    assert app.session_state.knowledge_map_item_id is None

    app.button(key="nav_capture").click().run(timeout=120)
    assert_clean(app)
    assert [option for option in app.radio(key="capture_mode").options] == [
        "AI ghi nhận",
        "Nhập CSV / Excel",
    ]


if __name__ == "__main__":
    test_four_page_navigation()
    print("[PASS] FINAL FOUR-PAGE STREAMLIT UI")

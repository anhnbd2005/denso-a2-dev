"""Presentation-only components for the DENSO A2 Streamlit demo."""

from html import escape

import streamlit as st


CSS = """
<style>
:root {
  --denso-red: #E60012;
  --surface: #FFFFFF;
  --canvas: #F7F8FA;
  --text: #20242B;
  --muted: #69707D;
  --border: #E5E7EB;
  --success: #16A34A;
  --warning: #D97706;
}
.stApp { background: #F6F7F9; color: var(--text); }
.stApp, button, input, textarea { font-family: Inter, "Segoe UI", Arial, sans-serif; }
.block-container { max-width: 1450px; padding-top: 1.25rem; padding-bottom: 3rem; }
#MainMenu, footer, [data-testid="stToolbar"], [data-testid="stStatusWidget"], .stDeployButton { display:none !important; }
header[data-testid="stHeader"] { background:transparent; height:0; }
[data-testid="stSidebar"] { width:260px !important; min-width:260px !important; background:#fff; border-right:1px solid #E5E7EB; }
[data-testid="stSidebar"] > div:first-child { width:260px !important; }
[data-testid="stSidebar"] .block-container { padding:1.7rem 1rem 1.2rem; height:100vh; }
.denso-hero {
  position: relative; overflow: hidden; isolation: isolate;
  display: flex; justify-content: space-between; align-items: center; gap: 22px;
  min-height: 136px; background:
    linear-gradient(90deg,rgba(255,255,255,.99) 0%,rgba(255,255,255,.96) 43%,rgba(242,247,250,.76) 70%,rgba(228,236,240,.58) 100%),
    repeating-linear-gradient(90deg,transparent 0 78px,rgba(100,116,139,.10) 79px 80px),
    repeating-linear-gradient(0deg,transparent 0 38px,rgba(100,116,139,.08) 39px 40px);
  border: 1px solid #E1E7EF; border-radius: 16px;
  padding: 22px 26px; margin-bottom: 14px; box-shadow: 0 8px 24px rgba(30,52,77,.06);
}
.denso-hero::after {
  content:""; position:absolute; z-index:-1; right:5%; top:-72px; width:340px; height:250px;
  border:22px solid rgba(230,0,18,.11); border-left-width:48px; border-radius:52% 18% 42% 20%;
  transform:rotate(-14deg); box-shadow:90px 74px 0 -44px rgba(230,0,18,.16);
}
.denso-kicker { display:none; }
.denso-title { margin: 2px 0 5px; font-size: clamp(29px, 2.15vw, 40px); line-height: 1.08; font-weight: 760; white-space: nowrap; }
.denso-subtitle { margin: 0; color: var(--muted); font-size: 15px; }
.denso-chip {
  display: inline-flex; align-items: center; gap: 7px; border: 1px solid var(--border);
  border-radius: 999px; padding: 6px 10px; background: #fff; color: #4B5563;
  font-size: 12px; font-weight: 650; white-space: nowrap;
}
.denso-chip::before { content: ""; width: 7px; height: 7px; border-radius: 50%; background: #9CA3AF; }
.denso-chip.ok::before { background: var(--success); }
.denso-chip.red::before { background: var(--denso-red); }
.denso-chip.warn::before { background: var(--warning); }
.denso-section { margin: 10px 0 14px; }
.denso-section h2 { font-size: 26px; line-height: 1.2; margin: 0 0 3px; }
.denso-section p { color: var(--muted); font-size: 14px; margin: 0; }
.denso-card {
  background: #fff; border: 1px solid var(--border); border-radius: 14px;
  padding: 19px 21px; box-shadow: 0 6px 20px rgba(30,52,77,.055); margin-bottom: 12px;
}
.denso-card h3 { margin: 0 0 6px; font-size: 19px; }
.denso-eyebrow { color: var(--muted); font-size: 11px; letter-spacing: .08em; font-weight: 750; }
.denso-result-section { border-top: 1px solid #EEF0F3; padding-top: 12px; margin-top: 12px; }
.denso-result-section strong { display:block; font-size: 11px; letter-spacing: .09em; color:#626A77; margin-bottom:4px; }
.denso-result-section p { margin: 0; font-size: 15px; line-height: 1.55; }
.denso-result-grid { display:grid; grid-template-columns:repeat(5,minmax(0,1fr)); gap:0; margin-top:12px; border-top:1px solid #EEF0F3; }
.denso-result-grid .denso-result-section { border-top:0; padding:14px 16px 4px; }
.denso-result-grid .denso-result-section + .denso-result-section { border-left:1px solid #E8EBF0; }
.denso-results-title { font-size:27px; line-height:1.2; margin:16px 0 10px; }
.denso-score-track { height:7px; border-radius:999px; overflow:hidden; background:#E8ECF2; margin-top:2px; }
.denso-score-track span { display:block; height:100%; border-radius:inherit; background:#E60012; }
.denso-sidebar-brand { padding:2px 10px 22px; margin-bottom:14px; }
.denso-sidebar-brand b { display:block; color:var(--denso-red); font-size:34px; line-height:1; font-style:italic; font-weight:900; letter-spacing:-.04em; }
.denso-sidebar-brand span { display:block; max-width:190px; margin-top:14px; color:#17213A; font-size:16px; line-height:1.25; letter-spacing:0; text-transform:none; font-weight:750; }
.denso-sidebar-brand small { display:block; margin-top:6px; color:#98A2B3; font-size:10px; letter-spacing:.08em; text-transform:uppercase; }
.denso-sidebar-spacer { height:clamp(130px,28vh,330px); }
[data-testid="stSidebar"] [data-testid="stButton"] { margin-bottom:9px; }
[data-testid="stSidebar"] [data-testid="stButton"] button {
  min-height:52px; height:52px; width:100%; justify-content:flex-start;
  padding:0 18px; border:0; border-radius:10px; box-shadow:none; font-weight:650;
}
[data-testid="stSidebar"] [data-testid="stButton"] button p {
  width:100%; margin:0; text-align:left;
}
[data-testid="stSidebar"] [data-testid="stButton"] button [data-testid="stMarkdownContainer"],
[data-testid="stSidebar"] [data-testid="stButton"] button > div {
  width:100%; justify-content:flex-start; text-align:left;
}
[data-testid="stSidebar"] [data-testid="stButton"] button[kind="secondary"] { background:transparent; color:#25324A; }
[data-testid="stSidebar"] [data-testid="stButton"] button[kind="secondary"]:hover { background:#FFF1F2; color:#B80010; }
[data-testid="stSidebar"] [data-testid="stButton"] button[kind="primary"] { background:#E60012 !important; color:#fff !important; }
.denso-mini-label { color: var(--muted); font-size: 10px; font-weight: 800; letter-spacing: .1em; margin: 12px 0 5px; }
.denso-footer-note { color: #8A919D; font-size: 11px; line-height: 1.55; margin-top: 18px; }
button[kind="primary"] { background: linear-gradient(135deg,#E60012,#F2253A) !important; border-color: var(--denso-red) !important; box-shadow:0 6px 14px rgba(230,0,18,.16); }
button[kind="primary"]:hover { filter:brightness(.96); }
[data-testid="stTextInput"] input { min-height:54px; border-radius:10px; background:#fff; }
[data-testid="stVerticalBlockBorderWrapper"] { background:#fff; border-radius:14px; box-shadow:0 2px 10px rgba(16,24,40,.04); }
[data-testid="stMetric"] { background:#fff; border:1px solid var(--border); border-radius:12px; padding:12px 14px; }
@media (max-width: 1500px) {
  .denso-title { font-size:29px; }
}
@media (max-width: 900px) {
  .denso-hero { align-items:flex-start; flex-direction:column; }
  .denso-title { white-space:normal; }
  .denso-result-grid { grid-template-columns:1fr; }
  .denso-result-grid .denso-result-section + .denso-result-section { border-left:0; }
}
</style>
"""


def inject_global_css() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


def inject_styles() -> None:
    inject_global_css()


def _navigate(page: str) -> None:
    if st.session_state.get("current_page") != page:
        st.session_state.current_page = page
        sequence = st.session_state.get("navigation_sequence", 0) + 1
        st.session_state.navigation_sequence = sequence
        st.session_state.scroll_to_top_token = sequence


def render_sidebar(current_page: str) -> None:
    items = [
        ("search", "Tra cứu"),
        ("capture", "Thu thập"),
        ("knowledge", "Kho tri thức"),
        ("report", "Báo cáo"),
    ]
    with st.sidebar:
        st.markdown(
            '<div class="denso-sidebar-brand"><b>DENSO</b>'
            '<span>Master Know-How<br>Accumulator</span>'
            '<small>A2 Knowledge System</small></div>',
            unsafe_allow_html=True,
        )
        for page, label in items:
            st.button(
                label,
                key=f"nav_{page}",
                type="primary" if current_page == page else "secondary",
                use_container_width=True,
                on_click=_navigate,
                args=(page,),
            )
        st.markdown('<div class="denso-sidebar-spacer"></div>', unsafe_allow_html=True)
        st.button(
            "Cài đặt nâng cao",
            key="nav_settings",
            type="primary" if current_page == "settings" else "secondary",
            use_container_width=True,
            on_click=_navigate,
            args=("settings",),
        )


def render_hero() -> None:
    st.markdown(
        f"""
        <div class="denso-hero">
          <div>
            <div class="denso-kicker">DENSO A2</div>
            <div class="denso-title">Master Know-How Accumulator</div>
            <p class="denso-subtitle">Số hóa, kiểm chứng và tái sử dụng bí quyết chuyên gia.</p>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def section_header(title: str, subtitle: str) -> None:
    st.markdown(
        f"""
        <div class="denso-section">
          <h2>{escape(title)}</h2>
          <p>{escape(subtitle)}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

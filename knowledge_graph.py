"""Deterministic selected-item knowledge map rendering."""

from html import escape
from typing import Any, Dict

from schema import KnowledgeItem


MISSING_TEXT = "Chưa có thông tin"


def _node(label: str, value: str | None) -> Dict[str, Any]:
    full_text = " ".join((value or "").split())
    missing = not bool(full_text)
    return {
        "label": label,
        "text": full_text or MISSING_TEXT,
        "missing": missing,
    }


def build_selected_knowledge_map(item: KnowledgeItem) -> Dict[str, Dict[str, Any]]:
    """Build the eight human-facing blocks for one knowledge item."""
    if not isinstance(item, KnowledgeItem):
        raise TypeError("Map input phải là một KnowledgeItem")
    return {
        "domain": _node("Lĩnh vực", item.domain.value),
        "equipment": _node("Thiết bị / trạm", item.equipment_or_station),
        "case": {
            **_node("Mã tri thức", item.title),
            "knowledge_id": item.knowledge_id,
        },
        "symptom": _node(
            "Hiện tượng / quan sát",
            f"{item.symptom_trigger} {item.expert_observation}",
        ),
        "root_cause": _node("Nguyên nhân / Bí quyết", item.root_cause_insight),
        "action": _node("Giải pháp", item.action_solution),
        "contraindication": _node("Điều cần tránh", item.contraindications),
        "verification": _node("Xác minh", item.verification_standard),
    }


def _card(node: Dict[str, Any], tone: str, *, case: bool = False) -> str:
    missing_class = " km-card--missing" if node["missing"] else ""
    case_class = " km-card--case" if case else ""
    identity = ""
    if case:
        identity = f'<div class="km-id">{escape(node["knowledge_id"])}</div>'
    return (
        f'<article class="km-card km-card--{tone}{missing_class}{case_class}">'
        f'<div class="km-label">{escape(node["label"])}</div>'
        f'{identity}<div class="km-body">{escape(node["text"])}</div>'
        "</article>"
    )


def _down_arrow(css_class: str = "") -> str:
    return f'<div class="km-arrow km-arrow--down {css_class}" aria-hidden="true"></div>'


def render_knowledge_map(model: Dict[str, Dict[str, Any]]) -> str:
    """Render an in-memory, responsive HTML/CSS map with SVG connectors."""
    expected = {
        "domain",
        "equipment",
        "case",
        "symptom",
        "root_cause",
        "action",
        "contraindication",
        "verification",
    }
    if set(model) != expected:
        raise ValueError("Knowledge map model không đúng tám nhóm nội dung")

    return f"""
<style>
.knowledge-map {{
  --km-line:#98A2B3;
  width:100%; box-sizing:border-box; overflow-x:hidden;
  padding:22px; margin:8px 0 18px; border:1px solid #E6E8EC;
  border-radius:16px; background:#FFFFFF;
  box-shadow:0 8px 24px rgba(16,24,40,.045);
}}
.knowledge-map * {{ box-sizing:border-box; }}
.km-context {{
  display:grid; grid-template-columns:minmax(0,1fr) 56px minmax(0,1fr);
  align-items:center; width:min(760px,100%); margin:0 auto;
}}
.km-card {{
  min-width:0; padding:14px 16px; border:1px solid #D0D5DD;
  border-radius:15px; background:#F9FAFB;
  box-shadow:0 2px 8px rgba(16,24,40,.035);
}}
.km-label {{
  margin-bottom:6px; color:#475467; font-size:13px; line-height:1.25;
  font-weight:700; letter-spacing:.01em;
}}
.km-body {{
  height:auto; color:#1D2939; font-size:14px; line-height:1.42;
  font-weight:500; white-space:normal; overflow-wrap:break-word;
  word-break:normal;
}}
.km-context .km-body {{
  display:-webkit-box; overflow:hidden; -webkit-box-orient:vertical;
  -webkit-line-clamp:2;
}}
.km-card--domain {{ background:#EEF4FF; border-color:#B2CCFF; }}
.km-card--equipment {{ background:#F2F4F7; border-color:#D0D5DD; }}
.km-card--case {{
  max-width:820px; margin:0 auto; padding:17px 20px;
  background:#FFF1F2; border-color:#FDA4AF;
  border-left:5px solid #E60012;
  box-shadow:0 5px 16px rgba(230,0,18,.08);
}}
.km-card--case .km-body {{ font-size:16px; font-weight:650; }}
.km-id {{ margin-bottom:5px; color:#C50011; font-size:15px; font-weight:800; }}
.km-card--symptom {{ background:#F9FAFB; border-color:#D0D5DD; }}
.km-card--cause {{ background:#FFF7E6; border-color:#FEC84B; }}
.km-card--action {{ background:#ECFDF3; border-color:#6CE9A6; }}
.km-card--avoid {{ background:#FFF4ED; border-color:#FDB022; }}
.km-card--verify {{ background:#F4F3FF; border-color:#C3B5FD; }}
.km-card--missing {{
  background:#F9FAFB; border-style:dashed; border-color:#D0D5DD;
}}
.km-card--missing .km-body {{ color:#667085; font-weight:450; }}
.km-arrow {{ position:relative; display:block; color:var(--km-line); }}
.km-arrow--horizontal {{ width:42px; height:24px; margin:auto; }}
.km-arrow--horizontal::before {{
  content:""; position:absolute; left:2px; right:8px; top:11px;
  border-top:1.5px solid var(--km-line);
}}
.km-arrow--horizontal::after {{
  content:""; position:absolute; right:4px; top:7px; width:8px; height:8px;
  border-top:1.5px solid var(--km-line); border-right:1.5px solid var(--km-line);
  transform:rotate(45deg);
}}
.km-arrow--down {{ width:24px; height:28px; margin:5px auto; }}
.km-arrow--down::before {{
  content:""; position:absolute; left:11px; top:1px; height:20px;
  border-left:1.5px solid var(--km-line);
}}
.km-arrow--down::after {{
  content:""; position:absolute; left:7px; top:15px; width:8px; height:8px;
  border-right:1.5px solid var(--km-line); border-bottom:1.5px solid var(--km-line);
  transform:rotate(45deg);
}}
.km-branch-connector {{ position:relative; width:100%; height:52px; }}
.km-branch-connector::before {{
  content:""; position:absolute; left:50%; top:0; height:21px;
  border-left:1.5px solid var(--km-line);
}}
.km-branch-bar {{
  position:absolute; left:16.67%; right:16.67%; top:20px;
  border-top:1.5px solid var(--km-line);
}}
.km-branch-stem {{
  position:absolute; top:20px; height:25px; border-left:1.5px solid var(--km-line);
}}
.km-branch-stem::after {{
  content:""; position:absolute; left:-5px; bottom:0; width:8px; height:8px;
  border-right:1.5px solid var(--km-line); border-bottom:1.5px solid var(--km-line);
  transform:rotate(45deg);
}}
.km-branch-stem--left {{ left:16.67%; }}
.km-branch-stem--center {{ left:50%; }}
.km-branch-stem--right {{ left:83.33%; }}
.km-branches {{
  display:grid; grid-template-columns:repeat(3,minmax(0,1fr));
  gap:18px; align-items:start;
}}
.km-branches > .km-card {{ height:auto; min-height:92px; }}
.km-action-stack {{ display:flex; min-width:0; flex-direction:column; }}
@media (max-width:900px) {{
  .knowledge-map {{ padding:18px; }}
  .km-context {{ grid-template-columns:1fr; gap:0; }}
  .km-arrow--horizontal {{ transform:rotate(90deg); margin:4px auto; }}
  .km-branch-connector {{ width:24px; height:28px; margin:5px auto; }}
  .km-branch-connector::before {{ height:20px; }}
  .km-branch-bar, .km-branch-stem {{ display:none; }}
  .km-branches {{ grid-template-columns:1fr; gap:12px; }}
  .km-branches > .km-card + .km-card::before,
  .km-action-stack::before {{ content:""; display:block; }}
}}
@media (max-width:560px) {{
  .knowledge-map {{ padding:14px; border-radius:14px; }}
  .km-card {{ padding:13px 14px; }}
  .km-card--case {{ padding:15px 16px; }}
}}
</style>
<section class="knowledge-map" aria-label="Sơ đồ tri thức đã chọn">
  <div class="km-context">
    {_card(model["domain"], "domain")}
    <div class="km-arrow km-arrow--horizontal" aria-hidden="true"></div>
    {_card(model["equipment"], "equipment")}
  </div>
  {_down_arrow()}
  {_card(model["case"], "case", case=True)}
  <div class="km-branch-connector" aria-hidden="true">
    <span class="km-branch-bar"></span>
    <span class="km-branch-stem km-branch-stem--left"></span>
    <span class="km-branch-stem km-branch-stem--center"></span>
    <span class="km-branch-stem km-branch-stem--right"></span>
  </div>
  <div class="km-branches">
    {_card(model["symptom"], "symptom")}
    {_card(model["root_cause"], "cause")}
    <div class="km-action-stack">
      {_card(model["action"], "action")}
      {_down_arrow()}
      {_card(model["contraindication"], "avoid")}
      {_down_arrow()}
      {_card(model["verification"], "verify")}
    </div>
  </div>
</section>
""".strip()

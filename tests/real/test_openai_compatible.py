"""Environment-driven real smoke test; never prints the API key."""

import os

from llm_client import UniversalLLMClient
from workflow_capture import WorkflowCaptureEngine


def run_real_smoke_test() -> None:
    client = UniversalLLMClient()
    print(f"Base URL: {client.base_url}")
    print(f"Model: {client.model_name}")
    print(f"Provider: {client.provider}")
    print(f"API key configured: {'YES' if os.getenv('OPENAI_API_KEY') else 'NO'}")
    if client.client is None:
        raise RuntimeError(
            "Set OPENAI_API_KEY before running the real proxy smoke test."
        )

    health = client.health_check()
    print(f"Health check: {'PASS' if health['ok'] else 'FAIL'}")
    if not health["ok"]:
        raise RuntimeError(health["error"])
    print(f"Response: {health.get('response', '')}")
    print("Free text:", client.generate("Trả lời đúng một từ: OK", "Trả lời ngắn."))

    engine = WorkflowCaptureEngine(mode="llm", llm_client=client)
    result = engine.extract_any_text(
        (
            "Sau khi thay bạc đạn motor băng tải số 2, máy rung mạnh. "
            "Kiểm tra thấy hai bulông chân đế phía trái bị lỏng. "
            "Siết chéo bốn bulông với lực 32 Nm thì hết rung. "
            "Không được siết hết lực từng con một vì dễ làm lệch chân đế."
        ),
        next_id="KB-SMOKE-001",
        author="Smoke Test Expert",
        created_date="2026-10-06",
    )
    print(f"Structured extraction status: {result.status}")
    print(f"Missing required fields: {result.missing_required_fields}")
    print(f"JSON path: {client.last_json_mode}")
    if result.status != "DRAFT" or result.knowledge_item is None:
        raise AssertionError("Real extraction did not produce a complete draft")
    item = result.knowledge_item
    supported_text = " ".join(
        [
            item.equipment_or_station,
            item.symptom_trigger,
            item.expert_observation,
            item.root_cause_insight,
            item.action_solution,
            item.contraindications or "",
        ]
    ).casefold()
    for marker in ("băng tải", "32 nm", "siết chéo", "hết rung"):
        if marker not in supported_text:
            raise AssertionError(f"Missing supported source fact: {marker}")
    if item.verification_standard:
        raise AssertionError(
            "Observed outcome was incorrectly promoted to a formal verification standard"
        )
    forbidden = ("50 chu kỳ", "100 chu kỳ", "8 giờ")
    serialized = " ".join(str(value) for value in item.model_dump().values()).casefold()
    if any(marker in serialized for marker in forbidden):
        raise AssertionError("Structured extraction fabricated a verification fact")


if __name__ == "__main__":
    run_real_smoke_test()

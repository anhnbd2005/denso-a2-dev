from typing import Any, Dict, Optional

from llm_client import UniversalLLMClient
from schema import (
    CaptureExtractionResult,
    DomainEnum,
    KnowledgeItem,
)


SYSTEM_EXTRACTION_PROMPT = """
Bạn là chuyên gia số hóa tri thức sản xuất.

Nhiệm vụ:
Đọc biên bản làm việc, transcript, ghi chú hoặc nội dung thô
của chuyên gia và trích xuất thành một KnowledgeItem chuẩn.

Nguyên tắc trích xuất:

1. root_cause_insight
Phải thể hiện nguyên nhân/bí quyết được hỗ trợ bởi input.
Không được tự phát minh nguyên nhân không xuất hiện trong nguồn.

2. contraindications
Phải tìm những nội dung dạng:
- không được;
- cấm;
- tránh;
- tuyệt đối không;
- cảnh báo rủi ro.

Nếu source không có thông tin này:
để None.

3. action_solution
Trích xuất hành động cụ thể.
Giữ lại các thông số quan trọng nếu source có:
- nhiệt độ;
- áp suất;
- thời gian;
- lượng dầu;
- kích thước;
- giá trị kỹ thuật.

Không tự thêm thông số.

4. domain
Chỉ chọn một trong năm DomainEnum hiện có:
- Bảo dưỡng
- Chế tạo máy
- Thiết kế công đoạn
- Hiện trường
- Chất lượng

5. Không được phát minh dữ kiện kỹ thuật. Field không được nguồn hỗ trợ
phải trả về null, không dùng placeholder và không tự thêm tiêu chí nghiệm thu.

verification_standard chỉ dùng cho tiêu chí, ngưỡng hoặc quy trình xác minh
được nguồn nêu rõ. Một kết quả quan sát sau thao tác (ví dụ "máy hết rung")
không tự động trở thành tiêu chuẩn xác minh. Giữ kết quả quan sát này trong
expert_observation dưới dạng "Kết quả quan sát: ..." (hoặc
trong action_solution nếu phù hợp), đồng thời để verification_standard là null
nếu nguồn không có tiêu chí xác minh chính thức.

6. Trả về đúng một JSON object có cấu trúc:
{
  "extracted_fields": {
    "title": string|null,
    "domain": string|null,
    "equipment_or_station": string|null,
    "symptom_trigger": string|null,
    "expert_observation": string|null,
    "root_cause_insight": string|null,
    "action_solution": string|null,
    "contraindications": string|null,
    "verification_standard": string|null,
    "tags": [string]
  }
}
Không có markdown hoặc giải thích ngoài JSON.
""".strip()


REQUIRED_CAPTURE_FIELDS = (
    "title",
    "domain",
    "equipment_or_station",
    "symptom_trigger",
    "expert_observation",
    "root_cause_insight",
    "action_solution",
)
CAPTURE_FIELDS = set(REQUIRED_CAPTURE_FIELDS) | {
    "contraindications",
    "verification_standard",
    "tags",
}


class WorkflowCaptureEngine:
    def __init__(
        self,
        mode: str = "deterministic",
        llm_client: Optional[UniversalLLMClient] = None,
    ):
        if mode not in {
            "deterministic",
            "llm",
        }:
            raise ValueError(
                "mode phải là 'deterministic' hoặc 'llm'"
            )

        self.mode = mode
        self.llm_client = llm_client

    def extract(
        self,
        raw_session: Dict[str, Any],
        next_id: str,
    ) -> KnowledgeItem:

        if self.mode != "deterministic":
            raise RuntimeError(
                "extract() is reserved for deterministic fixtures; "
                "use extract_any_text() for OpenAI-compatible capture."
            )

        self._validate_raw_session(
            raw_session
        )

        return self._extract_deterministic(
            raw_session,
            next_id,
        )

    @staticmethod
    def _validate_raw_session(
        raw: Dict[str, Any],
    ) -> None:

        required_fields = {
            "session_id",
            "source_type",
            "author",
            "date",
            "raw_content",
        }

        missing = (
            required_fields
            - set(raw.keys())
        )

        if missing:
            raise ValueError(
                "Raw session thiếu field: "
                + ", ".join(sorted(missing))
            )

        if not str(
            raw["raw_content"]
        ).strip():
            raise ValueError(
                "raw_content không được rỗng"
            )

    def _extract_deterministic(
        self,
        raw: Dict[str, Any],
        next_id: str,
    ) -> KnowledgeItem:

        sid = raw["session_id"]

        if sid == "RAW-SES-001":
            return KnowledgeItem(
                knowledge_id=next_id,
                title=(
                    "Xử lý kẹt xi-lanh cụm cấp phôi "
                    "do dùng sai chất bôi trơn"
                ),
                domain=DomainEnum.MAINTENANCE,
                equipment_or_station=(
                    "Trạm cấp phôi dán tem tự động số 3"
                ),
                symptom_trigger=(
                    "Cứ chạy khoảng 20 sản phẩm "
                    "là kẹt cứng xi-lanh đẩy phôi."
                ),
                expert_observation=(
                    "Thanh trượt có lớp mỡ cũ "
                    "bị keo lại làm tăng ma sát."
                ),
                root_cause_insight=(
                    "Dùng nhầm mỡ bò công nghiệp đặc "
                    "cho cụm trượt có khe hở chính xác "
                    "cực nhỏ (0.02mm), gây keo dính "
                    "khi hoạt động."
                ),
                action_solution=(
                    "1. Dùng giẻ tẩm cồn lau sạch "
                    "lớp mỡ cũ. "
                    "2. Tra 3 giọt dầu nhớt trắng "
                    "chuyên dụng số 10. "
                    "3. Cài đặt áp suất khí nén "
                    "về 0.45 MPa."
                ),
                contraindications=(
                    "Tuyệt đối không dùng mỡ đặc "
                    "và không xịt RP7 vì dung môi "
                    "sẽ làm trương nở phớt cao su "
                    "hỏng cụm xi-lanh."
                ),
                verification_standard=(
                    "Vận hành liên tục 100 chu kỳ "
                    "không phát sinh kẹt phôi; "
                    "áp suất khí duy trì ổn định "
                    "0.45 MPa."
                ),
                tags=[
                    "xi_lanh",
                    "boi_tron",
                    "ket_phoi",
                    "dan_tem",
                    "bao_duong",
                ],
                source_author=raw["author"],
                created_date=raw["date"],
            )

        if sid == "RAW-SES-002":
            return KnowledgeItem(
                knowledge_id=next_id,
                title=(
                    "Kiểm soát bavia mép cắt dập nhôm "
                    "bằng gia nhiệt dầu bôi trơn"
                ),
                domain=DomainEnum.QUALITY,
                equipment_or_station=(
                    "Line dập số 4 - "
                    "Cụm khuôn dập nắp chụp"
                ),
                symptom_trigger=(
                    "Bavia mép cắt chi tiết nắp chụp "
                    "vượt quá 0.08mm trong giai đoạn "
                    "đầu ca chạy."
                ),
                expert_observation=(
                    "Dao cắt không mòn; "
                    "phôi nhôm có hiện tượng dính cối "
                    "khi nhiệt độ dầu thấp."
                ),
                root_cause_insight=(
                    "Dầu bôi trơn dập đầu ca "
                    "dưới 25 độ C có độ nhớt cao, "
                    "cản trở thoát phoi và làm "
                    "dính phôi nhôm vào cối."
                ),
                action_solution=(
                    "Bật hệ thống sấy dầu tuần hoàn "
                    "trước khi bắt đầu ca 20 phút "
                    "để nhiệt độ dầu bôi trơn "
                    "đạt 40-45 độ C."
                ),
                contraindications=(
                    "Cấm tự ý điều chỉnh giảm khe hở "
                    "cối dao (clearance) khi thấy "
                    "bavia đầu ca vì sẽ gây sứt mẻ "
                    "lưỡi dao hợp kim."
                ),
                verification_standard=(
                    "Bavia mép cắt kiểm tra "
                    "đạt dưới 0.03mm ngay từ "
                    "10 sản phẩm đầu tiên của ca."
                ),
                tags=[
                    "khuon_dap",
                    "bavia",
                    "dau_boi_tron",
                    "nhiet_do",
                    "chat_luong",
                ],
                source_author=raw["author"],
                created_date=raw["date"],
            )

        raise ValueError(
            "Deterministic fixture extractor "
            f"chưa có mapping cho session: {sid}"
        )

    @staticmethod
    def _missing_fields(
        fields: Dict[str, Any],
    ) -> list[str]:
        missing = []
        valid_domains = {domain.value for domain in DomainEnum}
        for field in REQUIRED_CAPTURE_FIELDS:
            value = fields.get(field)
            if value is None or (isinstance(value, str) and not value.strip()):
                missing.append(field)
                continue
            if field == "domain" and value not in valid_domains:
                missing.append(field)
                continue
        return missing

    def finalize_draft(
        self,
        result: CaptureExtractionResult,
        next_id: str,
        author: str,
        created_date: str,
    ) -> KnowledgeItem:
        if result.missing_required_fields:
            raise ValueError(
                "Required expert input missing: "
                + ", ".join(result.missing_required_fields)
            )
        fields = {
            field: value
            for field, value in result.extracted_fields.items()
            if field in CAPTURE_FIELDS
        }
        fields["tags"] = fields.get("tags") or []
        fields["contraindications"] = fields.get("contraindications") or None
        fields["verification_standard"] = (
            fields.get("verification_standard") or None
        )
        return KnowledgeItem(
            knowledge_id=next_id,
            source_author=author,
            created_date=created_date,
            **fields,
        )

    def extract_any_text(
        self,
        raw_text: str,
        next_id: str,
        author: str,
        created_date: str,
    ) -> CaptureExtractionResult:
        if not raw_text.strip():
            raise ValueError("raw_text không được rỗng")
        if not author.strip():
            raise ValueError("author không được rỗng")
        if not created_date.strip():
            raise ValueError("created_date không được rỗng")
        if self.llm_client is None:
            raise RuntimeError("OpenAI-compatible API is not configured.")

        payload = self.llm_client.generate_json(
            prompt=(
                "Trích xuất bản nháp tri thức từ nội dung nguồn sau. "
                "Không phát minh thông tin còn thiếu.\n\nSOURCE:\n"
                + raw_text
            ),
            system_prompt=SYSTEM_EXTRACTION_PROMPT,
        )
        raw_fields = payload.get("extracted_fields")
        if not isinstance(raw_fields, dict):
            raw_fields = {}
        fields = {
            str(field): value
            for field, value in raw_fields.items()
            if str(field) in CAPTURE_FIELDS
        }
        missing = self._missing_fields(fields)
        result = CaptureExtractionResult(
            status=("NEEDS_EXPERT_INPUT" if missing else "DRAFT"),
            extracted_fields=fields,
            missing_required_fields=missing,
        )
        if not missing:
            result.knowledge_item = self.finalize_draft(
                result,
                next_id=next_id,
                author=author.strip(),
                created_date=created_date,
            )
        return result

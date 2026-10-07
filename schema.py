"""Core data models for the DENSO A2 knowledge workflow."""

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field


class DomainEnum(str, Enum):
    MAINTENANCE = "Bảo dưỡng"
    MACHINE_BUILDING = "Chế tạo máy"
    PROCESS_DESIGN = "Thiết kế công đoạn"
    GEMBA = "Hiện trường"
    QUALITY = "Chất lượng"


class KnowledgeItem(BaseModel):
    """One reusable unit of expert know-how."""

    model_config = ConfigDict(extra="allow")

    knowledge_id: str
    title: str
    domain: DomainEnum
    equipment_or_station: str
    symptom_trigger: str
    expert_observation: str
    root_cause_insight: str
    action_solution: str
    contraindications: Optional[str] = None
    verification_standard: Optional[str] = None
    tags: List[str] = Field(default_factory=list)
    source_author: str
    created_date: str
    extra_metadata: Dict[str, Any] = Field(default_factory=dict)


class CaptureExtractionResult(BaseModel):
    """Editable structured draft returned by the capture workflow."""

    status: str
    extracted_fields: Dict[str, Any]
    missing_required_fields: List[str] = Field(default_factory=list)
    knowledge_item: Optional[KnowledgeItem] = None


class KnowledgeBase(BaseModel):
    items: List[KnowledgeItem]


class EvaluationRubric(BaseModel):
    must_include_root_cause: str
    must_include_action: str
    must_include_contraindication: Optional[str] = None


class TestCase(BaseModel):
    test_id: str
    user_role: str
    query: str
    expected_kb_id: Optional[str]
    is_out_of_scope: bool = False
    rubric: EvaluationRubric

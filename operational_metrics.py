"""Evidence-labelled operational KPI calculations, separate from retrieval."""

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional


def compute_improvement_ratio(
    t_before_minutes: Optional[float],
    t_after_minutes: Optional[float],
) -> Optional[float]:
    if t_before_minutes is None or t_after_minutes is None:
        return None
    if t_before_minutes <= 0 or t_after_minutes <= 0:
        raise ValueError("Operational times must both be greater than zero")
    return t_before_minutes / t_after_minutes


@dataclass(frozen=True)
class OperationalMetricInput:
    t_before_minutes: Optional[float]
    t_after_minutes: Optional[float]
    status: str
    source_note: str
    retrieval_latency_seconds: Optional[float] = None

    def __post_init__(self) -> None:
        if self.status not in {"measured", "assumed"}:
            raise ValueError("status must be 'measured' or 'assumed'")
        compute_improvement_ratio(
            self.t_before_minutes,
            self.t_after_minutes,
        )
        if (
            self.retrieval_latency_seconds is not None
            and self.retrieval_latency_seconds <= 0
        ):
            raise ValueError("retrieval_latency_seconds must be positive")

    @property
    def improvement_ratio(self) -> Optional[float]:
        return compute_improvement_ratio(
            self.t_before_minutes,
            self.t_after_minutes,
        )

    @property
    def claim_type(self) -> str:
        return "MEASURED" if self.status == "measured" else "SCENARIO"

    def format_result(self) -> dict:
        return {
            **asdict(self),
            "improvement_ratio": self.improvement_ratio,
            "claim_type": self.claim_type,
        }


def load_operational_metric(
    path: str = "data/operational_metrics.json",
) -> Optional[OperationalMetricInput]:
    config_path = Path(path)
    if not config_path.exists():
        return None
    with config_path.open("r", encoding="utf-8") as file:
        return OperationalMetricInput(**json.load(file))

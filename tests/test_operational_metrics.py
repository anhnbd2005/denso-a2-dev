"""Tests for evidence-labelled operational metrics."""

from operational_metrics import (
    OperationalMetricInput,
    compute_improvement_ratio,
    load_operational_metric,
)


def run_operational_metrics_tests() -> None:
    assert compute_improvement_ratio(45.0, 1.5) == 30.0
    assert compute_improvement_ratio(10.0, 5.0) == 2.0
    assert compute_improvement_ratio(None, 5.0) is None
    assert compute_improvement_ratio(5.0, None) is None
    for before, after in [(0, 1), (1, 0), (-1, 1), (1, -1)]:
        try:
            compute_improvement_ratio(before, after)
        except ValueError:
            pass
        else:
            raise AssertionError("Non-positive time must be rejected")

    assumed = load_operational_metric()
    assert assumed is not None
    assert assumed.status == "assumed"
    assert assumed.claim_type == "SCENARIO"
    assert assumed.improvement_ratio == 30.0
    assert assumed.retrieval_latency_seconds is None

    measured = OperationalMetricInput(
        t_before_minutes=10,
        t_after_minutes=5,
        status="measured",
        source_note="Synthetic unit-test measurement label.",
    )
    assert measured.claim_type == "MEASURED"
    assert measured.format_result()["improvement_ratio"] == 2.0

    print("[PASS] OPERATIONAL METRICS TESTS")
    print("45/1.5=30x SCENARIO | 10/5=2x | invalid/missing safe")


if __name__ == "__main__":
    run_operational_metrics_tests()

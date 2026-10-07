"""Regression checks for frozen report evidence and fresh measurement."""

import hashlib
from pathlib import Path

from reporting import (
    BENCHMARK_PATH,
    KB_PATH,
    load_historical_benchmark,
    measure_frozen_hybrid_latency,
    run_frozen_benchmark,
)


EXPECTED_KB_HASH = "33A1702DCDB518D978E2DF5A8F53E2A6F645302374203D9237115FAA18BD57C8"
EXPECTED_BENCHMARK_HASH = "AAB15F735848ED05E41AA0ED1E8F811E82C8E396FBC34B0DF310CFCF9F2696DA"


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def run_reporting_tests() -> None:
    historical = load_historical_benchmark()
    assert historical["kind"] == "historical"
    assert historical["total_tests"] == 10
    assert historical["rows"] == []
    assert historical["modes"]["lexical"] == {
        "passed": 9,
        "total": 10,
        "percentage": 90.0,
    }
    assert historical["modes"]["semantic"] == {
        "passed": 8,
        "total": 10,
        "percentage": 80.0,
    }
    assert historical["modes"]["hybrid"]["passed"] == 10
    assert historical["modes"]["hybrid"]["percentage"] == 100.0
    assert historical["modes"]["hybrid"]["calibrated"] is True

    fresh = run_frozen_benchmark()
    assert fresh["kind"] == "fresh"
    assert fresh["total_tests"] == 10
    assert len(fresh["rows"]) == 10
    assert {row["test_id"] for row in fresh["rows"]} == {
        f"TC-{number:03d}" for number in range(1, 11)
    }
    assert fresh["modes"]["lexical"]["passed"] == 9
    assert fresh["modes"]["semantic"]["passed"] == 8
    assert fresh["modes"]["hybrid"]["passed"] == 10
    assert fresh["oos"] == {"passed": 2, "total": 2}
    assert fresh["latency"]["source"] == "measured_current_session"
    assert fresh["latency"]["sample_count"] == 10
    assert fresh["latency"]["mean_ms"] > 0

    measured = measure_frozen_hybrid_latency()
    assert measured["sample_count"] == 10
    assert measured["mean_ms"] > 0
    assert measured["median_ms"] > 0
    assert measured["minimum_ms"] <= measured["maximum_ms"]

    assert file_hash(KB_PATH) == EXPECTED_KB_HASH
    assert file_hash(BENCHMARK_PATH) == EXPECTED_BENCHMARK_HASH
    print("[PASS] REPORT PAGE EVIDENCE TESTS")
    print("Historical 90/80/100 | fresh rows=10 | OOS=2/2")
    print(
        f"Measured retrieval: n={measured['sample_count']} "
        f"mean={measured['mean_ms']:.3f} ms "
        f"median={measured['median_ms']:.3f} ms"
    )


if __name__ == "__main__":
    run_reporting_tests()

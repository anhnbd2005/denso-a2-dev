"""Frozen benchmark execution and report-safe result shaping."""

import io
import json
import statistics
import time
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any, Callable, Dict, List

from evaluate_hybrid import run_hybrid_evaluation
from evaluate_rag import run_rag_evaluation
from evaluate_usefulness import run_evaluation
from hybrid_retriever import HybridKnowledgeRetriever


APP_DIR = Path(__file__).resolve().parent
DATA_DIR = APP_DIR / "data"
KB_PATH = DATA_DIR / "mock_knowledge.json"
BENCHMARK_PATH = DATA_DIR / "test_benchmark.json"
SNAPSHOT_PATH = DATA_DIR / "historical_benchmark_snapshot.json"


def _load_benchmark_rows() -> List[Dict[str, Any]]:
    with BENCHMARK_PATH.open(encoding="utf-8") as file:
        return json.load(file)


def _with_percentages(summary: Dict[str, Any]) -> Dict[str, Any]:
    result = json.loads(json.dumps(summary))
    for mode in result["modes"].values():
        mode["percentage"] = 100.0 * mode["passed"] / mode["total"]
    return result


def load_historical_benchmark() -> Dict[str, Any]:
    """Load the validated snapshot and verify its frozen denominator."""
    with SNAPSHOT_PATH.open(encoding="utf-8") as file:
        snapshot = json.load(file)
    benchmark_count = len(_load_benchmark_rows())
    if snapshot["total_tests"] != benchmark_count:
        raise ValueError("Historical snapshot denominator differs from benchmark")
    if any(mode["total"] != benchmark_count for mode in snapshot["modes"].values()):
        raise ValueError("Historical mode denominator differs from benchmark")
    snapshot["kind"] = "historical"
    snapshot["rows"] = []
    return _with_percentages(snapshot)


def _run_quietly(evaluator: Callable[[], List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    output = io.StringIO()
    with redirect_stdout(output):
        return evaluator()


def _mode_summary(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    total = len(results)
    passed = sum(bool(result["is_useful"]) for result in results)
    return {
        "passed": passed,
        "total": total,
        "percentage": 100.0 * passed / total,
    }


def summarize_latency(latencies_ms: List[float]) -> Dict[str, Any]:
    if not latencies_ms:
        raise ValueError("Latency measurement requires at least one sample")
    return {
        "mean_ms": statistics.mean(latencies_ms),
        "median_ms": statistics.median(latencies_ms),
        "minimum_ms": min(latencies_ms),
        "maximum_ms": max(latencies_ms),
        "sample_count": len(latencies_ms),
        "source": "measured_current_session",
    }


def run_frozen_benchmark() -> Dict[str, Any]:
    """Run all established evaluators against frozen files only."""
    benchmark = _load_benchmark_rows()
    lexical = _run_quietly(run_evaluation)
    semantic = _run_quietly(run_rag_evaluation)
    hybrid = _run_quietly(run_hybrid_evaluation)
    expected_ids = [row["test_id"] for row in benchmark]
    indexes = {
        "lexical": {row["test_id"]: row for row in lexical},
        "semantic": {row["test_id"]: row for row in semantic},
        "hybrid": {row["test_id"]: row for row in hybrid},
    }
    if any(set(index) != set(expected_ids) for index in indexes.values()):
        raise ValueError("Evaluator output does not match the frozen benchmark")

    rows = []
    for benchmark_row in benchmark:
        test_id = benchmark_row["test_id"]
        rows.append(
            {
                "test_id": test_id,
                "question_type": (
                    "Ngoài phạm vi"
                    if benchmark_row.get("is_out_of_scope")
                    else "Trong phạm vi"
                ),
                "lexical_passed": bool(indexes["lexical"][test_id]["is_useful"]),
                "semantic_passed": bool(indexes["semantic"][test_id]["is_useful"]),
                "hybrid_passed": bool(indexes["hybrid"][test_id]["is_useful"]),
                "hybrid_score": float(indexes["hybrid"][test_id]["total"]),
            }
        )

    oos_results = [
        row for row in hybrid if row["scope"] == "out-of-scope"
    ]
    oos_passed = sum(
        row["retrieved"] is None and row["scores"]["retrieval"] == 1.0
        for row in oos_results
    )
    return {
        "kind": "fresh",
        "status": "fresh",
        "total_tests": len(benchmark),
        "modes": {
            "lexical": _mode_summary(lexical),
            "semantic": _mode_summary(semantic),
            "hybrid": {**_mode_summary(hybrid), "calibrated": True},
        },
        "oos": {"passed": oos_passed, "total": len(oos_results)},
        "rows": rows,
        "latency": summarize_latency(
            [float(row["latency_ms"]) for row in hybrid]
        ),
    }


def measure_frozen_hybrid_latency() -> Dict[str, Any]:
    """Measure search-only latency using frozen KB and frozen questions."""
    retriever = HybridKnowledgeRetriever(kb_path=str(KB_PATH))
    latencies_ms = []
    for row in _load_benchmark_rows():
        started = time.perf_counter()
        retriever.search(row["query"])
        latencies_ms.append((time.perf_counter() - started) * 1000)
    return summarize_latency(latencies_ms)

"""Read-only distance and scope-decision audit for calibrated OOS Gate V2."""

import json

from hybrid_retriever import HybridKnowledgeRetriever
from schema import TestCase


def run_oos_gate_v2_audit() -> None:
    with open("data/test_benchmark.json", "r", encoding="utf-8") as file:
        cases = [TestCase(**row) for row in json.load(file)]
    retriever = HybridKnowledgeRetriever(oos_distance_threshold=0.55)

    in_scope_distances = []
    oos_distances = []
    decisions = []
    for case in cases:
        item, metadata = retriever.search(case.query)
        distance = metadata["vector_distance"]
        rejected = metadata["reason"] == "OUT_OF_SCOPE_REJECTED"
        assert rejected == (distance > retriever.oos_distance_threshold)
        (oos_distances if case.is_out_of_scope else in_scope_distances).append(
            distance
        )
        decisions.append((case, item, distance, rejected))
        print(
            f"{case.test_id} scope="
            f"{'out-of-scope' if case.is_out_of_scope else 'in-scope'} "
            f"distance={distance:.6f} threshold=0.55 "
            f"decision={'REJECTED' if rejected else 'ACCEPTED'}"
        )

    max_in_scope = max(in_scope_distances)
    min_oos = min(oos_distances)
    print(
        f"max_in_scope={max_in_scope:.6f} min_oos={min_oos:.6f} "
        f"gap={min_oos - max_in_scope:.6f} "
        f"midpoint={(min_oos + max_in_scope) / 2:.6f}"
    )
    print(
        "Actual correct scope decisions: "
        f"{sum((item is None) == case.is_out_of_scope for case, item, _, _ in decisions)}"
        f"/{len(decisions)}"
    )


if __name__ == "__main__":
    run_oos_gate_v2_audit()

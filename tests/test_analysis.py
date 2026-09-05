from __future__ import annotations

from scripts.analyze_results import _aggregate_group, immutable_evidence


def test_query_level_aggregates_do_not_count_repetitions_as_queries() -> None:
    common = {
        "method": "group_pruning_beta0",
        "method_kind": "pruned",
        "dataset_hash": "dataset",
        "split_id": "split",
        "k": 1,
        "comparison_valid": True,
        "recall_at_k_exact_full_visible": 1.0,
        "recall_at_k_same_c_reference": 1.0,
        "id_symmetric_difference_exact": 0,
        "rank_position_difference_exact": 0,
        "requested_beta_l2": 0.0,
        "certified_beta_l2": 0.0,
        "observed_beta_upper_l2": 0.0,
        "groups_scanned": 1,
        "groups_skipped": 1,
        "component_timings_ns": {},
        "contract_violation": False,
        "baseline_validation_failure": False,
    }
    rows = [
        {
            **common,
            "query_id": 0,
            "query_position": 0,
            "repetition": repetition,
            "end_to_end_latency_ns": latency,
            "micro_latency_ns": latency,
            "tau_returned_l2": 10.0,
            "delta_exact_neighbor_count": 2,
            "delta_exact_neighbors_captured": 1,
            "delta_exact_neighbor_capture_rate": 0.5,
            "delta_in_same_c_reference_topk": True,
            "fallback": True,
            "fallback_reason": "x",
        }
        for repetition, latency in ((0, 100.0), (1, 300.0))
    ]
    rows.append(
        {
            **common,
            "query_id": 1,
            "query_position": 1,
            "repetition": 0,
            "end_to_end_latency_ns": 400.0,
            "micro_latency_ns": 400.0,
            "tau_returned_l2": 20.0,
            "delta_exact_neighbor_count": 0,
            "delta_exact_neighbors_captured": 0,
            "delta_exact_neighbor_capture_rate": None,
            "delta_in_same_c_reference_topk": False,
            "fallback": False,
            "fallback_reason": None,
        }
    )

    summary = _aggregate_group(("run", "experiment", "method"), rows, True, [])

    assert summary["query_rows"] == 3
    assert summary["unique_test_queries"] == 2
    assert summary["tau_returned_l2"]["count"] == 2
    assert summary["tau_returned_l2"]["p50"] == 15.0
    assert summary["delta_exact_neighbor_count_total"] == 2
    assert summary["delta_exact_neighbors_captured_total"] == 1
    assert summary["delta_exact_neighbor_capture_rate"] == 0.5
    assert summary["delta_exact_neighbor_capture_rate_macro"] == 0.5
    assert summary["delta_influence_query_fraction"] == 0.5
    assert summary["delta_influence_end_to_end_p50_ms"] == 0.0002
    assert summary["non_delta_end_to_end_p50_ms"] == 0.0004
    assert summary["fallback_count"] == 1
    assert summary["fallback_rate"] == 0.5
    assert summary["fallback_reason_counts"] == {"x": 1}
    assert summary["fallback_request_count"] == 2
    assert summary["fallback_request_rate"] == 2 / 3
    assert summary["fallback_reason_request_counts"] == {"x": 2}


def test_immutable_evidence_omits_local_paths_and_analysis_clock() -> None:
    first = {
        "schema_version": 1,
        "generated_at_utc": "first",
        "input": "/first/results/runs",
        "runs": [{"run_id": "run", "run_dir": "/first/run"}],
        "excluded_incomplete_runs": [{"run_id": "old", "run_dir": "/first/old"}],
        "totals": {"runs": 1},
    }
    second = {
        **first,
        "generated_at_utc": "second",
        "input": "/second/results/runs",
        "runs": [{"run_id": "run", "run_dir": "/second/run"}],
        "excluded_incomplete_runs": [{"run_id": "old", "run_dir": "/second/old"}],
    }

    assert immutable_evidence(first) == immutable_evidence(second)

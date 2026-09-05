"""Pre-registered, fail-closed policy contract for the Issue #3 final."""

from __future__ import annotations

from typing import Any, Mapping


class FinalPolicyError(ValueError):
    """Raised when the final policy differs from the pre-registered contract."""


EXPECTED_PRIMARY = {
    "dataset": "sift",
    "query_interval": [1200, 2200],
    "query_count": 1000,
    "independence_label": "fresh_pre_registered_holdout",
    "gate_eligible": True,
}

EXPECTED_ANCHORS = [
    {
        "validation_experiment_id": "sift-initial",
        "role": "fixed_anchor_sift_initial",
        "query_interval": [1200, 2200],
        "independence_label": "fresh_pre_registered_holdout",
        "gate_eligible": False,
    },
    {
        "validation_experiment_id": "gist-initial",
        "role": "fixed_anchor_gist_initial",
        "query_interval": [0, 1000],
        "independence_label": "reused_non_independent",
        "gate_eligible": False,
    },
    {
        "validation_experiment_id": "sift-delta-100000",
        "role": "fixed_anchor_sift_delta_100000",
        "query_interval": [1200, 2200],
        "independence_label": "fresh_pre_registered_holdout",
        "gate_eligible": False,
    },
]

EXPECTED_FINAL_GATE = {
    "primary_metric": "api_wall",
    "performance_paired_ci_lower_strictly_above": 1.0,
    "minimum_geometric_mean_speedup_vs_each_baseline": 1.10,
    "maximum_p95_latency_ratio_vs_each_baseline": 1.05,
    "required_baselines": ["F", "A"],
    "required_process_sessions": 3,
    "required_primary_queries": 1000,
    "require_A_quality_match": True,
    "engineering_GO_requires_delta_influence_subset": True,
    "engineering_GO_requires_build_cost_evidence": True,
}

EXPECTED_HNSW = {
    "run_only_after_fresh_final_performance_pass": True,
    "ef_search_values": [128, 512],
    "methods": ["base_plus_delta_hnsw", "full_population_hnsw"],
    "role": "approximate_reference_only_not_in_final_gate",
}

EXPECTED_LOCKED_EXECUTION = {
    "timed_methods": {
        "F": "native_F",
        "A": "faiss_A",
        "P": "validation_selected_native_P",
    },
    "kernel": {
        "backend": "pybind11_cpp17",
        "language_standard": "C++17",
        "numeric_contract": "interval_error_bounds_and_exact_boundary_ordering",
        "threshold_mode": "heap",
        "rank_strategy": "adaptive",
        "python_fallback_permitted": False,
    },
    "fallback_policy": {
        "validation_not_passed": "no_final_lock_or_fresh_holdout_load",
        "evidence_integrity_failure": "abort_inconclusive_fail_closed",
        "final_performance_not_passed": "NO_GO_and_do_not_run_HNSW",
        "duplicate_visible_logical_id_or_multiple_visible_version": (
            "certified_native_full_scan_before_skip"
        ),
        "population_below_k": "not_applicable",
    },
}


def _same_typed_json(actual: Any, expected: Any) -> bool:
    """JSON equality that does not treat bool as numeric 0/1."""

    if type(actual) is not type(expected):
        return False
    if isinstance(expected, dict):
        return set(actual) == set(expected) and all(
            _same_typed_json(actual[key], value) for key, value in expected.items()
        )
    if isinstance(expected, list):
        return len(actual) == len(expected) and all(
            _same_typed_json(left, right) for left, right in zip(actual, expected)
        )
    return bool(actual == expected)


def issue_fixed_decision_values() -> dict[str, Any]:
    """Return the values that must have existed before validation timing."""

    return {
        "primary_metric": "api_wall",
        "performance_required_baselines": ["F", "A"],
        "performance_paired_ci_lower_strictly_above": 1.0,
        "engineering_GO_minimum_geometric_mean_speedup": 1.10,
        "engineering_GO_maximum_p95_latency_ratio": 1.05,
        "engineering_GO_requires_A_quality_match": True,
        "required_process_sessions": 3,
        "fresh_primary_dataset": "sift",
        "fresh_primary_query_interval": [1200, 2200],
        "fresh_primary_query_count": 1000,
    }


def validate_issue_fixed_decision_values(value: Any) -> None:
    if not _same_typed_json(value, issue_fixed_decision_values()):
        raise FinalPolicyError("Issue-fixed final decision values differ")


def validate_final_policy(policy: Mapping[str, Any]) -> None:
    """Require the complete Issue-fixed policy, not merely matching hashes."""

    expected_keys = {
        "schema_version",
        "study_id",
        "validation_config",
        "holdout_manifest",
        "correctness_evidence",
        "output_root",
        "hnsw_output_root",
        "pre_hnsw_authorization",
        "process_session_seeds",
        "repetitions_per_session",
        "query_block_size",
        "warmup_queries",
        "bootstrap_resamples",
        "bootstrap_seed",
        "primary",
        "fixed_anchors",
        "final_gate",
        "hnsw_references",
        "locked_execution",
    }
    if set(policy) != expected_keys:
        raise FinalPolicyError("final policy schema/field set differs from registration")
    if not _same_typed_json(policy.get("schema_version"), 1) or not (
        _same_typed_json(policy.get("study_id"), "issue-3-native-recheck")
    ):
        raise FinalPolicyError("unsupported final policy schema/study")
    expected_paths = {
        "validation_config": "configs/native_recheck_validation.json",
        "holdout_manifest": "results/native_recheck_evidence/holdout_manifest.json",
        "correctness_evidence": "results/native_recheck_evidence/native_correctness.json",
        "output_root": "results/native_recheck_runs/final",
        "hnsw_output_root": "results/native_recheck_runs/final_hnsw",
        "pre_hnsw_authorization": (
            "results/native_recheck_evidence/pre_hnsw_authorization.json"
        ),
    }
    if any(policy.get(key) != value for key, value in expected_paths.items()):
        raise FinalPolicyError("final policy path contract differs from registration")
    if not _same_typed_json(policy.get("process_session_seeds"), [0, 1, 2]):
        raise FinalPolicyError("final process seeds must be exactly 0,1,2")
    if not _same_typed_json(policy.get("repetitions_per_session"), 3):
        raise FinalPolicyError("final repetitions must be exactly three per session")
    if (
        isinstance(policy.get("bootstrap_resamples"), bool)
        or not isinstance(policy.get("bootstrap_resamples"), int)
        or int(policy["bootstrap_resamples"]) < 2000
    ):
        raise FinalPolicyError("final bootstrap resamples must be at least 2000")
    if not _same_typed_json(policy.get("primary"), EXPECTED_PRIMARY):
        raise FinalPolicyError("final primary differs from the Issue-fixed fresh holdout")
    if not _same_typed_json(policy.get("fixed_anchors"), EXPECTED_ANCHORS):
        raise FinalPolicyError("final anchors differ from the Issue-fixed anchors")
    if not _same_typed_json(policy.get("final_gate"), EXPECTED_FINAL_GATE):
        raise FinalPolicyError("final decision thresholds differ from the Issue-fixed gate")
    if not _same_typed_json(policy.get("hnsw_references"), EXPECTED_HNSW):
        raise FinalPolicyError("final HNSW policy differs from the Issue-fixed references")
    if not _same_typed_json(policy.get("locked_execution"), EXPECTED_LOCKED_EXECUTION):
        raise FinalPolicyError("final kernel/method/fallback contract differs")
    for field in ("query_block_size", "warmup_queries", "bootstrap_seed"):
        value = policy.get(field)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise FinalPolicyError(f"final policy {field} is not a non-negative integer")
    if int(policy["query_block_size"]) == 0:
        raise FinalPolicyError("final query block size must be positive")


__all__ = [
    "FinalPolicyError",
    "issue_fixed_decision_values",
    "validate_final_policy",
    "validate_issue_fixed_decision_values",
]

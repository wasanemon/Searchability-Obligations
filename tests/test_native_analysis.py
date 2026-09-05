from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

import pytest

from searchability.artifacts import (
    atomic_write_json,
    atomic_write_jsonl,
    file_sha256,
    object_sha256,
)
from searchability.native_analysis import (
    MIN_BOOTSTRAP_RESAMPLES,
    NativeEvidenceError,
    analyze_native_path,
    analyze_native_rows,
    evaluate_validation_gate,
    load_native_evidence,
    paired_speedup,
)
from searchability.final_policy import issue_fixed_decision_values


_BINARY_SHA256 = "a" * 64


def _row(
    method: str,
    role: str,
    query: int,
    latency: float,
    *,
    experiment: str = "real-delta",
    session: str = "session-0",
    repetition: int = 0,
    influence: bool | None = True,
    n_delta: int = 1000,
    beta: float = 0.0,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "experiment_id": experiment,
        "method": method,
        "method_role": role,
        "session_id": session,
        "query_id": query,
        "query_position": query,
        "repetition": repetition,
        "micro_latency_ns": latency,
        "composed_e2e_latency_ns": latency,
        "api_wall_latency_ns": latency,
        "dataset_type": "texmex",
        "dataset_name": "sift",
        "n_delta": n_delta,
        "n_test": 4,
        "requested_beta_l2": beta,
    }
    if influence is not None:
        row["delta_influence"] = influence
    if role in {"F", "N", "P"}:
        row.update(
            {
                "execution_backend": "native_cpp_pybind11",
                "native_binary_sha256": _BINARY_SHA256,
                "native_call_count": 1,
                "native_call_index": 0,
                "python_fallback_used": False,
                "contract_violation": False,
                "contract_valid": True,
                "same_frozen_candidate_object_used": True,
                "api_candidate_hash_match": True,
                "candidate_set_id_or_hash": f"candidate-{query}",
                "receipt_candidate_set_id_or_hash": f"candidate-{query}",
                "receipt_requested_beta_l2": beta,
            }
        )
        if beta == 0.0:
            row["same_c_order_match"] = True
        else:
            row["observed_beta_upper_l2"] = beta / 4.0
            row["certified_beta_l2"] = beta / 2.0
    return row


def _gate_rows(*, n_delta: int = 1000) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for query in range(4):
        influence = query == 0
        rows.extend(
            [
                _row("F", "F", query, 10.0, influence=influence, n_delta=n_delta),
                _row("A", "A", query, 12.0, influence=influence, n_delta=n_delta),
                _row("N", "N", query, 8.0, influence=influence, n_delta=n_delta),
                _row(
                    "P-beta0",
                    "P",
                    query,
                    5.0,
                    influence=influence,
                    n_delta=n_delta,
                ),
            ]
        )
    return rows


def _write_completed_run(
    root: Path,
    rows: list[dict[str, Any]],
    *,
    run_id: str = "validation-run",
    phase: str = "validation",
    evidence_role: str = "validation",
    test_query_offset: int = 0,
) -> Path:
    run_dir = root / run_id
    source_config_hash = "c" * 64
    source_config_file_hash = "e" * 64
    final_policy_path = (
        Path(__file__).resolve().parents[1]
        / "configs"
        / "native_recheck_final_policy.json"
    )
    final_policy = json.loads(final_policy_path.read_text(encoding="utf-8"))
    final_policy_identity = {
        "path": str(final_policy_path),
        "file_sha256": file_sha256(final_policy_path),
        "object_sha256": object_sha256(final_policy),
    }
    holdout_path = root / "holdout_manifest.json"
    if not holdout_path.exists():
        atomic_write_json(
            holdout_path,
            {
                "schema_version": 1,
                "study_id": "issue-3-native-recheck",
                "sift": {
                    "historically_used_query_ids": {"interval": [0, 1200]},
                    "native_validation_selection": {
                        "beta_calibration_interval": [4, 8],
                        "measurement_interval": [0, 4],
                    },
                    "pre_registered_fresh_final_holdout": {
                        "interval": [1200, 2200],
                        "loaded_by_native_validation": False,
                        "selected_before_native_validation_timing": True,
                        "disjoint_from_historically_used_union": True,
                    },
                },
                "gist": {
                    "independence_label": "reused_non_independent",
                    "fresh_holdout_claim_permitted": False,
                },
                "selection_integrity": {
                    "validation_source_config_object_sha256": source_config_hash,
                    "validation_source_config_file_sha256": source_config_file_hash,
                    "final_policy_path": final_policy_identity["path"],
                    "final_policy_file_sha256": final_policy_identity["file_sha256"],
                    "final_policy_object_sha256": final_policy_identity["object_sha256"],
                },
                "pre_registered_final_policy": {
                    **final_policy_identity,
                    "issue_fixed_decision_values": issue_fixed_decision_values(),
                },
            },
        )
    effective = {
        "schema_version": 2,
        "phase": phase,
        "evidence_role": evidence_role,
        "source_config_object_hash": source_config_hash,
        "source_config_file_sha256": source_config_file_hash,
        "process_sessions": 1,
        "independent_oracle_queries": 1,
        "lb_audit_queries": 0,
        "holdout_policy": {
            "manifest": str(holdout_path),
            "validation_must_not_load_sift_query_ids_at_or_above": 1200,
            "sift_final_holdout_is_pre_registered_but_not_loaded": [1200, 2200],
            "gist_measurement_is_reused_non_independent": True,
            "pre_registered_final_policy": final_policy_identity,
        },
        "repetitions": 1,
        "experiments": [
            {
                "id": "real-delta",
                "axis": "test-fixture",
                "beta_factors": [0.0],
                "kernel_ablations": False,
                "dataset": {
                    "type": "texmex",
                    "name": "sift",
                    "n_delta": 1000,
                    "n_validation": 4,
                    "n_test": 4,
                    "validation_query_offset": 4,
                    "test_query_offset": test_query_offset,
                },
            }
        ],
    }
    config_hash = object_sha256(effective)
    implementation_hash = "b" * 64
    dataset_hash = "d" * 64
    split_unsigned = {
        "schema_version": 2,
        "experiment_id": "real-delta",
        "dataset_name": "texmex-sift",
        "dataset_hash": dataset_hash,
        "dimension": 128,
        "counts": {
            "base": 100,
            "delta": 1000,
            "validation_queries": 4,
            "test_queries": 4,
        },
        "id_ranges": {
            "base": [0, 99],
            "delta": [100, 1099],
            "validation": [4, 5, 6, 7],
            "test": [0, 1, 2, 3],
        },
        "metadata": {"test_fixture": True},
    }
    split_id = object_sha256(split_unsigned)
    atomic_write_json(
        run_dir / "splits" / "real-delta.json",
        {**split_unsigned, "split_id": split_id},
    )
    method_names = {
        "F": "native_F",
        "A": "faiss_A",
        "N": "native_N",
        "P-beta0": "native_P_beta0",
    }
    normalized_rows = [
        {
            **row,
            "method": method_names.get(str(row.get("method")), row.get("method")),
            **({"beta_factor": 0.0} if row.get("method_role") == "P" else {}),
        }
        for row in rows
    ]
    present_queries = sorted(
        {(int(row["query_position"]), int(row["query_id"])) for row in rows}
    )
    for position, query_id in present_queries:
        influence = any(
            row.get("query_position") == position and row.get("delta_influence") is True
            for row in rows
        )
        normalized_rows.append(
            _row(
                "faiss_A_reference",
                "A-reference",
                query_id,
                11.0,
                influence=influence,
            )
        )
        oracle = _row(
            "old_O",
            "O",
            query_id,
            0.0,
            repetition=-1,
            influence=influence,
        )
        oracle.update(
            {
                "micro_latency_ns": None,
                "composed_e2e_latency_ns": None,
                "api_wall_latency_ns": None,
                "fraction_oracle_checked": position == 0,
                "fraction_oracle_order_match": True if position == 0 else None,
            }
        )
        normalized_rows.append(oracle)
    bound_rows = [
        {
            **row,
            "run_id": run_id,
            "config_hash": config_hash,
            "implementation_tree_sha256": implementation_hash,
            "dataset_hash": dataset_hash,
            "split_id": split_id,
            "phase": phase,
            "evidence_role": evidence_role,
        }
        for row in normalized_rows
    ]
    raw_path = run_dir / "raw" / "block-0000.jsonl"
    atomic_write_jsonl(raw_path, bound_rows)
    atomic_write_json(run_dir / "effective_config.json", effective)
    atomic_write_json(run_dir / "build_manifest.json", {"test_fixture": True})
    audit_path = run_dir / "audits" / "real-delta.native-lb.json"
    atomic_write_json(
        audit_path,
        {
            "experiment_id": "real-delta",
            "dataset_hash": dataset_hash,
            "split_id": split_id,
            "query_count": 0,
            "passed": True,
            "failures": [],
            "rows": [],
        },
    )
    checkpoint = {
        "run_id": run_id,
        "config_hash": config_hash,
        "implementation_tree_sha256": implementation_hash,
        "completed_blocks": {
            "block-0000": {
                "path": "raw/block-0000.jsonl",
                "sha256": file_sha256(raw_path),
                "rows": len(bound_rows),
            }
        },
    }
    manifest = {
        "run_id": run_id,
        "config_hash": config_hash,
        "implementation_tree_sha256": implementation_hash,
        "phase": phase,
        "evidence_role": evidence_role,
        "status": "completed",
        "completed_blocks": 1,
        "total_blocks": 1,
        "dataset_hashes": {"real-delta": dataset_hash},
        "split_ids": {"real-delta": split_id},
        "holdout_registration": {
            "path": str(holdout_path),
            "sha256": file_sha256(holdout_path),
            "phase": "validation",
            "final_policy_path": final_policy_identity["path"],
            "final_policy_file_sha256": final_policy_identity["file_sha256"],
            "final_policy_object_sha256": final_policy_identity["object_sha256"],
            "issue_fixed_decision_values": issue_fixed_decision_values(),
        },
    }
    atomic_write_json(run_dir / "checkpoint.json", checkpoint)
    atomic_write_json(run_dir / "run_manifest.json", manifest)
    completion = {
        "run_id": run_id,
        "config_hash": config_hash,
        "implementation_tree_sha256": implementation_hash,
        "checkpoint_sha256": file_sha256(run_dir / "checkpoint.json"),
        "run_manifest_sha256": file_sha256(run_dir / "run_manifest.json"),
        "raw_shards": 1,
        "holdout_manifest_sha256": file_sha256(holdout_path),
        "final_policy_file_sha256": final_policy_identity["file_sha256"],
        "final_policy_object_sha256": final_policy_identity["object_sha256"],
        "ancillary_files": [
            {
                "path": path.relative_to(run_dir).as_posix(),
                "sha256": file_sha256(path),
                "bytes": path.stat().st_size,
            }
            for path in sorted(
                (
                    run_dir / "effective_config.json",
                    run_dir / "build_manifest.json",
                    run_dir / "splits" / "real-delta.json",
                    audit_path,
                ),
                key=lambda value: value.relative_to(run_dir).as_posix(),
            )
        ],
    }
    atomic_write_json(run_dir / "COMPLETED.json", completion)
    return run_dir


def _rebind_holdout_registration(run_dir: Path, holdout_path: Path) -> None:
    """Re-sign the outer holdout binding so inner registration checks run."""

    manifest_path = run_dir / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["holdout_registration"]["sha256"] = file_sha256(holdout_path)
    atomic_write_json(manifest_path, manifest)
    completion_path = run_dir / "COMPLETED.json"
    completion = json.loads(completion_path.read_text(encoding="utf-8"))
    completion["holdout_manifest_sha256"] = file_sha256(holdout_path)
    completion["run_manifest_sha256"] = file_sha256(manifest_path)
    atomic_write_json(completion_path, completion)


def _rewrite_checkpoint_and_completion(
    run_dir: Path, checkpoint: dict[str, Any]
) -> None:
    checkpoint_path = run_dir / "checkpoint.json"
    completion_path = run_dir / "COMPLETED.json"
    atomic_write_json(checkpoint_path, checkpoint)
    completion = json.loads(completion_path.read_text(encoding="utf-8"))
    completion["checkpoint_sha256"] = file_sha256(checkpoint_path)
    atomic_write_json(completion_path, completion)


def _rewrite_raw_rows(run_dir: Path, rows: list[dict[str, Any]]) -> None:
    raw_path = run_dir / "raw" / "block-0000.jsonl"
    atomic_write_jsonl(raw_path, rows)
    checkpoint_path = run_dir / "checkpoint.json"
    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    entry = checkpoint["completed_blocks"]["block-0000"]
    entry["sha256"] = file_sha256(raw_path)
    entry["rows"] = len(rows)
    _rewrite_checkpoint_and_completion(run_dir, checkpoint)


def _rebind_ancillary(run_dir: Path, relative: str) -> None:
    completion_path = run_dir / "COMPLETED.json"
    completion = json.loads(completion_path.read_text(encoding="utf-8"))
    path = run_dir / relative
    entry = next(
        value
        for value in completion["ancillary_files"]
        if value["path"] == relative
    )
    entry["sha256"] = file_sha256(path)
    entry["bytes"] = path.stat().st_size
    atomic_write_json(completion_path, completion)


def test_paired_speedup_folds_repetitions_then_sessions_then_queries() -> None:
    rows: list[dict[str, Any]] = []
    values = {
        (0, "session-0", "F"): [8.0, 12.0],
        (0, "session-0", "P"): [4.0, 6.0],
        (0, "session-1", "F"): [18.0, 22.0],
        (0, "session-1", "P"): [4.0, 6.0],
        (1, "session-0", "F"): [3.0, 3.0],
        (1, "session-0", "P"): [3.0, 3.0],
        (1, "session-1", "F"): [27.0, 27.0],
        (1, "session-1", "P"): [3.0, 3.0],
    }
    for (query, session, role), latencies in values.items():
        method = "F" if role == "F" else "P-beta0"
        for repetition, latency in enumerate(latencies):
            rows.append(
                _row(
                    method,
                    role,
                    query,
                    latency,
                    session=session,
                    repetition=repetition,
                    influence=query == 0,
                )
            )

    summary = paired_speedup(
        rows,
        experiment_id="real-delta",
        proposed_method="P-beta0",
        baseline_method="F",
        baseline_role="F",
        metric="api_wall",
        bootstrap_resamples=MIN_BOOTSTRAP_RESAMPLES,
        bootstrap_seed=7,
    )
    repeated = paired_speedup(
        rows,
        experiment_id="real-delta",
        proposed_method="P-beta0",
        baseline_method="F",
        baseline_role="F",
        metric="api_wall",
        bootstrap_resamples=MIN_BOOTSTRAP_RESAMPLES,
        bootstrap_seed=7,
    )

    expected_query_zero = math.exp(
        math.fsum((math.log(2.0), math.log(4.0))) / 2
    )
    expected_query_one = math.exp(
        math.fsum((math.log(1.0), math.log(9.0))) / 2
    )
    assert summary["paired_session_query_observations"] == 4
    assert summary["paired_queries"] == 2
    assert summary["sessions_per_query"] == {"min": 2, "max": 2}
    expected_projection = [
        {
            "query_id": "0",
            "query_position": 0,
            "sessions": 2,
            "ratio_geometric_mean": expected_query_zero,
        },
        {
            "query_id": "1",
            "query_position": 1,
            "sessions": 2,
            "ratio_geometric_mean": expected_query_one,
        },
    ]
    assert "query_ratios" not in summary
    assert summary["query_ratios_object_sha256"] == object_sha256(
        expected_projection
    )
    assert summary["geometric_mean"] == pytest.approx(
        math.sqrt(expected_query_zero * expected_query_one)
    )
    assert summary["delta_influence_subset"]["paired_queries"] == 1
    assert summary["delta_influence_subset"]["geometric_mean"] == pytest.approx(
        expected_query_zero
    )
    assert summary["delta_influence_subset"][
        "query_ratios_object_sha256"
    ] == object_sha256(expected_projection[:1])
    assert summary["bootstrap_95pct_lower"] == repeated["bootstrap_95pct_lower"]
    assert summary["bootstrap_95pct_upper"] == repeated["bootstrap_95pct_upper"]


def test_paired_speedup_keeps_reused_session_ids_separate_by_run() -> None:
    rows: list[dict[str, Any]] = []
    for run_id, f_latency, p_latency in (
        ("process-run-0", 20.0, 10.0),
        ("process-run-1", 90.0, 30.0),
    ):
        for role, latency in (("F", f_latency), ("P", p_latency)):
            row = _row(role, role, 0, latency, session="session-0")
            row["run_id"] = run_id
            rows.append(row)

    summary = paired_speedup(
        rows,
        experiment_id="real-delta",
        proposed_method="P",
        baseline_method="F",
        baseline_role="F",
        metric="api_wall",
        bootstrap_resamples=MIN_BOOTSTRAP_RESAMPLES,
    )

    assert summary["paired_session_query_observations"] == 2
    assert summary["sessions_per_query"] == {"min": 2, "max": 2}
    assert summary["geometric_mean"] == pytest.approx(math.sqrt(6.0))


def test_paired_bootstrap_rejects_fewer_than_two_thousand_query_resamples() -> None:
    rows = [_row("F", "F", 0, 2.0), _row("P-beta0", "P", 0, 1.0)]

    with pytest.raises(NativeEvidenceError, match="at least 2000"):
        paired_speedup(
            rows,
            experiment_id="real-delta",
            proposed_method="P-beta0",
            baseline_method="F",
            baseline_role="F",
            metric="micro",
            bootstrap_resamples=1999,
        )


def test_A_quality_mismatch_rows_are_never_selected_out_of_timing() -> None:
    mismatch = _row("A", "A", 0, 10.0, influence=True)
    mismatch.update(
        {
            "comparison_valid": False,
            "baseline_validation_failure": True,
            "baseline_quality_match": False,
        }
    )
    rows = [
        mismatch,
        _row("P-beta0", "P", 0, 5.0, influence=True),
        _row("A", "A", 1, 20.0, influence=False),
        _row("P-beta0", "P", 1, 5.0, influence=False),
    ]

    summary = paired_speedup(
        rows,
        experiment_id="real-delta",
        proposed_method="P-beta0",
        baseline_method="A",
        baseline_role="A",
        metric="micro",
    )

    assert summary["paired_queries"] == 2
    assert summary["geometric_mean"] == pytest.approx(math.sqrt(2.0 * 4.0))
    assert summary["baseline_quality_mismatch_rows_retained"] == 1
    assert summary["baseline_quality_mismatch_queries_retained"] == 1


def test_delta_influence_label_tampering_across_methods_is_rejected() -> None:
    rows = [
        _row("F", "F", 0, 2.0, influence=True),
        _row("P-beta0", "P", 0, 1.0, influence=False),
    ]

    with pytest.raises(NativeEvidenceError, match="influence label changed"):
        analyze_native_rows(rows)


def test_analysis_reports_F_and_A_over_P_for_all_three_timing_scopes() -> None:
    summary = analyze_native_rows(_gate_rows())

    observed = {
        (comparison["baseline_role"], comparison["metric"])
        for comparison in summary["comparisons"]
    }
    assert observed == {
        (baseline, metric)
        for baseline in ("F", "A")
        for metric in ("micro", "composed_e2e", "api_wall")
    }
    assert all(
        comparison["delta_influence_subset"]["paired_queries"] == 1
        for comparison in summary["comparisons"]
    )


def test_validation_gate_passes_only_a_real_influential_native_candidate() -> None:
    summary = analyze_native_rows(_gate_rows())
    gate = evaluate_validation_gate(summary)

    assert gate["gate_status"] == "PASSED"
    assert gate["selected_candidate"]["experiment_id"] == "real-delta"
    assert gate["selected_candidate"]["requested_beta_l2"] == 0.0
    assert gate["lock_created"] is False


def test_validation_gate_rejects_a_same_c_quality_mismatch_without_dropping_timing() -> None:
    rows = _gate_rows()
    mismatched = next(row for row in rows if row["method_role"] == "A")
    mismatched.update(
        {
            "comparison_valid": False,
            "baseline_validation_failure": True,
            "baseline_quality_match": False,
        }
    )

    gate = evaluate_validation_gate(analyze_native_rows(rows))

    assert gate["gate_status"] == "NOT_PASSED"
    candidate = gate["candidates"][0]
    assert "A_same_c_quality_mismatch" in candidate["reasons"]
    outcome = candidate["comparison_outcomes"]["A"]
    assert outcome["timing_passed"] is True
    assert outcome["same_c_quality_passed"] is False
    assert outcome["comparisons"][0][
        "baseline_quality_mismatch_queries_retained"
    ] == 1


def test_gist_only_speedup_cannot_authorize_the_sift_fresh_holdout() -> None:
    rows = _gate_rows()
    for row in rows:
        row["dataset_name"] = "texmex-gist"

    gate = evaluate_validation_gate(analyze_native_rows(rows))

    assert gate["gate_status"] == "NOT_PASSED"
    assert "not_sift_primary_for_fresh_holdout" in gate["candidates"][0]["reasons"]


@pytest.mark.parametrize(
    "mutation, expected_reason",
    [
        (
            lambda rows: next(
                row for row in rows if row["method_role"] == "P"
            ).__setitem__("native_call_count", 0),
            "native_backend_or_call_gate_failed",
        ),
        (
            lambda rows: next(
                row for row in rows if row["method_role"] == "P"
            ).__setitem__("native_binary_sha256", "tampered"),
            "native_backend_or_call_gate_failed",
        ),
        (
            lambda rows: next(
                row for row in rows if row["method_role"] == "P"
            ).__setitem__("same_c_order_match", False),
            "correctness_gate_failed",
        ),
        (
            lambda rows: [row.__setitem__("n_delta", 999) for row in rows],
            "no_non_degenerate_candidate_beats_both_F_and_A",
        ),
        (
            lambda rows: [row.__setitem__("delta_influence", False) for row in rows],
            "no_non_degenerate_candidate_beats_both_F_and_A",
        ),
        (
            lambda rows: [
                row.__setitem__("api_wall_latency_ns", 10.0)
                for row in rows
                if row["method_role"] == "P"
            ],
            "no_non_degenerate_candidate_beats_both_F_and_A",
        ),
    ],
)
def test_validation_gate_fails_closed(
    mutation: Any, expected_reason: str
) -> None:
    rows = _gate_rows()
    mutation(rows)

    gate = evaluate_validation_gate(analyze_native_rows(rows))

    assert gate["gate_status"] == "NOT_PASSED"
    assert expected_reason in gate["reasons"]
    assert gate["selected_candidate"] is None
    assert gate["lock_created"] is False


def test_missing_proposed_delta_influence_row_label_fails_gate() -> None:
    rows = _gate_rows()
    del next(row for row in rows if row["method_role"] == "P")[
        "delta_influence"
    ]

    gate = evaluate_validation_gate(analyze_native_rows(rows))

    assert gate["gate_status"] == "NOT_PASSED"
    assert "incomplete_delta_influence_row_labels" in gate["candidates"][0]["reasons"]


def test_dropping_a_query_from_both_sides_cannot_improve_the_gate() -> None:
    rows = [row for row in _gate_rows() if row["query_id"] != 3]

    gate = evaluate_validation_gate(analyze_native_rows(rows))

    assert gate["gate_status"] == "NOT_PASSED"
    assert gate["candidates"][0]["comparison_outcomes"]["F"]["passed"] is False
    assert gate["candidates"][0]["comparison_outcomes"]["A"]["passed"] is False


def test_completed_run_loader_detects_raw_tampering(tmp_path: Path) -> None:
    run_dir = _write_completed_run(tmp_path, _gate_rows())
    loaded = load_native_evidence(run_dir)
    # The completed-run fixture adds required A-reference and O rows.
    assert len(loaded.rows) == len(_gate_rows()) + 8

    raw_path = run_dir / "raw" / "block-0000.jsonl"
    raw_path.write_text(raw_path.read_text(encoding="utf-8") + "{}\n", encoding="utf-8")

    with pytest.raises(NativeEvidenceError, match="raw checksum mismatch"):
        load_native_evidence(run_dir)


@pytest.mark.parametrize(
    "malicious_path",
    ("../outside.jsonl", "/tmp/outside.jsonl"),
)
def test_completed_run_loader_rejects_escaping_checkpoint_raw_path(
    tmp_path: Path, malicious_path: str
) -> None:
    run_dir = _write_completed_run(tmp_path, _gate_rows())
    checkpoint_path = run_dir / "checkpoint.json"
    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    checkpoint["completed_blocks"]["block-0000"]["path"] = malicious_path
    _rewrite_checkpoint_and_completion(run_dir, checkpoint)

    with pytest.raises(NativeEvidenceError, match="checkpoint raw path escapes"):
        load_native_evidence(run_dir)


def test_completed_run_loader_rejects_checkpoint_raw_symlink(tmp_path: Path) -> None:
    run_dir = _write_completed_run(tmp_path, _gate_rows())
    raw_link = run_dir / "raw" / "linked.jsonl"
    raw_link.symlink_to(run_dir / "raw" / "block-0000.jsonl")
    checkpoint_path = run_dir / "checkpoint.json"
    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    checkpoint["completed_blocks"]["block-0000"]["path"] = "raw/linked.jsonl"
    _rewrite_checkpoint_and_completion(run_dir, checkpoint)

    with pytest.raises(
        NativeEvidenceError, match="checkpoint raw path is not a regular in-run file"
    ):
        load_native_evidence(run_dir)


def test_completed_run_loader_rejects_checkpoint_raw_symlink_escape(
    tmp_path: Path,
) -> None:
    run_dir = _write_completed_run(tmp_path, _gate_rows())
    outside_raw = tmp_path / "outside.jsonl"
    atomic_write_jsonl(outside_raw, [{"outside": True}])
    raw_link = run_dir / "raw" / "outside-link.jsonl"
    raw_link.symlink_to(outside_raw)
    checkpoint_path = run_dir / "checkpoint.json"
    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    entry = checkpoint["completed_blocks"]["block-0000"]
    entry["path"] = "raw/outside-link.jsonl"
    entry["sha256"] = file_sha256(outside_raw)
    entry["rows"] = 1
    _rewrite_checkpoint_and_completion(run_dir, checkpoint)

    with pytest.raises(
        NativeEvidenceError, match="checkpoint raw path escapes the run directory"
    ):
        load_native_evidence(run_dir)


def test_completed_run_loader_rejects_duplicate_checkpoint_raw_path(
    tmp_path: Path,
) -> None:
    run_dir = _write_completed_run(tmp_path, _gate_rows())
    checkpoint_path = run_dir / "checkpoint.json"
    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    checkpoint["completed_blocks"]["block-0001"] = dict(
        checkpoint["completed_blocks"]["block-0000"]
    )
    _rewrite_checkpoint_and_completion(run_dir, checkpoint)

    manifest_path = run_dir / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["completed_blocks"] = 2
    manifest["total_blocks"] = 2
    atomic_write_json(manifest_path, manifest)
    completion_path = run_dir / "COMPLETED.json"
    completion = json.loads(completion_path.read_text(encoding="utf-8"))
    completion["run_manifest_sha256"] = file_sha256(manifest_path)
    completion["raw_shards"] = 2
    atomic_write_json(completion_path, completion)

    with pytest.raises(NativeEvidenceError, match="duplicate checkpoint raw path"):
        load_native_evidence(run_dir)


def test_completed_run_loader_rejects_unlisted_raw_shard(tmp_path: Path) -> None:
    run_dir = _write_completed_run(tmp_path, _gate_rows())
    atomic_write_jsonl(run_dir / "raw" / "unlisted.jsonl", [{"unlisted": True}])

    with pytest.raises(NativeEvidenceError, match="raw shard inventory coverage mismatch"):
        load_native_evidence(run_dir)


def test_completed_run_loader_detects_manifest_identity_tampering(
    tmp_path: Path,
) -> None:
    run_dir = _write_completed_run(tmp_path, _gate_rows())
    manifest_path = run_dir / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["config_hash"] = "0" * 64
    atomic_write_json(manifest_path, manifest)

    with pytest.raises(NativeEvidenceError, match="identity mismatch"):
        load_native_evidence(run_dir)


def test_completed_run_loader_detects_ancillary_file_tampering(
    tmp_path: Path,
) -> None:
    run_dir = _write_completed_run(tmp_path, _gate_rows())
    (run_dir / "build_manifest.json").write_text("{}\n", encoding="utf-8")

    with pytest.raises(NativeEvidenceError, match="ancillary .*mismatch"):
        load_native_evidence(run_dir)


def test_completed_run_loader_rejects_escaping_ancillary_path(
    tmp_path: Path,
) -> None:
    run_dir = _write_completed_run(tmp_path, _gate_rows())
    completion_path = run_dir / "COMPLETED.json"
    completion = json.loads(completion_path.read_text(encoding="utf-8"))
    completion["ancillary_files"][0]["path"] = "../outside.json"
    atomic_write_json(completion_path, completion)

    with pytest.raises(NativeEvidenceError, match="escapes"):
        load_native_evidence(run_dir)


def test_completed_run_loader_rejects_duplicate_ancillary_path(
    tmp_path: Path,
) -> None:
    run_dir = _write_completed_run(tmp_path, _gate_rows())
    completion_path = run_dir / "COMPLETED.json"
    completion = json.loads(completion_path.read_text(encoding="utf-8"))
    completion["ancillary_files"].append(dict(completion["ancillary_files"][0]))
    atomic_write_json(completion_path, completion)

    with pytest.raises(NativeEvidenceError, match="duplicate ancillary path"):
        load_native_evidence(run_dir)


def test_completed_run_loader_rejects_uninventoried_ancillary_file(
    tmp_path: Path,
) -> None:
    run_dir = _write_completed_run(tmp_path, _gate_rows())
    atomic_write_json(run_dir / "audits" / "extra.json", {"unexpected": True})

    with pytest.raises(NativeEvidenceError, match="inventory coverage mismatch"):
        load_native_evidence(run_dir)


def test_completed_run_loader_rejects_checksum_consistent_query_selection(
    tmp_path: Path,
) -> None:
    selected_rows = [row for row in _gate_rows() if row["query_id"] != 3]
    run_dir = _write_completed_run(tmp_path, selected_rows)

    with pytest.raises(NativeEvidenceError, match="incomplete raw query coverage"):
        load_native_evidence(run_dir)


def test_completed_run_loader_requires_full_configured_method_matrix(
    tmp_path: Path,
) -> None:
    run_dir = _write_completed_run(tmp_path, _gate_rows())
    raw_path = run_dir / "raw" / "block-0000.jsonl"
    rows = [json.loads(line) for line in raw_path.read_text().splitlines()]
    rows = [row for row in rows if row["method"] != "faiss_A_reference"]
    _rewrite_raw_rows(run_dir, rows)

    with pytest.raises(NativeEvidenceError, match="method matrix differs"):
        load_native_evidence(run_dir)


def test_completed_run_loader_requires_registered_fraction_oracle_prefix(
    tmp_path: Path,
) -> None:
    run_dir = _write_completed_run(tmp_path, _gate_rows())
    raw_path = run_dir / "raw" / "block-0000.jsonl"
    rows = [json.loads(line) for line in raw_path.read_text().splitlines()]
    oracle = next(
        row
        for row in rows
        if row["method"] == "old_O" and row["query_position"] == 0
    )
    oracle["fraction_oracle_checked"] = False
    oracle["fraction_oracle_order_match"] = None
    _rewrite_raw_rows(run_dir, rows)

    with pytest.raises(NativeEvidenceError, match="Fraction-oracle declaration"):
        load_native_evidence(run_dir)


def test_completed_run_loader_rejects_checksum_bound_failed_lb_audit(
    tmp_path: Path,
) -> None:
    run_dir = _write_completed_run(tmp_path, _gate_rows())
    audit_path = run_dir / "audits" / "real-delta.native-lb.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    audit["passed"] = False
    audit["failures"] = [{"counterexample": True}]
    atomic_write_json(audit_path, audit)
    _rebind_ancillary(run_dir, "audits/real-delta.native-lb.json")

    with pytest.raises(NativeEvidenceError, match="LB audit identity/query-count/pass"):
        load_native_evidence(run_dir)


def test_completed_run_loader_rejects_checksum_bound_scientific_failure(
    tmp_path: Path,
) -> None:
    run_dir = _write_completed_run(tmp_path, _gate_rows())
    manifest_path = run_dir / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["failures"] = [
        {"kind": "native_contract_failure", "query_id": 2}
    ]
    atomic_write_json(manifest_path, manifest)
    completion_path = run_dir / "COMPLETED.json"
    completion = json.loads(completion_path.read_text(encoding="utf-8"))
    completion["run_manifest_sha256"] = file_sha256(manifest_path)
    atomic_write_json(completion_path, completion)

    with pytest.raises(NativeEvidenceError, match="scientific failure"):
        load_native_evidence(run_dir)


def test_completed_validation_run_rejects_unregistered_source_config(
    tmp_path: Path,
) -> None:
    run_dir = _write_completed_run(tmp_path, _gate_rows())
    holdout_path = tmp_path / "holdout_manifest.json"
    holdout = json.loads(holdout_path.read_text(encoding="utf-8"))
    holdout["selection_integrity"][
        "validation_source_config_object_sha256"
    ] = "f" * 64
    atomic_write_json(holdout_path, holdout)
    _rebind_holdout_registration(run_dir, holdout_path)

    with pytest.raises(NativeEvidenceError, match="pre-registered hash"):
        load_native_evidence(run_dir)


def test_completed_validation_run_rejects_unregistered_source_config_file(
    tmp_path: Path,
) -> None:
    run_dir = _write_completed_run(tmp_path, _gate_rows())
    holdout_path = tmp_path / "holdout_manifest.json"
    holdout = json.loads(holdout_path.read_text(encoding="utf-8"))
    holdout["selection_integrity"][
        "validation_source_config_file_sha256"
    ] = "f" * 64
    atomic_write_json(holdout_path, holdout)
    _rebind_holdout_registration(run_dir, holdout_path)

    with pytest.raises(NativeEvidenceError, match="source config file"):
        load_native_evidence(run_dir)


def test_completed_validation_run_rejects_reserved_sift_interval(
    tmp_path: Path,
) -> None:
    run_dir = _write_completed_run(
        tmp_path, _gate_rows(), test_query_offset=1200
    )

    with pytest.raises(NativeEvidenceError, match="differ from pre-registration"):
        load_native_evidence(run_dir)


def test_formal_gate_accepts_completed_validation_manifests_only(
    tmp_path: Path,
) -> None:
    validation_root = tmp_path / "validation-only"
    _write_completed_run(validation_root, _gate_rows())
    validation_summary = analyze_native_path(validation_root)
    assert validation_summary["source_integrity"]["validation_gate_source_passed"]
    assert evaluate_validation_gate(validation_summary)["gate_status"] == "PASSED"

    mixed_root = tmp_path / "mixed"
    _write_completed_run(mixed_root, _gate_rows(), run_id="validation")
    _write_completed_run(
        mixed_root,
        _gate_rows(),
        run_id="smoke",
        phase="smoke",
        evidence_role="smoke",
    )
    mixed_summary = analyze_native_path(mixed_root)
    mixed_gate = evaluate_validation_gate(mixed_summary)
    assert mixed_summary["source_integrity"]["validation_gate_source_passed"] is False
    assert mixed_gate["gate_status"] == "NOT_PASSED"
    assert "validation_evidence_source_gate_failed" in mixed_gate["reasons"]

    duplicate_root = tmp_path / "duplicate-validation"
    _write_completed_run(duplicate_root, _gate_rows(), run_id="validation-1")
    _write_completed_run(duplicate_root, _gate_rows(), run_id="validation-2")
    duplicate_summary = analyze_native_path(duplicate_root)
    duplicate_gate = evaluate_validation_gate(duplicate_summary)
    assert duplicate_summary["source_integrity"]["completed_validation_runs"] == 2
    assert (
        duplicate_summary["source_integrity"]["validation_gate_source_passed"]
        is False
    )
    assert duplicate_gate["gate_status"] == "NOT_PASSED"
    assert "validation_evidence_source_gate_failed" in duplicate_gate["reasons"]


def test_unmanifested_rows_write_durable_NOT_PASSED_without_a_lock(
    tmp_path: Path,
) -> None:
    raw_path = tmp_path / "raw.jsonl"
    analysis_path = tmp_path / "analysis.json"
    gate_path = tmp_path / "validation_gate.json"
    atomic_write_jsonl(raw_path, _gate_rows())
    repository = Path(__file__).resolve().parents[1]
    environment = {
        **os.environ,
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
        "VECLIB_MAXIMUM_THREADS": "1",
    }

    completed = subprocess.run(
        [
            sys.executable,
            str(repository / "scripts" / "analyze_native_recheck.py"),
            "--input",
            str(raw_path),
            "--output",
            str(analysis_path),
            "--gate-output",
            str(gate_path),
        ],
        cwd=repository,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    assert gate["gate_status"] == "NOT_PASSED"
    assert gate["lock_created"] is False
    assert "validation_evidence_source_gate_failed" in gate["reasons"]
    assert not (tmp_path / "final.lock.json").exists()


def test_validation_config_encodes_the_required_one_axis_matrix() -> None:
    repository = Path(__file__).resolve().parents[1]
    config = json.loads(
        (repository / "configs" / "native_recheck_validation.json").read_text(
            encoding="utf-8"
        )
    )
    experiments = {row["id"]: row for row in config["experiments"]}
    real_ids = {
        "sift-initial",
        "gist-initial",
        "sift-delta-0",
        "sift-delta-1000",
        "sift-delta-100000",
        "sift-groups-64",
        "sift-groups-512",
        "sift-k-1",
        "sift-k-100",
    }
    synthetic_ids = {
        "synthetic-clustered",
        "synthetic-high-dimensional-isotropic",
        "synthetic-delta-near-queries",
        "synthetic-outlier-radius",
    }
    assert set(experiments) == real_ids | synthetic_ids
    assert config["phase"] == "validation"
    assert config["evidence_role"] == "validation"
    assert {method["symbol"] for method in config["methods"]} == {
        "O",
        "F",
        "A",
        "A-reference",
        "N",
        "P",
        "P-old",
    }
    assert {stage["id"] for stage in config["ablation_stages"]} == {
        "packed-layout",
        "incremental-top-k",
        "adaptive-exact",
        "pruning",
        "shared-base-preparation",
    }
    for experiment_id in real_ids:
        dataset = experiments[experiment_id]["dataset"]
        assert dataset["validation_query_offset"] == 0
        assert dataset["test_query_offset"] == 200
        assert dataset["n_validation"] == dataset["n_test"] == 200
        assert dataset["test_query_offset"] + dataset["n_test"] <= 1200
    assert experiments["sift-k-100"]["candidate_count"] == 100
    assert experiments["sift-initial"]["positive_beta_factors"] == [
        0.01,
        0.05,
        0.10,
    ]
    assert experiments["synthetic-high-dimensional-isotropic"]["dataset"][
        "dimension"
    ] == 128


def test_holdout_manifest_records_fresh_sift_and_non_independent_gist() -> None:
    repository = Path(__file__).resolve().parents[1]
    manifest = json.loads(
        (
            repository
            / "results"
            / "native_recheck_evidence"
            / "holdout_manifest.json"
        ).read_text(encoding="utf-8")
    )

    def ids_sha256(start: int, stop: int) -> str:
        payload = "".join(f"{value}\n" for value in range(start, stop)).encode()
        return hashlib.sha256(payload).hexdigest()

    assert manifest["audit_scope"]["all_existing_split_manifests_inspected"] is True
    assert manifest["audit_scope"]["manifest_count"] == 44
    assert manifest["sift"]["historically_used_query_ids"]["interval"] == [0, 1200]
    assert manifest["sift"]["historically_used_query_ids"]["set_sha256"] == ids_sha256(
        0, 1200
    )
    holdout = manifest["sift"]["pre_registered_fresh_final_holdout"]
    assert holdout["interval"] == [1200, 2200]
    assert holdout["set_sha256"] == ids_sha256(1200, 2200)
    assert holdout["loaded_by_native_validation"] is False
    assert holdout["disjoint_from_historically_used_union"] is True
    assert manifest["gist"]["historically_used_query_ids"]["interval"] == [0, 1000]
    assert manifest["gist"]["unused_query_rows"] == 0
    assert manifest["gist"]["independence_label"] == "reused_non_independent"
    assert manifest["gist"]["fresh_holdout_claim_permitted"] is False
    validation_config_path = repository / "configs" / "native_recheck_validation.json"
    assert manifest["selection_integrity"][
        "validation_source_config_file_sha256"
    ] == file_sha256(validation_config_path)
    assert manifest["selection_integrity"][
        "validation_source_config_object_sha256"
    ] == object_sha256(
        json.loads(validation_config_path.read_text(encoding="utf-8"))
    )
    final_policy_path = repository / "configs" / "native_recheck_final_policy.json"
    final_policy = json.loads(final_policy_path.read_text(encoding="utf-8"))
    registration = manifest["pre_registered_final_policy"]
    assert registration["path"] == "configs/native_recheck_final_policy.json"
    assert registration["file_sha256"] == file_sha256(final_policy_path)
    assert registration["object_sha256"] == object_sha256(final_policy)
    assert registration["issue_fixed_decision_values"] == (
        issue_fixed_decision_values()
    )
    assert manifest["selection_integrity"]["final_policy_file_sha256"] == (
        registration["file_sha256"]
    )
    configured_registration = json.loads(
        validation_config_path.read_text(encoding="utf-8")
    )["holdout_policy"]["pre_registered_final_policy"]
    assert configured_registration == {
        key: registration[key] for key in ("path", "file_sha256", "object_sha256")
    }

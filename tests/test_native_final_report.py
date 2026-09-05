from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from scripts.decide_native_final import (
    _reject_negative_final_artifacts,
    _verify_registered_hnsw_outputs,
)
from scripts.report_native_recheck import (
    _ablation_table,
    _completed_artifacts,
    _delta_influence_tables,
    _fallback_counts,
    _final_evidence,
    _geometry_table,
    _main_build_table,
    _main_component_table,
    _main_latency_scope_table,
    _main_positive_beta_table,
    _rq_result_answers,
    _verify_final_decision_correctness,
)
from searchability.artifacts import atomic_write_json, file_sha256


REPOSITORY = Path(__file__).resolve().parents[1]
THREAD_VARIABLES = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)


def _write(path: Path, value: object) -> None:
    atomic_write_json(path, value)


def _validation_files(tmp_path: Path, status: str) -> tuple[Path, Path]:
    gate_path = tmp_path / "validation_gate.json"
    summary_path = tmp_path / "validation_summary.json"
    gate = {
        "study_id": "issue-3-native-recheck",
        "gate_status": status,
    }
    _write(gate_path, gate)
    _write(
        summary_path,
        {
            "study_id": "issue-3-native-recheck",
            "validation_gate": gate,
        },
    )
    return summary_path, gate_path


def _negative_decision(
    validation_summary: Path, validation_gate: Path
) -> dict[str, object]:
    return {
        "study_id": "issue-3-native-recheck",
        "gate_sha256": file_sha256(validation_gate),
        "validation_summary_sha256": file_sha256(validation_summary),
        "validation_gate_status": "NOT_PASSED",
        "final_status": "NOT_RUN_GATE_NOT_PASSED",
        "performance_gate_status": "NOT_RUN_GATE_NOT_PASSED",
        "performance_verdict": "NOT_SUPPORTED_IN_TESTED_REGIME",
        "verdict": "NOT_SUPPORTED_IN_TESTED_REGIME",
        "engineering_decision": "NO_GO",
        "lock_created": False,
        "large_final_started": False,
        "fresh_sift_holdout_loaded": False,
    }


def _paths(tmp_path: Path) -> dict[str, Path]:
    return {
        "decision_path": tmp_path / "decision.json",
        "summary_path": tmp_path / "final_summary.json",
        "gate_path": tmp_path / "final_gate.json",
        "lock_path": tmp_path / "final_lock.json",
        "config_path": tmp_path / "final_config.json",
        "pre_hnsw_authorization_path": tmp_path / "pre_hnsw_authorization.json",
        "hnsw_path": tmp_path / "final_hnsw",
    }


def test_rq_answers_report_counts_and_descriptive_mechanism_metrics() -> None:
    comparison = lambda role, value, passed: {
        "passed": passed,
        "comparisons": [
            {
                "geometric_mean": value,
                "bootstrap_95pct_lower": value - 0.05,
                "bootstrap_95pct_upper": value + 0.05,
                "baseline_quality_mismatch_rows_retained": 2 if role == "A" else 0,
                "baseline_quality_mismatch_queries_retained": 1 if role == "A" else 0,
            }
        ],
    }
    gate = {
        "gate_status": "NOT_PASSED",
        "candidates": [
            {
                "real_non_degenerate": True,
                "sift_primary_for_fresh_holdout": True,
                "delta_influence_queries": 1,
                "experiment_id": "sift-initial",
                "proposed_method": "native_P_beta0",
                "comparison_outcomes": {
                    "F": comparison("F", 1.2, True),
                    "A": comparison("A", 0.9, False),
                },
            },
            {
                "real_non_degenerate": True,
                "sift_primary_for_fresh_holdout": True,
                "delta_influence_queries": 1,
                "experiment_id": "sift-k-1",
                "proposed_method": "native_P_beta0",
                "comparison_outcomes": {
                    "F": comparison("F", 1.3, True),
                    "A": comparison("A", 0.95, False),
                },
            },
            {
                "real_non_degenerate": True,
                "sift_primary_for_fresh_holdout": False,
                "delta_influence_queries": 1,
                "experiment_id": "gist-initial",
                "proposed_method": "native_P_beta0",
                "comparison_outcomes": {
                    "F": comparison("F", 0.9, False),
                    "A": comparison("A", 0.8, False),
                },
            },
        ],
    }
    summary = {
        "methods_by_experiment": {
            "sift-initial": {
                "F": ["native_F"],
                "N": ["native_N"],
                "P": ["native_P_beta0"],
            }
        }
    }
    base = {
        "run_id": "run",
        "session_id": "session",
        "query_id": 1200,
        "query_position": 0,
        "experiment_id": "sift-initial",
        "method_role": "P",
        "receipt": {"groups_skipped": 3, "groups_scanned": 1},
    }
    rows = [
        {
            **base,
            "method": "native_F",
            "method_role": "F",
            "api_wall_latency_ns": 120.0,
        },
        {
            **base,
            "method": "native_N",
            "method_role": "N",
            "api_wall_latency_ns": 110.0,
        },
        {**base, "method": "native_P_beta0", "api_wall_latency_ns": 100.0},
    ]
    builds = {
        "sift-initial": [
            {
                "groups": {
                    "center_training_ns": 100,
                    "assignment_ns": 200,
                    "packing_and_radius_ns": 300,
                    "member_bytes": 10,
                    "metadata_bytes": 20,
                }
            }
        ]
    }
    audits = {
        "sift-initial": [
            {
                "rows": [
                    {
                        "action": "scan",
                        "exact_min_l2_lower": 2.0,
                        "lb_lower": 0.5,
                    }
                ]
            }
        ]
    }
    rq1, rq2, rq3 = _rq_result_answers(
        gate=gate,
        summary=summary,
        rows=rows,
        builds=builds,
        audits=audits,
    )
    assert "事前指定 main `sift-initial` の F/P geomean 範囲は 1.200–1.200" in rq1
    assert "gate-eligible SIFT 候補 2 件中、F/P criterion 通過は 2 件" in rq1
    assert "記述対象は計 3 件（GIST anchor 1 件を含み、その F/P 通過は 0/1）" in rq1
    assert "secondary sweep 上の事後的最大 F/P=1.300" in rq1
    assert "事前指定 main family の確認的要約とは区別する" in rq1
    assert "事前登録済みのgate candidate 集合には含まれる" in rq1
    assert "validation gate は `NOT_PASSED`" in rq1
    assert "`selected_candidate` は `null`" in rq1
    assert "事前指定 main `sift-initial` の A/P geomean 範囲は 0.900–0.900" in rq2
    assert "A/P timing+quality criterion 通過は 0 件" in rq2
    assert "GIST anchor のA/P 通過は 0/1" in rq2
    assert "secondary sweep 上の事後的最大 A/P=0.950" in rq2
    assert "4 rows / 2 queries" in rq2
    assert "N/P geomean 範囲=1.100–1.100" in rq3
    assert "median skipped groups=3.0–3.0/4" in rq3
    assert "scan gap median/p95=1.500/1.500 L2" in rq3
    assert "group-only の F 比有限 break-even は 1/1 operating points（30.0–30.0 queries）" in rq3
    assert "L2 gap を pooled aggregate しない" in rq3


def _subset_candidate(
    experiment: str, method: str, *, sift: bool, influenced: int
) -> dict[str, object]:
    outcomes: dict[str, object] = {}
    for role, value in (("F", 1.2), ("A", 0.8)):
        outcomes[role] = {
            "passed": role == "F",
            "comparisons": [
                {
                    "geometric_mean": value,
                    "bootstrap_95pct_lower": value - 0.1,
                    "bootstrap_95pct_upper": value + 0.1,
                    "delta_influence_subset": {
                        "geometric_mean": value + 0.01,
                        "bootstrap_95pct_lower": value - 0.02,
                        "bootstrap_95pct_upper": value + 0.03,
                        "paired_queries": influenced,
                    },
                }
            ],
        }
    return {
        "real_non_degenerate": True,
        "sift_primary_for_fresh_holdout": sift,
        "delta_influence_queries": influenced,
        "experiment_id": experiment,
        "proposed_method": method,
        "requested_beta_l2": 2.5,
        "comparison_outcomes": outcomes,
    }


def test_delta_influence_tables_include_every_sift_candidate_and_separate_gist() -> None:
    gate = {
        "candidates": [
            _subset_candidate("sift-initial", "p0", sift=True, influenced=7),
            _subset_candidate("sift-k-1", "p1", sift=True, influenced=5),
            _subset_candidate("gist-initial", "pg", sift=False, influenced=3),
            {
                **_subset_candidate("synthetic", "ps", sift=False, influenced=2),
                "real_non_degenerate": False,
            },
        ]
    }
    sift, gist, counts = _delta_influence_tables(gate)
    assert counts == {
        "sift": 2,
        "gist": 1,
        "sift_f_ci_lower_above_one": 2,
        "sift_a_ci_lower_above_one": 0,
        "sift_a_ci_upper_below_one": 2,
    }
    assert len(sift) == 2 and "| 7 |" in sift[0] and "| 5 |" in sift[1]
    assert len(gist) == 1 and "gist-initial" in gist[0] and "| 3 |" in gist[0]

    broken = json.loads(json.dumps(gate))
    del broken["candidates"][0]["comparison_outcomes"]["F"]["comparisons"][0][
        "delta_influence_subset"
    ]
    with pytest.raises(RuntimeError, match="subset is missing"):
        _delta_influence_tables(broken)


def _main_report_fixture() -> tuple[dict[str, object], list[dict[str, object]]]:
    methods = ("native_F", "native_N", "faiss_A", "native_P_beta0", "native_P_beta1")
    rows: list[dict[str, object]] = []
    timing = {
        "native_F": (1_000.0, 1_200.0, 1_400.0),
        "native_N": (950.0, 1_150.0, 1_350.0),
        "faiss_A": (900.0, 1_100.0, 1_300.0),
        "native_P_beta0": (800.0, 1_000.0, 1_200.0),
        "native_P_beta1": (700.0, 900.0, 1_050.0),
    }
    for position in (0, 1):
        for method in methods:
            micro, composed, api = timing[method]
            if method == "faiss_A":
                components = {
                    "delta_faiss_search_ns": 200.0,
                    "shortlist_exact_merge_ns": 400.0,
                }
            else:
                components = {
                    "kernel_total_ns": 300.0,
                    "lb_calculation_ns": 20.0,
                    "group_ordering_ns": 10.0,
                    "group_scan_ns": 200.0,
                    "raw_scan_ns": 30.0,
                    "adaptive_exact_ns": 200.0,
                    "receipt_ns": 100.0,
                }
            role = {
                "native_F": "F",
                "native_N": "N",
                "faiss_A": "A",
                "native_P_beta0": "P",
                "native_P_beta1": "P",
            }[method]
            rows.append(
                {
                    "run_id": "run",
                    "session_id": "session",
                    "query_id": position,
                    "query_position": position,
                    "experiment_id": "sift-initial",
                    "method": method,
                    "method_role": role,
                    "repetition": 0,
                    "requested_beta_l2": (
                        0.0 if method == "native_P_beta0" else 2.5
                    ) if role == "P" else 0.0,
                    "observed_beta_upper_l2": 0.0 if role == "P" else None,
                    "certified_beta_l2": (
                        0.0 if method == "native_P_beta0" else 2.25
                    ) if role == "P" else None,
                    "micro_latency_ns": micro + position,
                    "composed_e2e_latency_ns": composed + position,
                    "api_wall_latency_ns": api + position,
                    "base_prepare_wall_ns": 100.0 + position,
                    "native_query_prepare_wall_ns": (
                        0.0 if method == "faiss_A" else 50.0 + position
                    ),
                    "component_timings_ns": components,
                    "receipt": {
                        "groups_skipped": (
                            4 if method == "native_P_beta1" else 2
                        ),
                        "vectors_scanned": (
                            60 if method == "native_P_beta1" else 80
                        ),
                    },
                    "python_fallback_used": False,
                }
            )
    comparisons: list[dict[str, object]] = []
    for proposed in ("native_P_beta0", "native_P_beta1"):
        for metric in ("micro", "composed_e2e", "api_wall"):
            for role, value in (("F", 1.2), ("A", 0.9)):
                comparisons.append(
                    {
                        "experiment_id": "sift-initial",
                        "proposed_method": proposed,
                        "baseline_role": role,
                        "metric": metric,
                        "geometric_mean": value,
                        "bootstrap_95pct_lower": value - 0.05,
                        "bootstrap_95pct_upper": value + 0.05,
                        "paired_queries": 2,
                        "paired_session_query_observations": 2,
                        "sessions_per_query": {"min": 1, "max": 1},
                    }
                )
    summary: dict[str, object] = {
        "methods_by_experiment": {
            "sift-initial": {
                "F": ["native_F"],
                "N": ["native_N"],
                "A": ["faiss_A"],
                "P": ["native_P_beta0", "native_P_beta1"],
            }
        },
        "comparisons": comparisons,
    }
    return summary, rows


def test_main_scope_components_positive_beta_and_fallback_have_denominators() -> None:
    summary, rows = _main_report_fixture()
    latency, coverage = _main_latency_scope_table(summary, rows)
    assert [line.split(" | ")[0] for line in latency] == [
        "| micro",
        "| composed_e2e",
        "| api_wall",
    ]
    assert coverage == {
        "paired_queries": 2,
        "paired_session_query_observations": 2,
        "process_sessions": 1,
        "sessions_per_query_min": 1,
        "sessions_per_query_max": 1,
    }
    assert all(line.endswith("| 2/2 |") for line in latency)

    components = _main_component_table(summary, rows)
    assert len(components) == 5
    assert any("| A | faiss_A |" in line and "| 2 |" in line for line in components)
    assert any("native_P_beta1" in line and "2.500000" in line for line in components)

    beta_rows, interpretation = _main_positive_beta_table(summary, rows)
    assert len(beta_rows) == 2
    assert "native_P_beta1" in beta_rows[1] and "-12.49" in beta_rows[1]
    assert "pruning" in interpretation and "session-query n=2" in interpretation
    assert "nonzero observed gap は 0/2 rows" in interpretation
    assert "positive-beta raw rows（repetitionを含む）n=2" in interpretation
    assert "0.000000/2.250000/2.500000 L2" in interpretation
    assert "trade-off で得たものではない" in interpretation

    cross_scale_rows = json.loads(json.dumps(rows))
    cross_scale = next(
        row for row in cross_scale_rows if row["method"] == "native_P_beta1"
    ).copy()
    cross_scale.update(
        {
            "experiment_id": "gist-initial",
            "requested_beta_l2": 100.0,
            "certified_beta_l2": 90.0,
            "observed_beta_upper_l2": 0.0,
        }
    )
    cross_scale_rows.append(cross_scale)
    _, cross_scale_interpretation = _main_positive_beta_table(
        summary, cross_scale_rows
    )
    assert "nonzero observed gap は 0/3 rows" in cross_scale_interpretation
    assert "0.000000/2.250000/2.500000 L2" in cross_scale_interpretation
    assert "90.000000/100.000000" not in cross_scale_interpretation

    assert _fallback_counts(rows) == {
        "fnp_rows": 8,
        "fnp_fallbacks": 0,
        "p_rows": 4,
        "p_fallbacks": 0,
    }

    broken = json.loads(json.dumps(summary))
    broken["comparisons"] = [
        comparison
        for comparison in broken["comparisons"]
        if not (
            comparison["metric"] == "micro"
            and comparison["baseline_role"] == "F"
            and comparison["proposed_method"] == "native_P_beta0"
        )
    ]
    with pytest.raises(RuntimeError, match="comparison is missing"):
        _main_latency_scope_table(broken, rows)

    invalid_chain = json.loads(json.dumps(rows))
    next(
        row
        for row in invalid_chain
        if row["method"] == "native_P_beta1"
    )["observed_beta_upper_l2"] = 2.4
    with pytest.raises(RuntimeError, match="chain is invalid"):
        _main_positive_beta_table(summary, invalid_chain)


def test_geometry_and_build_tables_validate_counts_and_report_cold_warm_memory() -> None:
    audits = {
        "sift-initial": [
            {
                "scope": "validation_queries_only_outside_timing",
                "query_count": 1,
                "groups_checked": 2,
                "scanned_groups_checked": 1,
                "skipped_groups_checked": 1,
                "rows": [
                    {
                        "query_id": 0,
                        "query_position": 0,
                        "action": "scan",
                        "radius_upper": 3.0,
                        "exact_min_l2_lower": 4.0,
                        "lb_lower": 1.0,
                    },
                    {
                        "query_id": 0,
                        "query_position": 0,
                        "action": "skip",
                        "radius_upper": 2.0,
                        "exact_min_l2_lower": 5.0,
                        "lb_lower": 4.0,
                    },
                ],
            }
        ]
    }
    geometry, coverage = _geometry_table(audits)
    assert len(geometry) == 2 and all("| 1 |" in line for line in geometry)
    assert coverage["audited_query_sets"] == 1
    assert coverage["decisions"] == 2
    broken_audits = json.loads(json.dumps(audits))
    broken_audits["sift-initial"][0]["groups_checked"] = 3
    with pytest.raises(RuntimeError, match="count/action mismatch"):
        _geometry_table(broken_audits)

    builds = {
        "sift-initial": [
            {
                "groups": {
                    "center_training_ns": 100,
                    "assignment_ns": 200,
                    "packing_and_radius_ns": 300,
                },
                "native_packed_view": {"build_ns": 400, "warm_cache_lookup_ns": 5},
                "memory": {
                    "group_member_bytes": 10,
                    "group_metadata_bytes_estimate": 2,
                    "packed_python_owned_bytes": 20,
                    "packed_native_owned_bytes": 30,
                    "rss_change_bytes": 1000,
                },
            }
        ]
    }
    cache = _cache_audits()
    cache["sift-initial"][0]["cache_stats"] = {
        "cache_build_ns": 50,
        "cache_lookup_ns": 7,
        "hits": 3,
        "misses": 1,
    }
    table = _main_build_table(builds, cache)
    rendered = "\n".join(table)
    assert len(table) == 12
    assert "warm cache identity lookup" in rendered
    assert "62 bytes" in rendered
    assert "not P-only attribution" in rendered


def test_report_rejects_final_decision_bound_to_stale_correctness(tmp_path: Path) -> None:
    correctness = tmp_path / "correctness.json"
    _write(correctness, {"status": "passed"})
    decision = {"correctness_sha256": file_sha256(correctness)}
    _verify_final_decision_correctness(decision, correctness)
    _write(correctness, {"status": "failed"})
    with pytest.raises(RuntimeError, match="not bound to current correctness"):
        _verify_final_decision_correctness(decision, correctness)


def test_negative_report_accepts_absent_final_artifacts_and_rejects_wrong_verdict(
    tmp_path: Path,
) -> None:
    validation_summary, validation_gate = _validation_files(tmp_path, "NOT_PASSED")
    paths = _paths(tmp_path)
    decision = _negative_decision(validation_summary, validation_gate)
    _write(paths["decision_path"], decision)

    loaded, section = _final_evidence(
        **paths,
        validation_summary_path=validation_summary,
        validation_gate_path=validation_gate,
    )
    assert loaded["verdict"] == "NOT_SUPPORTED_IN_TESTED_REGIME"
    assert "作成せず" in section
    assert "fresh-final holdout estimate ではない" in section
    assert "| final_status | `NOT_RUN_GATE_NOT_PASSED` |" in section
    assert "| selected_candidate | `null` |" in section

    decision["verdict"] = "INCONCLUSIVE"
    _write(paths["decision_path"], decision)
    with pytest.raises(RuntimeError, match="no-lock/no-load"):
        _final_evidence(
            **paths,
            validation_summary_path=validation_summary,
            validation_gate_path=validation_gate,
        )
    decision["verdict"] = "NOT_SUPPORTED_IN_TESTED_REGIME"
    _write(paths["decision_path"], decision)
    _write(paths["pre_hnsw_authorization_path"], {"stale": True})
    with pytest.raises(RuntimeError, match="no-lock/no-load"):
        _final_evidence(
            **paths,
            validation_summary_path=validation_summary,
            validation_gate_path=validation_gate,
        )
    paths["pre_hnsw_authorization_path"].unlink()
    paths["hnsw_path"].mkdir()
    with pytest.raises(RuntimeError, match="no-lock/no-load"):
        _final_evidence(
            **paths,
            validation_summary_path=validation_summary,
            validation_gate_path=validation_gate,
        )


def test_authorized_report_is_strictly_inconclusive(tmp_path: Path) -> None:
    validation_summary, validation_gate = _validation_files(tmp_path, "PASSED")
    paths = _paths(tmp_path)
    _write(paths["lock_path"], {"status": "LOCKED_FOR_FRESH_FINAL"})
    _write(paths["config_path"], {"study_id": "issue-3-native-recheck"})
    decision = {
        "study_id": "issue-3-native-recheck",
        "gate_sha256": file_sha256(validation_gate),
        "validation_summary_sha256": file_sha256(validation_summary),
        "validation_gate_status": "PASSED",
        "final_status": "AUTHORIZED_NOT_YET_RUN",
        "performance_gate_status": "NOT_RUN",
        "performance_verdict": "INCONCLUSIVE",
        "verdict": "INCONCLUSIVE",
        "engineering_decision": "NO_GO",
        "lock_created": True,
        "large_final_started": False,
        "fresh_sift_holdout_loaded": False,
        "final_lock_sha256": file_sha256(paths["lock_path"]),
        "final_config_file_sha256": file_sha256(paths["config_path"]),
    }
    _write(paths["decision_path"], decision)

    loaded, section = _final_evidence(
        **paths,
        validation_summary_path=validation_summary,
        validation_gate_path=validation_gate,
    )
    assert loaded["verdict"] == "INCONCLUSIVE"
    assert "未完了" in section

    decision["verdict"] = "SUPPORTED_IN_TESTED_REGIME"
    _write(paths["decision_path"], decision)
    with pytest.raises(RuntimeError, match="pre-final"):
        _final_evidence(
            **paths,
            validation_summary_path=validation_summary,
            validation_gate_path=validation_gate,
        )


def _write_completed_fixture(
    tmp_path: Path,
    *,
    engineering_decision: str,
) -> tuple[dict[str, Path], Path, Path, dict[str, object]]:
    validation_summary, validation_gate = _validation_files(tmp_path, "PASSED")
    paths = _paths(tmp_path)
    _write(paths["lock_path"], {"status": "LOCKED_FOR_FRESH_FINAL"})
    _write(paths["config_path"], {"study_id": "issue-3-native-recheck"})
    lock_sha = file_sha256(paths["lock_path"])
    config_sha = file_sha256(paths["config_path"])
    config_object_sha = "c" * 64
    hnsw_sha = "d" * 64
    _write(paths["pre_hnsw_authorization_path"], {"status": "authorized"})
    pre_hnsw_sha = file_sha256(paths["pre_hnsw_authorization_path"])
    paths["hnsw_path"].mkdir()
    outcomes: dict[str, dict[str, object]] = {}
    for role in ("F", "A"):
        engineering_passed = engineering_decision == "GO" or role == "A"
        outcomes[role] = {
            "passed": True,
            "performance_gate_passed": True,
            "engineering_GO_passed": engineering_passed,
            "complete_query_session_pairs": True,
            "bootstrap_CI_supports_speedup": True,
            "bootstrap_95pct_lower": 1.01,
            "bootstrap_95pct_lower_must_be_strictly_above": 1.0,
            "geometric_mean_speedup": 1.08 if role == "F" else 1.12,
            "minimum_required_speedup": 1.10,
            "P_query_folded_p95_ns": 100.0,
            "baseline_query_folded_p95_ns": 110.0,
            "P_over_baseline_p95_ratio": 100.0 / 110.0,
            "maximum_allowed_p95_ratio": 1.05,
            "A_quality_match": True if role == "A" else None,
        }
    final_gate = {
        "study_id": "issue-3-native-recheck",
        "gate_status": "PASSED",
        "performance_gate_status": "PASSED",
        "performance_verdict": "SUPPORTED_IN_TESTED_REGIME",
        "verdict": "SUPPORTED_IN_TESTED_REGIME",
        "engineering_decision": engineering_decision,
        "engineering_reasons": (
            []
            if engineering_decision == "GO"
            else ["speedup_or_p95_requirement_failed_vs_F"]
        ),
        "passed": True,
        "fresh_query_interval": [1200, 2200],
        "process_session_seeds": [0, 1, 2],
        "comparison_outcomes": outcomes,
        "fresh_delta_influence": {
            "labelled_queries": 1000,
            "influenced_queries": 1,
            "required_minimum": 1,
            "evidence_complete": True,
        },
        "build_cost_evidence_complete": True,
        "gist_used_in_gate": False,
        "hnsw_references_used_in_gate": False,
        "hnsw_reference_status": "completed",
        "final_lock_sha256": lock_sha,
        "final_config_file_sha256": config_sha,
        "hnsw_completion_sha256": hnsw_sha,
        "pre_hnsw_authorization_sha256": pre_hnsw_sha,
    }
    final_summary = {
        "study_id": "issue-3-native-recheck",
        "evidence_role": "final",
        "fresh_primary": {
            "experiment_id": "sift-final-primary",
            "proposed_method": "native_P_beta0",
            "query_interval": [1200, 2200],
            "independence_label": "fresh_pre_registered_holdout",
        },
        "sessions": [{"session_seed": value} for value in (0, 1, 2)],
        "primary_session_metrics": [
            {
                "session_seed": seed,
                "session_id": f"session-seed-{seed}:process-0000",
                "query_count": 1000,
                "methods": {
                    "F": "native_F",
                    "A": "faiss_A",
                    "P": "native_P_beta0",
                },
                "F": {
                    "api_wall_p50_ns": 120_000 + seed,
                    "api_wall_p95_ns": 160_000 + seed,
                },
                "A": {
                    "api_wall_p50_ns": 110_000 + seed,
                    "api_wall_p95_ns": 150_000 + seed,
                },
                "P": {
                    "api_wall_p50_ns": 100_000 + seed,
                    "api_wall_p95_ns": 140_000 + seed,
                },
                "F_over_P_geomean": 1.20 + seed / 100,
                "A_over_P_geomean": 1.10 + seed / 100,
            }
            for seed in (0, 1, 2)
        ],
        "gist_interpretation": {
            "independence_label": "reused_non_independent",
            "gate_eligible": False,
        },
        "hnsw_references": {
            "status": "completed",
            "role": "approximate_reference_only_not_in_final_gate",
            "completion_sha256": hnsw_sha,
            "pre_hnsw_authorization_sha256": pre_hnsw_sha,
        },
        "final_gate": final_gate,
        "final_lock_sha256": lock_sha,
        "final_config_file_sha256": config_sha,
        "final_config_object_sha256": config_object_sha,
        "validation_gate_sha256": file_sha256(validation_gate),
        "validation_summary_sha256": file_sha256(validation_summary),
    }
    _write(paths["gate_path"], final_gate)
    _write(paths["summary_path"], final_summary)
    decision = {
        "study_id": "issue-3-native-recheck",
        "gate_sha256": file_sha256(validation_gate),
        "validation_summary_sha256": file_sha256(validation_summary),
        "validation_gate_status": "PASSED",
        "final_status": "COMPLETED_PASSED",
        "performance_gate_status": "PASSED",
        "performance_verdict": "SUPPORTED_IN_TESTED_REGIME",
        "verdict": "SUPPORTED_IN_TESTED_REGIME",
        "engineering_decision": engineering_decision,
        "hnsw_reference_status": "completed",
        "pre_hnsw_authorization_sha256": pre_hnsw_sha,
        "lock_created": True,
        "large_final_started": True,
        "fresh_sift_holdout_loaded": True,
        "final_lock_sha256": lock_sha,
        "final_config_file_sha256": config_sha,
        "final_config_object_sha256": config_object_sha,
        "hnsw_completion_sha256": hnsw_sha,
        "final_gate_sha256": file_sha256(paths["gate_path"]),
        "final_summary_sha256": file_sha256(paths["summary_path"]),
    }
    _write(paths["decision_path"], decision)
    return paths, validation_summary, validation_gate, decision


def test_completed_report_accepts_supported_no_go_and_is_hash_bound(
    tmp_path: Path,
) -> None:
    paths, validation_summary, validation_gate, decision = _write_completed_fixture(
        tmp_path,
        engineering_decision="NO_GO",
    )
    loaded, section = _final_evidence(
        **paths,
        validation_summary_path=validation_summary,
        validation_gate_path=validation_gate,
    )
    assert loaded["verdict"] == "SUPPORTED_IN_TESTED_REGIME"
    assert loaded["engineering_decision"] == "NO_GO"
    assert "両立し得る" in section
    assert "session 間 fold 前" in section
    assert "| 0 |" in section and "| 2 |" in section

    decision["final_status"] = "COMPLETED_NOT_PASSED"
    _write(paths["decision_path"], decision)
    with pytest.raises(RuntimeError, match="identity mismatch"):
        _final_evidence(
            **paths,
            validation_summary_path=validation_summary,
            validation_gate_path=validation_gate,
        )

    decision["final_status"] = "COMPLETED_PASSED"
    decision["final_gate_sha256"] = "0" * 64
    _write(paths["decision_path"], decision)
    with pytest.raises(RuntimeError, match="identity mismatch"):
        _final_evidence(
            **paths,
            validation_summary_path=validation_summary,
            validation_gate_path=validation_gate,
        )


def test_completed_report_rejects_tampered_per_seed_display(tmp_path: Path) -> None:
    paths, validation_summary, validation_gate, decision = _write_completed_fixture(
        tmp_path,
        engineering_decision="NO_GO",
    )
    final_summary = json.loads(paths["summary_path"].read_text(encoding="utf-8"))
    final_summary["primary_session_metrics"][2]["session_seed"] = 1
    _write(paths["summary_path"], final_summary)
    decision["final_summary_sha256"] = file_sha256(paths["summary_path"])
    _write(paths["decision_path"], decision)
    with pytest.raises(RuntimeError, match="per-seed metrics"):
        _final_evidence(
            **paths,
            validation_summary_path=validation_summary,
            validation_gate_path=validation_gate,
        )


def _ablation_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    timings = {
        "ablation_P_beta0_rescan_adaptive": (200_000, 220_000),
        "native_P_beta0": (100_000, 120_000),
        "ablation_F_heap_all_exact": (180_000, 200_000),
        "native_F": (90_000, 100_000),
        "faiss_A_reference": (70_000, 80_000),
        "faiss_A": (75_000, 85_000),
    }
    for position in (0, 1):
        for method, (micro, api) in timings.items():
            rows.append(
                {
                    "run_id": "run",
                    "session_id": "session",
                    "query_id": str(position),
                    "query_position": position,
                    "experiment_id": "sift-initial",
                    "method": method,
                    "micro_latency_ns": micro + position,
                    "api_wall_latency_ns": api + position,
                    "full_visible_exact_recall": 0.9 + position / 20,
                    "exact_boundary_rechecks": (
                        None if method == "faiss_A_reference" else 2 + position
                    ),
                }
            )
    return rows


def _cache_audits() -> dict[str, list[dict[str, object]]]:
    rows: list[dict[str, object]] = []
    for position in (0, 1):
        for mode, wall in (("cached", 10_000), ("legacy_rebuild", 30_000)):
            rows.append(
                {
                    "query_id": position,
                    "query_position": position,
                    "mode": mode,
                    "wall_ns": wall + position,
                    "candidate_hash": f"hash-{position}",
                    "candidate_keys": [[position, 0]],
                }
            )
    return {"sift-initial": [{"query_count": 2, "rows": rows}]}


def test_measured_ablation_table_is_complete_and_fail_closed() -> None:
    profile = {
        "condition": {"development_queries": 5},
        "unprofiled_run": {
            "old_pruning_beta0_micro_milliseconds": 74.0,
            "old_pruning_beta0_e2e_milliseconds": 84.0,
        }
    }
    table = _ablation_table(_ablation_rows(), _cache_audits(), profile)
    rendered = "\n".join(table)
    assert len(table) == 5
    assert "P rescan → heap" in rendered
    assert "F all-exact → adaptive" in rendered
    assert "Base legacy rebuild → cached" in rendered
    assert "3.000" in rendered
    assert "**nonpaired**" in rendered
    assert "A-reference → A" in rendered and "exact rechecks" in rendered

    incomplete = [
        row for row in _ablation_rows() if row["method"] != "ablation_F_heap_all_exact"
    ]
    with pytest.raises(RuntimeError, match="missing/invalid measured ablation"):
        _ablation_table(incomplete, _cache_audits(), profile)


def _completed_run_fixture(tmp_path: Path) -> tuple[Path, list[dict[str, object]]]:
    run = tmp_path / "run"
    (run / "audits").mkdir(parents=True)
    (run / "raw").mkdir()
    build = {
        "sift-initial": {"dataset_hash": "dataset", "split_id": "split"}
    }
    lb = {
        "experiment_id": "sift-initial",
        "dataset_hash": "dataset",
        "split_id": "split",
        "passed": True,
        "rows": [],
    }
    cache = {
        "experiment_id": "sift-initial",
        "dataset_hash": "dataset",
        "split_id": "split",
        "query_count": 1,
        "rows": [
            {
                "query_id": 0,
                "query_position": 0,
                "mode": "cached",
                "wall_ns": 1,
                "candidate_hash": "same",
                "candidate_keys": [[0, 0]],
            },
            {
                "query_id": 0,
                "query_position": 0,
                "mode": "legacy_rebuild",
                "wall_ns": 2,
                "candidate_hash": "same",
                "candidate_keys": [[0, 0]],
            },
        ],
    }
    _write(run / "build_manifest.json", build)
    _write(run / "audits" / "sift-initial.native-lb.json", lb)
    _write(run / "audits" / "sift-initial.base-cache.json", cache)
    _write(run / "raw" / "part.jsonl", {"row": 1})
    _write(
        run / "run_manifest.json",
        {
            "run_id": "run-1",
            "config_hash": "c" * 64,
            "implementation_tree_sha256": "i" * 64,
        },
    )
    ancillary = []
    for relative in (
        "build_manifest.json",
        "audits/sift-initial.native-lb.json",
        "audits/sift-initial.base-cache.json",
    ):
        path = run / relative
        ancillary.append(
            {"path": relative, "bytes": path.stat().st_size, "sha256": file_sha256(path)}
        )
    _write(run / "COMPLETED.json", {"ancillary_files": ancillary})
    source_runs: list[dict[str, object]] = [
        {
            "run_id": "run-1",
            "config_hash": "c" * 64,
            "implementation_tree_sha256": "i" * 64,
            "completion_sha256": file_sha256(run / "COMPLETED.json"),
            "raw_files": [
                {
                    "path": "raw/part.jsonl",
                    "sha256": file_sha256(run / "raw" / "part.jsonl"),
                    "rows": 1,
                }
            ],
        }
    ]
    return run, source_runs


def test_completed_artifacts_bind_exactly_to_analyzer_runs(tmp_path: Path) -> None:
    _, source_runs = _completed_run_fixture(tmp_path)
    builds, audits, cache, bound = _completed_artifacts(tmp_path, source_runs)
    assert set(builds) == set(audits) == set(cache) == {"sift-initial"}
    assert bound[0]["raw_files"][0]["bytes"] > 0

    orphan = tmp_path / "orphan"
    orphan.mkdir()
    _write(orphan / "run_manifest.json", {"run_id": "orphan"})
    _write(orphan / "COMPLETED.json", {"ancillary_files": []})
    with pytest.raises(RuntimeError, match="orphan/unaccepted"):
        _completed_artifacts(tmp_path, source_runs)


def test_completed_artifacts_reject_symlinked_ancillary(tmp_path: Path) -> None:
    run, source_runs = _completed_run_fixture(tmp_path)
    target = run / "linked-target.json"
    _write(target, {"passed": True})
    link = run / "audits" / "linked.native-lb.json"
    link.symlink_to("../linked-target.json")
    completion_path = run / "COMPLETED.json"
    completion = json.loads(completion_path.read_text(encoding="utf-8"))
    completion["ancillary_files"].append(
        {
            "path": "audits/linked.native-lb.json",
            "bytes": link.stat().st_size,
            "sha256": file_sha256(link),
        }
    )
    _write(completion_path, completion)
    source_runs[0]["completion_sha256"] = file_sha256(completion_path)
    with pytest.raises(RuntimeError, match="ancillary identity mismatch"):
        _completed_artifacts(tmp_path, source_runs)


def test_completed_performance_negative_requires_hnsw_not_run(tmp_path: Path) -> None:
    paths, validation_summary, validation_gate, decision = _write_completed_fixture(
        tmp_path,
        engineering_decision="NO_GO",
    )
    final_gate = json.loads(paths["gate_path"].read_text(encoding="utf-8"))
    for role in ("F", "A"):
        final_gate["comparison_outcomes"][role].update(
            {
                "passed": False,
                "performance_gate_passed": False,
                "engineering_GO_passed": False,
                "bootstrap_CI_supports_speedup": False,
                "bootstrap_95pct_lower": 0.99,
            }
        )
    final_gate.update(
        {
            "gate_status": "NOT_PASSED",
            "performance_gate_status": "NOT_PASSED",
            "performance_verdict": "NOT_SUPPORTED_IN_TESTED_REGIME",
            "verdict": "NOT_SUPPORTED_IN_TESTED_REGIME",
            "engineering_decision": "NO_GO",
            "engineering_reasons": [
                "performance_gate_failed_vs_F",
                "performance_gate_failed_vs_A",
            ],
            "passed": False,
            "hnsw_reference_status": "not_run_performance_gate_not_passed",
        }
    )
    final_gate.pop("hnsw_completion_sha256")
    final_gate.pop("pre_hnsw_authorization_sha256")
    final_summary = json.loads(paths["summary_path"].read_text(encoding="utf-8"))
    final_summary["final_gate"] = final_gate
    final_summary["hnsw_references"] = {
        "status": "not_run_performance_gate_not_passed",
        "role": "approximate_reference_only_not_in_final_gate",
        "reason": "fresh F/A/P performance gate did not pass",
    }
    _write(paths["gate_path"], final_gate)
    _write(paths["summary_path"], final_summary)
    decision.update(
        {
            "final_status": "COMPLETED_NOT_PASSED",
            "performance_gate_status": "NOT_PASSED",
            "performance_verdict": "NOT_SUPPORTED_IN_TESTED_REGIME",
            "verdict": "NOT_SUPPORTED_IN_TESTED_REGIME",
            "engineering_decision": "NO_GO",
            "hnsw_reference_status": "not_run_performance_gate_not_passed",
            "final_gate_sha256": file_sha256(paths["gate_path"]),
            "final_summary_sha256": file_sha256(paths["summary_path"]),
        }
    )
    decision.pop("hnsw_completion_sha256")
    decision.pop("pre_hnsw_authorization_sha256")
    paths["pre_hnsw_authorization_path"].unlink()
    paths["hnsw_path"].rmdir()
    _write(paths["decision_path"], decision)

    loaded, section = _final_evidence(
        **paths,
        validation_summary_path=validation_summary,
        validation_gate_path=validation_gate,
    )
    assert loaded["final_status"] == "COMPLETED_NOT_PASSED"
    assert "HNSW reference は実行していない" in section


def test_negative_decision_rejects_stale_final_summary_and_gate(tmp_path: Path) -> None:
    clean = tuple(
        tmp_path / name
        for name in ("lock", "config", "summary", "gate", "pre-hnsw", "hnsw")
    )
    _reject_negative_final_artifacts(clean)
    _write(clean[2], {"stale": True})
    with pytest.raises(RuntimeError, match="final artifacts already exist"):
        _reject_negative_final_artifacts(clean)


def test_decision_rejects_hnsw_paths_that_differ_from_policy(tmp_path: Path) -> None:
    registered_pre = tmp_path / "registered-pre.json"
    registered_hnsw = tmp_path / "registered-hnsw"
    policy = {
        "pre_hnsw_authorization": str(registered_pre),
        "hnsw_output_root": str(registered_hnsw),
    }
    _verify_registered_hnsw_outputs(
        policy,
        pre_hnsw_authorization_output=registered_pre,
        hnsw_output=registered_hnsw,
    )
    with pytest.raises(RuntimeError, match="differ from the registered policy"):
        _verify_registered_hnsw_outputs(
            policy,
            pre_hnsw_authorization_output=tmp_path / "override-pre.json",
            hnsw_output=registered_hnsw,
        )


def test_negative_final_cli_does_not_create_lock_config_hnsw_or_analysis(
    tmp_path: Path,
) -> None:
    validation_summary, validation_gate = _validation_files(tmp_path, "NOT_PASSED")
    paths = _paths(tmp_path)
    _write(
        paths["decision_path"],
        _negative_decision(validation_summary, validation_gate),
    )
    hnsw = paths["hnsw_path"]
    environment = dict(os.environ)
    environment.update({name: "1" for name in THREAD_VARIABLES})
    run = subprocess.run(
        [
            sys.executable,
            str(REPOSITORY / "scripts" / "run_native_final.py"),
            "--decision",
            str(paths["decision_path"]),
            "--lock",
            str(paths["lock_path"]),
            "--config",
            str(paths["config_path"]),
            "--hnsw-script",
            str(REPOSITORY / "scripts" / "run_native_hnsw_references.py"),
            "--hnsw-output",
            str(hnsw),
        ],
        cwd=REPOSITORY,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    assert run.returncode == 0, run.stderr
    analysis = subprocess.run(
        [
            sys.executable,
            str(REPOSITORY / "scripts" / "analyze_native_final.py"),
            "--decision",
            str(paths["decision_path"]),
            "--lock",
            str(paths["lock_path"]),
            "--config",
            str(paths["config_path"]),
            "--hnsw-input",
            str(hnsw),
            "--summary-output",
            str(paths["summary_path"]),
            "--gate-output",
            str(paths["gate_path"]),
        ],
        cwd=REPOSITORY,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    assert analysis.returncode == 0, analysis.stderr
    assert not any(
        path.exists()
        for path in (
            paths["lock_path"],
            paths["config_path"],
            paths["summary_path"],
            paths["gate_path"],
            hnsw,
        )
    )

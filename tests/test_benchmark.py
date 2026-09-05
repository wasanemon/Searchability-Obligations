from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import pytest
import numpy as np

import searchability.benchmark as benchmark_module
from searchability.artifacts import file_sha256
from searchability.baselines import PreparedFaissIndex
from searchability.benchmark import (
    CompletedRunError,
    _split_payload,
    apply_resource_limits,
    run_benchmark,
)
from searchability.datasets import synthetic_dataset
from searchability.models import CandidateSet, Receipt, VectorRecord
from searchability.search import SearchEngine


def _config(output_root: Path, *, test_queries: int = 4) -> dict[str, object]:
    return {
        "schema_version": 1,
        "run_name": "unit-benchmark",
        "output_root": str(output_root),
        "threads": 1,
        "warmup_queries": 1,
        "repetitions": 2,
        "query_block_size": 2,
        "method_order_seed": 7717,
        "evidence_role": "final",
        "throughput_queries": 2,
        "throughput_repetitions": 1,
        "independent_oracle_queries": 1,
        "independent_oracle_max_population": 64,
        "lb_audit_queries": test_queries,
        "experiments": [
            {
                "id": "tiny-clustered",
                "dataset": {
                    "type": "synthetic",
                    "kind": "clustered",
                    "n_base": 32,
                    "n_delta": 8,
                    "n_validation": 2,
                    "n_test": test_queries,
                    "dimension": 4,
                    "clusters": 3,
                    "seed": 19,
                },
                "k": 3,
                "candidate_count": 8,
                "n_groups": 2,
                "raw_pending_count": 1,
                "center_training_size": 32,
                "group_seed": 23,
                "kmeans_iterations": 3,
                "positive_beta_factors": [0.05],
                "positive_beta_min": 1e-12,
                "hnsw_m": 4,
                "hnsw_ef_construction": 16,
                "hnsw_ef_search": [4, 12],
            }
        ],
    }


def _raw_rows(run_dir: Path) -> list[dict[str, object]]:
    rows = []
    for path in sorted((run_dir / "raw").glob("*.jsonl")):
        rows.extend(json.loads(line) for line in path.read_text().splitlines() if line)
    return rows


def test_full_harness_reuses_one_candidate_object_and_writes_all_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    candidate_objects: dict[bytes, set[int]] = {}
    observed_uses: list[tuple[bytes, int]] = []
    original_prepare = SearchEngine.prepare_candidates
    original_full = SearchEngine.search_full_scan
    original_pruned = SearchEngine.search_pruned
    original_merge = PreparedFaissIndex.search_and_merge_candidates_performance

    def tracked_prepare(self: SearchEngine, query: object, **kwargs: object):
        candidate = original_prepare(self, query, **kwargs)
        candidate_objects.setdefault(query.tobytes(), set()).add(id(candidate))  # type: ignore[union-attr]
        return candidate

    def tracked_full(self: SearchEngine, query: object, **kwargs: object):
        candidate = kwargs["candidate_set"]
        observed_uses.append((query.tobytes(), id(candidate)))  # type: ignore[union-attr]
        assert id(candidate) in candidate_objects[query.tobytes()]  # type: ignore[union-attr]
        return original_full(self, query, **kwargs)

    def tracked_pruned(self: SearchEngine, query: object, **kwargs: object):
        candidate = kwargs["candidate_set"]
        observed_uses.append((query.tobytes(), id(candidate)))  # type: ignore[union-attr]
        assert id(candidate) in candidate_objects[query.tobytes()]  # type: ignore[union-attr]
        return original_pruned(self, query, **kwargs)

    def tracked_merge(self: PreparedFaissIndex, query: object, **kwargs: object):
        candidate = kwargs["base_candidates"]
        observed_uses.append((query.tobytes(), id(candidate)))  # type: ignore[union-attr]
        assert id(candidate) in candidate_objects[query.tobytes()]  # type: ignore[union-attr]
        return original_merge(self, query, **kwargs)

    monkeypatch.setattr(SearchEngine, "prepare_candidates", tracked_prepare)
    monkeypatch.setattr(SearchEngine, "search_full_scan", tracked_full)
    monkeypatch.setattr(SearchEngine, "search_pruned", tracked_pruned)
    monkeypatch.setattr(
        PreparedFaissIndex,
        "search_and_merge_candidates_performance",
        tracked_merge,
    )

    outcome = run_benchmark(_config(tmp_path / "runs"))
    assert outcome.completed
    assert outcome.completed_blocks == outcome.total_blocks == 2
    assert observed_uses
    for name in (
        "effective_config.json",
        "run_manifest.json",
        "checkpoint.json",
        "build_manifest.json",
        "COMPLETED.json",
        "splits/tiny-clustered.json",
        "validation/tiny-clustered.json",
        "throughput/tiny-clustered.json",
        "oracle_checks/tiny-clustered.json",
    ):
        assert (outcome.run_dir / name).is_file()

    rows = _raw_rows(outcome.run_dir)
    methods = {str(row["method"]) for row in rows}
    assert {
        "delta_flat_full_scan",
        "certified_full_delta_reference",
        "group_pruning_beta0",
        "group_pruning_beta0_recompute_intervals_ablation",
        "group_pruning_beta_factor_0p05",
        "delta_hnsw_ef4",
        "delta_hnsw_ef12",
        "full_base_delta_hnsw_ef4",
        "full_base_delta_hnsw_ef12",
        "full_base_delta_flat_scan_performance",
        "full_base_delta_exact_flat",
        "grouping_no_pruning_ablation",
    } == methods
    timed_method_count = len(methods) - 2
    assert len(rows) == 4 * (2 * timed_method_count + 2)
    assert all(int(row["threads"]) == 1 for row in rows)
    coupled = [row for row in rows if row["micro_latency_ns"] is not None]
    assert coupled
    assert all(
        int(row["end_to_end_latency_ns"])
        == int(row["base_prepare_wall_ns"]) + int(row["latency_wall_ns"])
        for row in coupled
    )
    assert not [row for row in rows if row["contract_violation"]]
    positive = [row for row in rows if row["method"] == "group_pruning_beta_factor_0p05"]
    assert positive and all(float(row["requested_beta_l2"]) > 0 for row in positive)
    assert all(row["dataset_hash"] and row["split_id"] for row in rows)
    assert all(row["implementation_tree_sha256"] for row in rows)
    assert all(int(row["base_hnsw_ef_search"]) == 128 for row in rows)
    truth = [row for row in rows if row["method_kind"] in {"certified_full", "full_exact_truth"}]
    assert len(truth) == 4 * 2
    assert all(int(row["repetition"]) == -1 for row in truth)
    certified = [row for row in rows if row["method_kind"] in {"certified_full", "pruned", "no_pruning"}]
    certified.extend(row for row in rows if row["method_kind"] == "pruned_recompute")
    assert certified and all(isinstance(row["receipt"], dict) for row in certified)
    recompute = [
        row
        for row in rows
        if row["method"] == "group_pruning_beta0_recompute_intervals_ablation"
    ]
    assert recompute
    assert all(
        row["receipt"]["certificate_details"]["final_interval_strategy"]
        == "recomputed_full_survivor_matrix_ablation"
        for row in recompute
    )
    audited = [row for row in rows if row["lb_audit_non_timed"] is not None]
    assert len(audited) == 4

    throughput = json.loads(
        (outcome.run_dir / "throughput/tiny-clustered.json").read_text()
    )["rows"]
    assert throughput
    assert all(float(row["end_to_end_actual_batch_qps"]) > 0 for row in throughput)
    oracle_rows = json.loads(
        (outcome.run_dir / "oracle_checks/tiny-clustered.json").read_text()
    )["rows"]
    assert len(oracle_rows) == 1
    assert oracle_rows[0]["status"] == "executed_match"

    output = tmp_path / "analysis"
    report = tmp_path / "REPORT_ja.generated.md"
    figures = tmp_path / "figures"
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/analyze_results.py",
            "--input",
            str(tmp_path / "runs"),
            "--output",
            str(output),
            "--report",
            str(report),
            "--figures",
            str(figures),
            "--evidence-role",
            "final",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    summary = json.loads((output / "summary.json").read_text())
    assert summary["totals"]["contract_violation_rows"] == 0
    assert summary["totals"]["baseline_validation_failure_rows"] == 0
    assert summary["totals"]["independent_oracle_executed"] == 1
    assert summary["totals"]["raw_rows"] == len(rows)
    assert all(
        "end_to_end_single_query_capacity_estimate_qps" in row
        and "end_to_end_qps" not in row
        for row in summary["method_summaries"]
    )
    measured_summaries = [
        row
        for row in summary["method_summaries"]
        if row["method_kind"] not in {"certified_full", "full_exact_truth"}
    ]
    assert measured_summaries
    assert all(row["end_to_end_actual_batch_qps"]["count"] == 1 for row in measured_summaries)
    assert (output / "summary.csv").is_file()
    assert "保存済み JSONL" in report.read_text()
    assert (figures / "latency_quantiles.png").is_file()


def test_query_block_checkpoint_resume_and_completed_run_non_destructive(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path / "runs", test_queries=4)
    first = run_benchmark(config, run_id="checkpoint-case", stop_after_blocks=1)
    assert not first.completed
    assert not (first.run_dir / "COMPLETED.json").exists()
    checkpoint = json.loads((first.run_dir / "checkpoint.json").read_text())
    assert len(checkpoint["completed_blocks"]) == 1
    first_entry = next(iter(checkpoint["completed_blocks"].values()))
    first_part = first.run_dir / first_entry["path"]
    first_hash = file_sha256(first_part)
    first_mtime = first_part.stat().st_mtime_ns

    resumed = run_benchmark(config, resume=True, run_id="checkpoint-case")
    assert resumed.completed
    assert resumed.completed_blocks == resumed.total_blocks == 2
    assert file_sha256(first_part) == first_hash
    assert first_part.stat().st_mtime_ns == first_mtime
    manifest_before = (resumed.run_dir / "run_manifest.json").read_bytes()
    with pytest.raises(CompletedRunError):
        run_benchmark(config, resume=True, run_id="checkpoint-case")
    assert (resumed.run_dir / "run_manifest.json").read_bytes() == manifest_before


def test_final_analysis_excludes_incomplete_run_but_preserves_failure_state(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path / "runs", test_queries=2)
    outcome = run_benchmark(config, run_id="incomplete-final", stop_after_blocks=1)
    assert not outcome.completed
    output = tmp_path / "analysis"
    report = tmp_path / "report.md"
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/analyze_results.py",
            "--input",
            str(tmp_path / "runs"),
            "--output",
            str(output),
            "--report",
            str(report),
            "--figures",
            str(tmp_path / "figures"),
            "--evidence-role",
            "final",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    summary = json.loads((output / "summary.json").read_text())
    assert summary["totals"]["runs"] == 0
    assert summary["totals"]["raw_rows"] == 0
    assert summary["totals"]["excluded_incomplete_final_runs"] == 1
    excluded = summary["excluded_incomplete_runs"]
    assert excluded[0]["run_id"] == "incomplete-final"
    assert excluded[0]["status"] == "incomplete_controlled_stop"
    assert excluded[0]["failures"] == []
    assert "missing_COMPLETED_json" in excluded[0]["reason"]
    assert "集計から除外した未完了 final run" in report.read_text()


def test_exact_truth_is_reused_across_identical_dataset_query_k_experiments(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path / "runs", test_queries=1)
    config["warmup_queries"] = 0
    config["repetitions"] = 1
    config["throughput_queries"] = 0
    config["independent_oracle_queries"] = 0
    config["lb_audit_queries"] = 0
    first = config["experiments"][0]
    second = json.loads(json.dumps(first))
    first["repetitions"] = 2
    second["id"] = "tiny-clustered-other-index-seed"
    second["group_seed"] = int(second["group_seed"]) + 1
    config["experiments"] = [first, second]

    outcome = run_benchmark(config)
    exact_rows = sorted(
        (
            row
            for row in _raw_rows(outcome.run_dir)
            if row["method_kind"] == "full_exact_truth"
        ),
        key=lambda row: str(row["experiment_id"]),
    )
    assert len(exact_rows) == 2
    misses = [row for row in exact_rows if not row["truth_cache_hit"]]
    hits = [row for row in exact_rows if row["truth_cache_hit"]]
    assert len(misses) == len(hits) == 1
    assert misses[0]["truth_cache_key"] == hits[0]["truth_cache_key"]
    assert misses[0]["result_keys"] == hits[0]["result_keys"]
    assert misses[0]["timing_scope"].startswith("validation_truth_generation")
    assert hits[0]["timing_scope"] == "validation_truth_reused_from_run_cache"
    certified_rows = [
        row
        for row in _raw_rows(outcome.run_dir)
        if row["method_kind"] == "certified_full"
    ]
    assert len(certified_rows) == 2
    assert not any(row["truth_cache_hit"] for row in certified_rows)
    manifest = json.loads((outcome.run_dir / "run_manifest.json").read_text())
    assert manifest["experiment_repetitions"] == {
        "tiny-clustered": 2,
        "tiny-clustered-other-index-seed": 1,
    }


def test_resource_limits_are_hashed_as_effective_configuration(tmp_path: Path) -> None:
    config = _config(tmp_path / "runs")
    config["experiments"][0]["repetitions"] = 7
    effective = apply_resource_limits(
        config,
        max_base=8,
        max_delta=0,
        max_validation_queries=1,
        max_test_queries=1,
        max_repetitions=1,
    )
    experiment = effective["experiments"][0]
    assert experiment["dataset"]["n_base"] == 8
    assert experiment["dataset"]["n_delta"] == 0
    assert experiment["dataset"]["n_validation"] == 1
    assert experiment["dataset"]["n_test"] == 1
    assert effective["repetitions"] == 1
    assert experiment["repetitions"] == 1
    assert effective["applied_resource_limits"]["max_delta"] == 0
    assert effective["source_evidence_role_before_resource_limits"] == "final"
    assert effective["evidence_role"] == "calibration"


def test_certificate_validation_fails_closed_on_missing_receipt_fields(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = Receipt.to_dict

    def incomplete(self: Receipt) -> dict[str, object]:
        value = original(self)
        value.pop("certificate_status", None)
        value.pop("numeric_mode", None)
        return value

    monkeypatch.setattr(Receipt, "to_dict", incomplete)
    config = _config(tmp_path / "runs", test_queries=1)
    config["repetitions"] = 1
    config["throughput_queries"] = 0
    config["independent_oracle_queries"] = 0
    outcome = run_benchmark(config)
    certified = [
        row
        for row in _raw_rows(outcome.run_dir)
        if row["method_kind"] in {"certified_full", "pruned", "no_pruning"}
    ]
    assert certified
    assert all(row["contract_violation"] for row in certified)
    assert all(
        any("missing_receipt_fields" in reason for reason in row["violation_reasons"])
        for row in certified
    )


def test_resume_rejects_a_different_implementation_tree_hash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(tmp_path / "runs", test_queries=2)
    monkeypatch.setattr(
        benchmark_module, "implementation_tree_sha256", lambda root: "a" * 64
    )
    first = run_benchmark(config, run_id="source-change", stop_after_blocks=1)
    assert not first.completed
    monkeypatch.setattr(
        benchmark_module, "implementation_tree_sha256", lambda root: "b" * 64
    )
    with pytest.raises(RuntimeError, match="implementation changed"):
        run_benchmark(config, run_id="source-change", resume=True)


def test_flat_ground_truth_uses_exact_tie_order_not_faiss_label_order() -> None:
    """Faiss returns the first inserted label for the deterministic +/-1 tie."""

    records = (
        VectorRecord(9, 0, np.asarray([+1.0], dtype=np.float32)),
        VectorRecord(3, 0, np.asarray([-1.0], dtype=np.float32)),
    )
    index = PreparedFaissIndex(records, kind="flat", threads=1)
    assert not index.vectors.flags.writeable
    with pytest.raises(ValueError):
        index.vectors.setflags(write=True)
    query = np.asarray([0.0], dtype=np.float32)

    _, raw_labels = index.index.search(query[None, :], 1)
    assert int(raw_labels[0, 0]) == 0  # insertion-order ID 9: the old bug
    exact = index.search(query, k=1)
    assert exact.ids == (3,)
    assert exact.candidate_count == 1
    assert exact.search_ns > 0
    assert exact.merge_ns == 0


def test_delta_flat_same_c_reference_supplements_faiss_tie_boundary() -> None:
    delta = (
        VectorRecord(9, 0, np.asarray([+1.0], dtype=np.float32)),
        VectorRecord(3, 0, np.asarray([-1.0], dtype=np.float32)),
    )
    index = PreparedFaissIndex(delta, kind="flat", threads=1)
    candidates = CandidateSet(
        records=(VectorRecord(20, 0, np.asarray([10.0], dtype=np.float32)),),
        vectors=np.asarray([[10.0]], dtype=np.float32),
        candidate_hash="frozen-c",
        visibility_rejections=0,
        search_ns=0,
        requested_count=1,
    )

    result = index.search_and_merge_candidates(
        np.asarray([0.0], dtype=np.float32),
        k=1,
        base_candidates=candidates,
        ann_candidate_count=1,
        source="delta_flat",
    )
    assert result.ids == (3,)
    assert result.candidate_count == 1

    performance = index.search_and_merge_candidates_performance(
        np.asarray([0.0], dtype=np.float32),
        k=1,
        base_candidates=candidates,
        ann_candidate_count=1,
    )
    # The optimized path performs no hidden all-Delta certification pass.  Its
    # possible tie mismatch is why benchmark truth is a distinct method/scope.
    assert performance.ids == (9,)


def test_flat_performance_path_is_distinct_from_exact_truth_on_tie() -> None:
    records = (
        VectorRecord(9, 0, np.asarray([+1.0], dtype=np.float32)),
        VectorRecord(3, 0, np.asarray([-1.0], dtype=np.float32)),
    )
    index = PreparedFaissIndex(records, kind="flat", threads=1)
    query = np.asarray([0.0], dtype=np.float32)
    assert index.search_performance(query, k=1, candidate_count=1).ids == (9,)
    assert index.search(query, k=1).ids == (3,)


def test_split_id_is_bound_to_dataset_contents_not_only_id_arrays() -> None:
    first = synthetic_dataset(
        kind="isotropic",
        n_base=8,
        n_delta=2,
        n_validation=1,
        n_test=2,
        dimension=3,
        seed=101,
    )
    second = synthetic_dataset(
        kind="isotropic",
        n_base=8,
        n_delta=2,
        n_validation=1,
        n_test=2,
        dimension=3,
        seed=202,
    )
    assert first.base_ids.tolist() == second.base_ids.tolist()
    assert first.test_query_ids.tolist() == second.test_query_ids.tolist()
    first_hash = first.sha256()
    second_hash = second.sha256()
    assert first_hash != second_hash
    first_payload = _split_payload(first, "same-id-layout", first_hash)
    second_payload = _split_payload(second, "same-id-layout", second_hash)
    assert first_payload["split_id"] != second_payload["split_id"]

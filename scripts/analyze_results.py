#!/usr/bin/env python3
"""Regenerate aggregate tables, figures, and Japanese data notes from raw runs."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import io
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Iterable, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from searchability.artifacts import (
    atomic_write_bytes,
    atomic_write_json,
    file_sha256,
    object_sha256,
)


def _read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def discover_runs(input_path: str | Path) -> list[Path]:
    root = Path(input_path)
    if (root / "run_manifest.json").is_file():
        return [root]
    return sorted(
        path.parent
        for path in root.rglob("run_manifest.json")
        if "analysis" not in path.parts
    )


def _load_rows(run_dir: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    checkpoint_path = run_dir / "checkpoint.json"
    if not checkpoint_path.is_file():
        raise ValueError(f"run has no checkpoint: {run_dir}")
    checkpoint = _read_object(checkpoint_path)
    completed = checkpoint.get("completed_blocks")
    if not isinstance(completed, dict):
        raise ValueError(f"malformed checkpoint: {checkpoint_path}")
    rows: list[dict[str, Any]] = []
    raw_files: list[dict[str, Any]] = []
    for block_key, entry in sorted(completed.items()):
        if not isinstance(entry, dict):
            raise ValueError(f"malformed checkpoint entry {block_key}")
        raw_path = run_dir / str(entry["path"])
        observed_hash = file_sha256(raw_path)
        if observed_hash != entry.get("sha256"):
            raise ValueError(f"raw checksum mismatch: {raw_path}")
        count = 0
        with raw_path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(f"invalid JSONL {raw_path}:{line_number}") from error
                if not isinstance(row, dict):
                    raise ValueError(f"non-object JSONL row {raw_path}:{line_number}")
                rows.append(row)
                count += 1
        if count != int(entry.get("rows", -1)):
            raise ValueError(f"raw row count mismatch: {raw_path}")
        raw_files.append(
            {
                "path": str(raw_path),
                "sha256": observed_hash,
                "rows": count,
                "block_key": block_key,
            }
        )
    return rows, raw_files


def _numbers(rows: Iterable[Mapping[str, Any]], key: str) -> list[float]:
    result = []
    for row in rows:
        value = row.get(key)
        if value is not None:
            converted = float(value)
            if math.isfinite(converted):
                result.append(converted)
    return result


def _quantile(values: Sequence[float], probability: float) -> float | None:
    if not values:
        return None
    return float(np.quantile(np.asarray(values, dtype=np.float64), probability))


def _mean(values: Sequence[float]) -> float | None:
    return None if not values else float(np.mean(np.asarray(values, dtype=np.float64)))


def _rate(numerator: float, denominator: float) -> float | None:
    return None if denominator == 0 else float(numerator / denominator)


def _distribution(values: Sequence[float]) -> dict[str, Any]:
    return {
        "count": len(values),
        "min": min(values) if values else None,
        "p50": _quantile(values, 0.50),
        "p95": _quantile(values, 0.95),
        "p99": _quantile(values, 0.99),
        "max": max(values) if values else None,
        "mean": _mean(values),
    }


def _per_query_medians(
    rows: Sequence[Mapping[str, Any]], key: str
) -> list[float]:
    grouped: dict[tuple[int, int], list[float]] = {}
    for row in rows:
        value = row.get(key)
        if value is None:
            continue
        converted = float(value)
        if math.isfinite(converted):
            identity = (int(row["query_id"]), int(row.get("query_position", -1)))
            grouped.setdefault(identity, []).append(converted)
    return [float(np.median(values)) for _, values in sorted(grouped.items())]


def _aggregate_group(
    key: tuple[str, str, str],
    rows: Sequence[Mapping[str, Any]],
    complete: bool,
    throughput_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    run_id, experiment_id, method = key
    method_kind = str(rows[0].get("method_kind"))
    is_truth_generation = method_kind in {"certified_full", "full_exact_truth"}
    end_to_end_requests = _numbers(rows, "end_to_end_latency_ns")
    uncached_truth_rows = [row for row in rows if not row.get("truth_cache_hit", False)]
    truth_generation_requests = _numbers(
        uncached_truth_rows, "end_to_end_latency_ns"
    )
    micro_requests = _numbers(rows, "micro_latency_ns")
    # Repetitions are not extra independent queries.  Primary percentiles use
    # one median observation per unique query; pooled request values remain in
    # explicitly named diagnostic fields.
    end_to_end = _per_query_medians(rows, "end_to_end_latency_ns")
    micro = _per_query_medians(rows, "micro_latency_ns")
    performance_end_to_end = [] if is_truth_generation else end_to_end
    performance_micro = [] if is_truth_generation else micro
    recalls = _numbers(rows, "recall_at_k_exact_full_visible")
    reference_recalls = _numbers(rows, "recall_at_k_same_c_reference")
    id_differences = _numbers(rows, "id_symmetric_difference_exact")
    rank_differences = _numbers(rows, "rank_position_difference_exact")
    vectors = _numbers(rows, "vectors_scanned_or_read")
    observed = _numbers(rows, "observed_beta_upper_l2")
    certified = _numbers(rows, "certified_beta_l2")
    requested = _numbers(rows, "requested_beta_l2")
    groups_scanned = sum(_numbers(rows, "groups_scanned"))
    groups_skipped = sum(_numbers(rows, "groups_skipped"))
    delta_rows = [row for row in rows if row.get("delta_in_same_c_reference_topk")]
    non_delta_rows = [row for row in rows if not row.get("delta_in_same_c_reference_topk")]
    fallbacks = sum(bool(row.get("fallback")) for row in rows)
    violations = sum(bool(row.get("contract_violation")) for row in rows)
    baseline_failures = sum(bool(row.get("baseline_validation_failure")) for row in rows)
    unique_queries = {
        (int(row["query_id"]), int(row.get("query_position", -1))) for row in rows
    }
    repetitions = {int(row.get("repetition", 0)) for row in rows}
    timing_keys = sorted(
        {
            timing_key
            for row in rows
            for timing_key in (row.get("component_timings_ns") or {}).keys()
        }
    )
    component_means = {
        timing_key: _mean(
            [
                float(row["component_timings_ns"][timing_key])
                for row in rows
                if timing_key in (row.get("component_timings_ns") or {})
            ]
        )
        for timing_key in timing_keys
    }
    return {
        "run_id": run_id,
        "run_complete": complete,
        "experiment_id": experiment_id,
        "method": method,
        "method_kind": rows[0].get("method_kind"),
        "performance_eligible": not is_truth_generation,
        "dataset_hash": rows[0].get("dataset_hash"),
        "split_id": rows[0].get("split_id"),
        "k": rows[0].get("k"),
        "query_rows": len(rows),
        "unique_test_queries": len(unique_queries),
        "repetitions_observed": len(repetitions),
        "end_to_end_p50_ms": None if not performance_end_to_end else _quantile(performance_end_to_end, 0.50) / 1e6,
        "end_to_end_p95_ms": None if not performance_end_to_end else _quantile(performance_end_to_end, 0.95) / 1e6,
        "end_to_end_p99_ms": None if not performance_end_to_end else _quantile(performance_end_to_end, 0.99) / 1e6,
        "end_to_end_single_query_capacity_estimate_qps": (
            None
            if not performance_end_to_end
            else 1e9 / float(np.mean(performance_end_to_end))
        ),
        "end_to_end_pooled_request_p99_ms": (
            None
            if not end_to_end_requests
            else _quantile(end_to_end_requests, 0.99) / 1e6
        ),
        "micro_p50_ms": None if not performance_micro else _quantile(performance_micro, 0.50) / 1e6,
        "micro_p95_ms": None if not performance_micro else _quantile(performance_micro, 0.95) / 1e6,
        "micro_p99_ms": None if not performance_micro else _quantile(performance_micro, 0.99) / 1e6,
        "micro_single_query_capacity_estimate_qps": (
            None if not performance_micro else 1e9 / float(np.mean(performance_micro))
        ),
        "micro_pooled_request_p99_ms": (
            None if not micro_requests else _quantile(micro_requests, 0.99) / 1e6
        ),
        "end_to_end_actual_batch_qps": _distribution(
            _numbers(throughput_rows, "end_to_end_actual_batch_qps")
        ),
        "micro_actual_batch_qps": _distribution(
            _numbers(throughput_rows, "micro_actual_batch_qps")
        ),
        "truth_generation_wall_ms": (
            _distribution([value / 1e6 for value in truth_generation_requests])
            if is_truth_generation
            else _distribution([])
        ),
        "truth_cache_hit_count": sum(
            bool(row.get("truth_cache_hit", False)) for row in rows
        ),
        "mean_exact_recall_at_k": _mean(recalls),
        "min_exact_recall_at_k": min(recalls) if recalls else None,
        "mean_same_c_reference_recall_at_k": _mean(reference_recalls),
        "mean_exact_id_symmetric_difference": _mean(id_differences),
        "mean_exact_rank_position_difference": _mean(rank_differences),
        "mean_vectors_scanned_or_read": _mean(vectors),
        "groups_scanned_total": groups_scanned,
        "groups_skipped_total": groups_skipped,
        "group_skip_rate": _rate(groups_skipped, groups_scanned + groups_skipped),
        "fallback_count": fallbacks,
        "fallback_rate": _rate(fallbacks, len(rows)),
        "contract_violation_count": violations,
        "baseline_validation_failure_count": baseline_failures,
        "requested_beta_l2_min": min(requested) if requested else None,
        "requested_beta_l2_max": max(requested) if requested else None,
        "certified_beta_l2_max": max(certified) if certified else None,
        "observed_beta_upper_l2_max": max(observed) if observed else None,
        "delta_influence_query_fraction": _rate(len(delta_rows), len(rows)),
        "delta_influence_end_to_end_p50_ms": (
            None
            if not delta_rows
            else _quantile(_numbers(delta_rows, "end_to_end_latency_ns"), 0.50) / 1e6
        ),
        "delta_influence_mean_exact_recall": _mean(
            _numbers(delta_rows, "recall_at_k_exact_full_visible")
        ),
        "non_delta_end_to_end_p50_ms": (
            None
            if not non_delta_rows
            else _quantile(_numbers(non_delta_rows, "end_to_end_latency_ns"), 0.50) / 1e6
        ),
        "component_timing_mean_ns": component_means,
        "p99_sample_warning": (
            "fewer_than_1000_unique_test_queries"
            if len(unique_queries) < 1000
            else "empirical_1000_plus_query_tail_still_has_sampling_uncertainty"
        ),
    }


def _query_latency_map(
    rows: Sequence[Mapping[str, Any]], key: str
) -> dict[tuple[int, int], float]:
    values: dict[tuple[int, int], list[float]] = {}
    for row in rows:
        if row.get("comparison_valid") is False:
            continue
        value = row.get(key)
        if value is None:
            continue
        converted = float(value)
        if not math.isfinite(converted) or converted <= 0.0:
            continue
        identity = (int(row["query_id"]), int(row.get("query_position", -1)))
        values.setdefault(identity, []).append(converted)
    return {identity: float(np.median(items)) for identity, items in values.items()}


def _add_paired_speedups(
    summaries: list[dict[str, Any]],
    grouped: Mapping[tuple[str, str, str], Sequence[Mapping[str, Any]]],
) -> None:
    """Attach deterministic paired-query bootstrap intervals vs Delta Flat."""

    baseline_name = "delta_flat_full_scan"
    for summary in summaries:
        run_id = str(summary["run_id"])
        experiment_id = str(summary["experiment_id"])
        if not summary.get("performance_eligible", True):
            summary["paired_speedup_vs_delta_flat"] = {
                "paired_queries": 0,
                "median": None,
                "bootstrap_95pct_lower": None,
                "bootstrap_95pct_upper": None,
                "bootstrap_resamples": 0,
                "seed": None,
                "reason": "truth_generation_is_not_a_performance_method",
            }
            continue
        baseline_rows = grouped.get((run_id, experiment_id, baseline_name), ())
        method_rows = grouped.get(
            (run_id, experiment_id, str(summary["method"])), ()
        )
        baseline = _query_latency_map(baseline_rows, "end_to_end_latency_ns")
        method = _query_latency_map(method_rows, "end_to_end_latency_ns")
        identities = sorted(set(baseline).intersection(method))
        ratios = np.asarray(
            [baseline[identity] / method[identity] for identity in identities],
            dtype=np.float64,
        )
        if ratios.size == 0:
            summary["paired_speedup_vs_delta_flat"] = {
                "paired_queries": 0,
                "median": None,
                "bootstrap_95pct_lower": None,
                "bootstrap_95pct_upper": None,
                "bootstrap_resamples": 0,
                "seed": None,
            }
            continue
        seed = int(
            object_sha256(
                {
                    "purpose": "paired_speedup_bootstrap_v1",
                    "run_id": run_id,
                    "experiment_id": experiment_id,
                    "method": summary["method"],
                }
            )[:16],
            16,
        )
        resamples = 2000
        generator = np.random.default_rng(seed)
        indices = generator.integers(0, ratios.size, size=(resamples, ratios.size))
        bootstrap = np.median(ratios[indices], axis=1)
        summary["paired_speedup_vs_delta_flat"] = {
            "paired_queries": int(ratios.size),
            "median": float(np.median(ratios)),
            "bootstrap_95pct_lower": float(np.quantile(bootstrap, 0.025)),
            "bootstrap_95pct_upper": float(np.quantile(bootstrap, 0.975)),
            "bootstrap_resamples": resamples,
            "seed": seed,
            "ratio_definition": "delta_flat_E2E_query_median / method_E2E_query_median",
        }


def _group_build_seed_variation(
    summaries: Sequence[Mapping[str, Any]],
    specifications: Mapping[tuple[str, str], Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Summarize explicitly configured group k-means build-seed replicates."""

    grouped: dict[tuple[Any, ...], list[tuple[int, Mapping[str, Any]]]] = {}
    for summary in summaries:
        if not summary.get("performance_eligible", True):
            continue
        identity = (str(summary["run_id"]), str(summary["experiment_id"]))
        specification = specifications.get(identity)
        if specification is None or specification.get("axis") not in {
            "initial",
            "group_build_seed",
        }:
            continue
        dataset = specification.get("dataset", {})
        key = (
            identity[0],
            dataset.get("type"),
            dataset.get("name", dataset.get("kind")),
            dataset.get("n_base"),
            dataset.get("n_delta"),
            specification.get("k"),
            specification.get("candidate_count"),
            specification.get("n_groups"),
            specification.get("base_hnsw_ef_search", 128),
            summary.get("method"),
        )
        grouped.setdefault(key, []).append(
            (int(specification.get("group_seed", 0)), summary)
        )

    output: list[dict[str, Any]] = []
    for key, entries in sorted(grouped.items(), key=lambda item: str(item[0])):
        if len(entries) < 2:
            continue
        entries.sort(key=lambda item: item[0])
        latencies = np.asarray(
            [float(item[1]["end_to_end_p50_ms"]) for item in entries],
            dtype=np.float64,
        )
        recalls = np.asarray(
            [float(item[1]["mean_exact_recall_at_k"]) for item in entries],
            dtype=np.float64,
        )
        seed = int(
            object_sha256(
                {"purpose": "group_build_seed_bootstrap_v1", "key": key}
            )[:16],
            16,
        )
        generator = np.random.default_rng(seed)
        resamples = 2000
        indices = generator.integers(0, len(entries), size=(resamples, len(entries)))
        latency_means = np.mean(latencies[indices], axis=1)
        recall_means = np.mean(recalls[indices], axis=1)
        output.append(
            {
                "run_id": key[0],
                "dataset": key[2],
                "n_base": key[3],
                "n_delta": key[4],
                "k": key[5],
                "candidate_count": key[6],
                "n_groups": key[7],
                "base_hnsw_ef_search": key[8],
                "method": key[9],
                "group_build_seeds": [item[0] for item in entries],
                "experiments": [item[1]["experiment_id"] for item in entries],
                "end_to_end_p50_ms_mean": float(np.mean(latencies)),
                "end_to_end_p50_ms_sample_stddev": float(np.std(latencies, ddof=1)),
                "end_to_end_p50_ms_bootstrap_mean_95pct": [
                    float(np.quantile(latency_means, 0.025)),
                    float(np.quantile(latency_means, 0.975)),
                ],
                "exact_recall_mean": float(np.mean(recalls)),
                "exact_recall_sample_stddev": float(np.std(recalls, ddof=1)),
                "exact_recall_bootstrap_mean_95pct": [
                    float(np.quantile(recall_means, 0.025)),
                    float(np.quantile(recall_means, 0.975)),
                ],
                "bootstrap_resamples": resamples,
                "bootstrap_seed": seed,
                "warning": "three-seed descriptive uncertainty; not a population guarantee",
            }
        )
    return output


def analyze(
    input_path: str | Path,
    *,
    evidence_roles: set[str] | None = None,
) -> dict[str, Any]:
    run_dirs = discover_runs(input_path)
    if not run_dirs:
        raise ValueError(f"no benchmark runs found beneath {input_path}")
    all_rows: list[dict[str, Any]] = []
    raw_files: list[dict[str, Any]] = []
    runs: list[dict[str, Any]] = []
    excluded_incomplete_runs: list[dict[str, Any]] = []
    run_complete: dict[str, bool] = {}
    throughput_by_key: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    oracle_checks: list[dict[str, Any]] = []
    builds: list[dict[str, Any]] = []
    experiment_specifications: dict[tuple[str, str], dict[str, Any]] = {}
    radii_by_experiment: dict[str, list[float]] = {}
    matching_role_count = 0
    for run_dir in run_dirs:
        manifest = _read_object(run_dir / "run_manifest.json")
        evidence_role = str(manifest.get("evidence_role", "legacy_unspecified"))
        if evidence_roles is not None and evidence_role not in evidence_roles:
            continue
        matching_role_count += 1
        run_id = str(manifest["run_id"])
        config_hash = str(manifest["config_hash"])
        complete = (run_dir / "COMPLETED.json").is_file()
        # A run labelled as final evidence is admissible only after the writer
        # has revalidated every checkpointed shard and atomically published its
        # completion marker.  Keep the failure/status metadata visible without
        # allowing partial raw rows to enter any scientific aggregate.
        if evidence_role == "final" and not complete:
            excluded_incomplete_runs.append(
                {
                    "run_dir": str(run_dir),
                    "run_id": run_id,
                    "config_hash": config_hash,
                    "evidence_role": evidence_role,
                    "status": manifest.get("status"),
                    "failures": manifest.get("failures", []),
                    "completed_blocks": manifest.get("completed_blocks", 0),
                    "total_blocks": manifest.get("total_blocks"),
                    "reason": "final_evidence_missing_COMPLETED_json",
                }
            )
            continue
        rows, files = _load_rows(run_dir)
        effective_path = run_dir / "effective_config.json"
        if effective_path.is_file():
            effective = _read_object(effective_path)
            for specification in effective.get("experiments", []):
                if isinstance(specification, dict) and specification.get("id") is not None:
                    experiment_specifications[(run_id, str(specification["id"]))] = dict(
                        specification
                    )
        for row in rows:
            if row.get("run_id") != run_id or row.get("config_hash") != config_hash:
                raise ValueError(f"raw identity mismatch in {run_dir}")
            implementation_hash = manifest.get("implementation_tree_sha256")
            if implementation_hash is not None and row.get(
                "implementation_tree_sha256"
            ) != implementation_hash:
                raise ValueError(f"raw implementation identity mismatch in {run_dir}")
        run_complete[run_id] = complete
        all_rows.extend(rows)
        raw_files.extend(files)
        runs.append(
            {
                "run_dir": str(run_dir),
                "run_id": run_id,
                "config_hash": config_hash,
                "status": manifest.get("status"),
                "complete_marker": complete,
                "dataset_hashes": manifest.get("dataset_hashes", {}),
                "split_ids": manifest.get("split_ids", {}),
                "completed_blocks": manifest.get("completed_blocks", 0),
                "total_blocks": manifest.get("total_blocks"),
                "environment": manifest.get("environment", {}),
                "measurement_notes": manifest.get("measurement_notes", {}),
                "evidence_role": evidence_role,
            }
        )
        for experiment_id, entry in manifest.get("throughput_artifacts", {}).items():
            path = run_dir / str(entry["path"])
            if file_sha256(path) != entry.get("sha256"):
                raise ValueError(f"throughput checksum mismatch: {path}")
            payload = _read_object(path)
            for throughput_row in payload.get("rows", []):
                key = (run_id, str(experiment_id), str(throughput_row["method"]))
                throughput_by_key.setdefault(key, []).append(dict(throughput_row))
        for experiment_id, entry in manifest.get("oracle_check_artifacts", {}).items():
            path = run_dir / str(entry["path"])
            if file_sha256(path) != entry.get("sha256"):
                raise ValueError(f"oracle-check checksum mismatch: {path}")
            payload = _read_object(path)
            for oracle_row in payload.get("rows", []):
                oracle_checks.append(
                    {"run_id": run_id, **dict(oracle_row)}
                )
        build_path = run_dir / "build_manifest.json"
        if build_path.exists():
            build = _read_object(build_path)
            for experiment_id, details in build.items():
                builds.append(
                    {
                        "run_id": run_id,
                        "experiment_id": experiment_id,
                        "details": details,
                    }
                )
                radii_by_experiment[f"{run_id}/{experiment_id}"] = [
                    float(value)
                    for value in details.get("groups", {}).get("radius_upper_values", [])
                ]

    if not runs and matching_role_count == 0:
        selected = "all" if evidence_roles is None else ",".join(sorted(evidence_roles))
        raise ValueError(f"no benchmark runs with evidence_role={selected} beneath {input_path}")
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for row in all_rows:
        key = (str(row["run_id"]), str(row["experiment_id"]), str(row["method"]))
        grouped.setdefault(key, []).append(row)
    summaries = [
        _aggregate_group(
            key,
            rows,
            run_complete[key[0]],
            throughput_by_key.get(key, []),
        )
        for key, rows in sorted(grouped.items())
    ]
    _add_paired_speedups(summaries, grouped)

    lb_by_experiment: dict[str, list[float]] = {}
    for row in all_rows:
        values = row.get("lb_lower_values_l2")
        if values:
            key = f"{row['run_id']}/{row['experiment_id']}"
            lb_by_experiment.setdefault(key, []).extend(float(value) for value in values)
    distributions = {
        key: {
            "lb_lower_l2": _distribution(lb_by_experiment.get(key, [])),
            "group_radius_upper_l2": _distribution(radii_by_experiment.get(key, [])),
        }
        for key in sorted(set(lb_by_experiment).union(radii_by_experiment))
    }
    violation_rows = [row for row in all_rows if row.get("contract_violation")]
    baseline_failure_rows = [
        row for row in all_rows if row.get("baseline_validation_failure")
    ]
    return {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "input": str(input_path),
        "runs": runs,
        "excluded_incomplete_runs": excluded_incomplete_runs,
        "selected_evidence_roles": (
            None if evidence_roles is None else sorted(evidence_roles)
        ),
        "raw_files": raw_files,
        "method_summaries": summaries,
        "group_build_seed_variation": _group_build_seed_variation(
            summaries, experiment_specifications
        ),
        "build_manifests": builds,
        "oracle_checks": oracle_checks,
        "distributions": distributions,
        "totals": {
            "matching_role_runs_discovered": matching_role_count,
            "runs": len(runs),
            "completed_runs": sum(run["complete_marker"] for run in runs),
            "excluded_incomplete_final_runs": len(excluded_incomplete_runs),
            "raw_rows": len(all_rows),
            "query_method_groups": len(summaries),
            "contract_violation_rows": len(violation_rows),
            "baseline_validation_failure_rows": len(baseline_failure_rows),
            "independent_oracle_executed": sum(
                str(row.get("status", "")).startswith("executed_")
                for row in oracle_checks
            ),
            "independent_oracle_mismatches": sum(
                row.get("match") is False for row in oracle_checks
            ),
            "independent_oracle_skipped": sum(
                str(row.get("status", "")).startswith("skipped_")
                for row in oracle_checks
            ),
        },
        "violation_examples": violation_rows[:100],
        "baseline_validation_failure_examples": baseline_failure_rows[:100],
        "analysis_scope": (
            "Aggregates were regenerated only from checksummed saved JSONL rows and saved manifests; "
            "no search or experiment was rerun. Final-role runs without COMPLETED.json were "
            "listed but excluded from every aggregate."
        ),
    }


def _flatten_summary(row: Mapping[str, Any]) -> dict[str, Any]:
    flattened = {key: value for key, value in row.items() if key != "component_timing_mean_ns"}
    for key, value in (row.get("component_timing_mean_ns") or {}).items():
        flattened[f"component_mean_{key}"] = value
    return flattened


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    flattened = [_flatten_summary(row) for row in rows]
    fieldnames = sorted({key for row in flattened for key in row})
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    writer.writerows(flattened)
    atomic_write_bytes(path, output.getvalue().encode("utf-8"))


def _save_figure(figure: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(descriptor)
    try:
        figure.savefig(temporary_name, format="png", dpi=150, bbox_inches="tight")
        os.replace(temporary_name, path)
    finally:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        plt.close(figure)


def write_figures(figures_dir: Path, summaries: Sequence[Mapping[str, Any]]) -> list[Path]:
    performance = [row for row in summaries if row.get("performance_eligible", True)]
    if not performance:
        return []
    labels = [f"{row['experiment_id']}\n{row['method']}" for row in performance]
    indices = np.arange(len(performance))
    widths = max(9.0, min(30.0, len(performance) * 0.42))

    figure, axis = plt.subplots(figsize=(widths, 6))
    p50 = [float(row.get("end_to_end_p50_ms") or 0.0) for row in performance]
    p95 = [float(row.get("end_to_end_p95_ms") or 0.0) for row in performance]
    p99 = [float(row.get("end_to_end_p99_ms") or 0.0) for row in performance]
    axis.plot(indices, p50, marker="o", label="p50")
    axis.plot(indices, p95, marker=".", label="p95")
    axis.plot(indices, p99, marker="x", label="p99")
    axis.set_ylabel("end-to-end latency (ms)")
    axis.set_xticks(indices, labels, rotation=90, fontsize=7)
    axis.set_title("Saved per-query latency quantiles")
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    latency_path = figures_dir / "latency_quantiles.png"
    _save_figure(figure, latency_path)

    figure, axis = plt.subplots(figsize=(widths, 6))
    recalls = [float(row.get("mean_exact_recall_at_k") or 0.0) for row in performance]
    latencies = [float(row.get("end_to_end_p50_ms") or 0.0) for row in performance]
    axis.scatter(latencies, recalls, alpha=0.8)
    for index, label in enumerate(labels):
        axis.annotate(label.replace("\n", "/"), (latencies[index], recalls[index]), fontsize=5)
    axis.set_xlabel("end-to-end p50 (ms)")
    axis.set_ylabel("mean recall@k vs full exact Flat")
    axis.set_ylim(-0.02, 1.02)
    axis.set_title("Quality / latency observations")
    axis.grid(alpha=0.25)
    quality_path = figures_dir / "quality_latency.png"
    _save_figure(figure, quality_path)

    pruning = [row for row in performance if row.get("groups_scanned_total") or row.get("groups_skipped_total")]
    paths = [latency_path, quality_path]
    if pruning:
        pruning_labels = [f"{row['experiment_id']}\n{row['method']}" for row in pruning]
        scan = [float(row.get("groups_scanned_total") or 0.0) for row in pruning]
        skip = [float(row.get("groups_skipped_total") or 0.0) for row in pruning]
        x = np.arange(len(pruning))
        figure, axis = plt.subplots(figsize=(max(8.0, len(pruning) * 0.5), 5))
        axis.bar(x, scan, label="scanned")
        axis.bar(x, skip, bottom=scan, label="skipped")
        axis.set_xticks(x, pruning_labels, rotation=90, fontsize=7)
        axis.set_ylabel("group decisions")
        axis.set_title("Group scan / skip totals")
        axis.legend()
        pruning_path = figures_dir / "group_scan_skip.png"
        _save_figure(figure, pruning_path)
        paths.append(pruning_path)
    return paths


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return f"{value:.{digits}g}"
    return str(value).replace("|", "\\|")


def japanese_report(summary: Mapping[str, Any], figures: Sequence[Path]) -> str:
    totals = summary["totals"]
    lines = [
        "# 実験生データからの自動集計（最終報告データ節素材）",
        "",
        f"生成時刻 (UTC): `{summary['generated_at_utc']}`",
        "",
        "この文書は保存済み JSONL と manifest のみから再生成した集計素材であり、研究仮説の最終判断を自動で行わない。`final` 指定の未完了 run は集計から fail-closed で除外し、下表に状態を残す。その他の未完了 run は `完了` 列で区別する。",
        "",
        "## 入力と完全性",
        "",
        f"- run 数: {totals['runs']}（完了 marker あり: {totals['completed_runs']}）",
        f"- 未完了のため除外した final run: {totals['excluded_incomplete_final_runs']}",
        f"- query-method 生行数: {totals['raw_rows']}",
        f"- 契約違反として保存された行数: {totals['contract_violation_rows']}",
        f"- optimized baseline 検証失敗行数: {totals['baseline_validation_failure_rows']}",
        f"- 独立 Fraction oracle: 実行 {totals['independent_oracle_executed']} / mismatch {totals['independent_oracle_mismatches']} / 上限による skip {totals['independent_oracle_skipped']}",
        "- raw shard は checkpoint 記載の SHA-256 と行数を再検証済み。検索処理は分析時に再実行していない。",
        "",
        "### Run 識別子",
        "",
        "| run ID | 完了 | config hash | dataset hash / split ID | 状態 |",
        "|---|---:|---|---|---|",
    ]
    for run in summary["runs"]:
        identities = "; ".join(
            f"{name}: {digest[:12]} / {str(run.get('split_ids', {}).get(name, 'N/A'))[:12]}"
            for name, digest in sorted(run.get("dataset_hashes", {}).items())
        )
        lines.append(
            f"| `{run['run_id']}` | {_fmt(run['complete_marker'])} | `{run['config_hash']}` | {identities} | {_fmt(run['status'])} |"
        )

    excluded = summary.get("excluded_incomplete_runs", [])
    if excluded:
        lines.extend(
            [
                "",
                "### 集計から除外した未完了 final run",
                "",
                "| run ID | 状態 | 完了 block / 全 block | failure | 除外理由 |",
                "|---|---|---:|---|---|",
            ]
        )
        for run in excluded:
            failures = json.dumps(run.get("failures", []), ensure_ascii=False)
            lines.append(
                "| `{run_id}` | {status} | {completed}/{total} | {failures} | `{reason}` |".format(
                    run_id=run["run_id"],
                    status=_fmt(run.get("status")),
                    completed=_fmt(run.get("completed_blocks")),
                    total=_fmt(run.get("total_blocks")),
                    failures=_fmt(failures),
                    reason=run["reason"],
                )
            )

    lines.extend(
        [
            "",
            "## 方式別集計",
            "",
            "| run / 条件 | 方式 | 完了 | query | E2E p50/p95/p99 ms | 実測batch QPS p50 | 単query容量推定QPS | exact recall@k 平均 | skip率 | fallback率 | 違反 |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in summary["method_summaries"]:
        latency = "/".join(
            _fmt(row[key])
            for key in ("end_to_end_p50_ms", "end_to_end_p95_ms", "end_to_end_p99_ms")
        )
        lines.append(
            "| `{run}` / `{experiment}` | `{method}` | {complete} | {queries} | {latency} | {batch_qps} | {capacity_qps} | {recall} | {skip} | {fallback} | {violations} |".format(
                run=row["run_id"],
                experiment=row["experiment_id"],
                method=row["method"],
                complete=_fmt(row["run_complete"]),
                queries=row["unique_test_queries"],
                latency=latency,
                batch_qps=_fmt(row["end_to_end_actual_batch_qps"]["p50"]),
                capacity_qps=_fmt(
                    row["end_to_end_single_query_capacity_estimate_qps"]
                ),
                recall=_fmt(row["mean_exact_recall_at_k"]),
                skip=_fmt(row["group_skip_rate"]),
                fallback=_fmt(row["fallback_rate"]),
                violations=row["contract_violation_count"],
            )
        )

    lines.extend(["", "## LB と radius の保存分布", ""])
    for identity, distributions in summary["distributions"].items():
        lb = distributions["lb_lower_l2"]
        radius = distributions["group_radius_upper_l2"]
        lines.append(
            f"- `{identity}`: LB n={lb['count']}, p50={_fmt(lb['p50'])}, p95={_fmt(lb['p95'])}; radius n={radius['count']}, p50={_fmt(radius['p50'])}, p95={_fmt(radius['p95'])}"
        )

    lines.extend(
        [
            "",
            "## 読み方と制約",
            "",
            "- coupled Delta micro は凍結済み base 候補 `C` の取得時間を除外し、E2E はその取得時間を加算する。全体 HNSW / 全体 exact Flat は E2E のみである。",
            "- Faiss の L2 index が返す二乗距離は通常 L2 に変換して保存した。保証付き方式の observed beta は保存 float32 の k 番目距離を Fraction で再評価した区間上限で判定した。独立 oracle の全順位照合は manifest に保存された決定的sampleだけであり、population上限によるskipも隠さない。",
            "- `実測batch QPS` は artifact書込みとtruth検証を除いた単一thread逐次batchの wall time。`単query容量推定QPS` は queryごとの service time 平均の逆数であり、実throughputではない。主percentileはrepeatをqueryごとにmedian化してから集計し、pooled repeat p99はsummaryの補助列にのみ残す。",
            "- `certified_full` と `full_exact_truth` は queryごとに1回だけ生成する検証用truthであり、性能比較対象から除外した。その生成wall timeは summary の `truth_generation_wall_ms` に別保存した。",
            "- HNSW の per-query visited vector 数は利用中の Faiss API から信頼できる形で取得できないため `null`。構築/add は ACID 更新性能ではなく component cost である。",
            "- 1,000 query の p99 も有限標本であり、共有ホストの負荷変動を含む。1,000 未満の条件には summary の警告列が付く。",
            "- Delta が same-C 基準 top-k に入る query の割合、当該部分集合の latency/recall は summary JSON/CSV に保存した。",
            "- この自動素材だけから画像 descriptor の結果を RAG 全般へ外挿しない。",
        ]
    )
    if figures:
        lines.extend(["", "## 図", ""])
        for path in figures:
            lines.append(f"- `{path}`")
    lines.extend(
        [
            "",
            "最終の `SUPPORTED_IN_TESTED_REGIME` / `NOT_SUPPORTED_IN_TESTED_REGIME` / `INCONCLUSIVE` 判定は、完了 run、契約違反、維持費、実データ範囲、negative result を併せて人が行う。",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--report", type=Path, default=Path("reports/REPORT_ja.generated.md")
    )
    parser.add_argument("--figures", type=Path, default=Path("reports/figures"))
    parser.add_argument(
        "--evidence-role",
        action="append",
        choices=("final", "calibration", "obsolete", "legacy_unspecified"),
        help="Include only this manifest evidence role (repeatable)",
    )
    args = parser.parse_args()
    output = args.output or args.input / "analysis"
    output.mkdir(parents=True, exist_ok=True)
    summary = analyze(
        args.input,
        evidence_roles=None if args.evidence_role is None else set(args.evidence_role),
    )
    atomic_write_json(output / "summary.json", summary)
    write_csv(output / "summary.csv", summary["method_summaries"])
    figures = write_figures(args.figures, summary["method_summaries"])
    atomic_write_bytes(
        args.report, japanese_report(summary, figures).encode("utf-8")
    )
    print(
        json.dumps(
            {
                "summary_json": str(output / "summary.json"),
                "summary_csv": str(output / "summary.csv"),
                "report": str(args.report),
                "figures": [str(path) for path in figures],
                "totals": summary["totals"],
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

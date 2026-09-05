"""Crash-safe benchmark harness for the Issue #3 native-kernel recheck.

The measured question is intentionally narrow: after freezing one Base ANN
candidate set ``C`` for a query, compare the compiled certified full Delta
scan (F), an optimized one-pass Faiss Delta Flat scan (A), compiled grouped
scan without pruning (N), and compiled certified group pruning (P).  The old
Python implementation and the independent Fraction oracle are correctness
references only and never enter timing aggregates.

Every completed raw shard is checksum-addressed.  A completed run is
immutable, while an interrupted run can be resumed only with the same config
and implementation-tree hashes.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from fractions import Fraction
import json
import math
import os
from pathlib import Path
import random
import time
import traceback
from typing import Any, Mapping, Sequence

import faiss
import numpy as np

from .artifacts import (
    atomic_write_json,
    atomic_write_jsonl,
    file_sha256,
    implementation_tree_sha256,
    native_test_tree_sha256,
    object_sha256,
)
from .base import BaseIndex
from .baselines import PreparedFaissIndex
from .benchmark import (
    BenchmarkConfigurationError,
    CompletedRunError,
    RunOutcome,
    THREAD_VARIABLES,
    _current_rss_bytes,
    _peak_rss_bytes,
    collect_environment,
    load_json_config,
)
from .datasets import DatasetSplit, load_dataset
from .groups import DeltaStore, GroupDirectory
from .final_policy import (
    FinalPolicyError,
    issue_fixed_decision_values,
    validate_final_policy,
    validate_issue_fixed_decision_values,
)
from .models import CandidateSet, SearchHit, VectorRecord, as_float32_matrix
from .native import native_build_info, native_call_counter, require_native
from .numerics import distance_intervals, stable_topk
from .oracle import (
    exact_squared_l2 as oracle_exact_squared_l2,
    observed_gap_bracket,
    reference_topk,
    sqrt_bracket as oracle_sqrt_bracket,
)
from .search import SearchEngine


NATIVE_BENCHMARK_SCHEMA_VERSION = 2
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
NATIVE_ROLES = {"F", "N", "P"}


@dataclass(frozen=True, slots=True)
class _Method:
    name: str
    role: str
    beta: float = 0.0
    beta_factor: float | None = None
    native_mode: str | None = None
    threshold_mode: str = "heap"
    rank_strategy: str = "adaptive"


@dataclass(slots=True)
class _Prepared:
    engine: SearchEngine
    base: BaseIndex
    delta_flat: PreparedFaissIndex
    base_records: tuple[VectorRecord, ...]
    delta_records: tuple[VectorRecord, ...]
    grouped_records: tuple[VectorRecord, ...]
    raw_records: tuple[VectorRecord, ...]
    full_vectors: np.ndarray
    records_by_key: dict[tuple[int, int], VectorRecord]
    delta_logical_ids: frozenset[int]
    build_manifest: dict[str, Any]


@dataclass(slots=True)
class _Measured:
    hits: tuple[SearchHit, ...]
    micro_ns: int
    api_wall_ns: int
    api_candidate_hash: str
    native_calls: int
    backend: str
    receipt: dict[str, Any] | None
    components: dict[str, int]
    candidate_count: int | None = None
    exact_rechecks: int | None = None


@dataclass(slots=True)
class _ApiMeasurement:
    hits: tuple[SearchHit, ...]
    wall_ns: int
    candidate_hash: str
    native_calls: int
    backend: str


@dataclass(slots=True)
class _MicroMeasurement:
    hits: tuple[SearchHit, ...]
    wall_ns: int
    native_calls: int
    backend: str
    receipt: dict[str, Any] | None
    components: dict[str, int]
    candidate_count: int | None = None
    exact_rechecks: int | None = None


def _json_copy(value: Mapping[str, Any]) -> dict[str, Any]:
    try:
        return json.loads(json.dumps(value, allow_nan=False))
    except (TypeError, ValueError) as error:
        raise BenchmarkConfigurationError(
            "configuration must contain only finite JSON values"
        ) from error


def _materialize_defaults(config: Mapping[str, Any]) -> dict[str, Any]:
    """Freeze inherited config defaults into every experiment before hashing."""

    effective = _json_copy(config)
    defaults = effective.get("defaults", {})
    if not isinstance(defaults, dict):
        raise BenchmarkConfigurationError("defaults must be an object")
    materialized: list[dict[str, Any]] = []
    for value in effective.get("experiments", []):
        if not isinstance(value, dict):
            raise BenchmarkConfigurationError("experiment entries must be objects")
        row = {**defaults, **value}
        if "beta_factors" not in row:
            positive = row.get("positive_beta_factors", [])
            row["beta_factors"] = [0.0, *[float(item) for item in positive]]
        materialized.append(row)
    effective["experiments"] = materialized
    effective["defaults_materialized"] = True
    return effective


def _safe_name(value: str) -> str:
    result = "".join(
        character if character.isalnum() or character in "-_" else "-"
        for character in value
    ).strip("-")
    if not result:
        raise BenchmarkConfigurationError("identifier has no safe characters")
    return result


def _float_name(value: float) -> str:
    return format(value, ".8g").replace("-", "m").replace("+", "p").replace(".", "p")


def _records(vectors: np.ndarray, identifiers: np.ndarray) -> tuple[VectorRecord, ...]:
    return tuple(
        VectorRecord(int(identifier), 0, vector)
        for identifier, vector in zip(identifiers.tolist(), vectors)
    )


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"cannot read {path}") from error
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object in {path}")
    return value


def _validate_config(config: Mapping[str, Any]) -> None:
    if int(config.get("schema_version", -1)) != NATIVE_BENCHMARK_SCHEMA_VERSION:
        raise BenchmarkConfigurationError(
            f"schema_version must be {NATIVE_BENCHMARK_SCHEMA_VERSION}"
        )
    if str(config.get("study_id", "")) != "issue-3-native-recheck":
        raise BenchmarkConfigurationError("study_id must be issue-3-native-recheck")
    if not str(config.get("run_name", "")).strip():
        raise BenchmarkConfigurationError("run_name is required")
    if not str(config.get("output_root", "")).strip():
        raise BenchmarkConfigurationError("output_root is required")
    if int(config.get("threads", 0)) != 1:
        raise BenchmarkConfigurationError("Issue #3 measurements require exactly one thread")
    if int(config.get("repetitions", 0)) <= 0:
        raise BenchmarkConfigurationError("repetitions must be positive")
    if int(config.get("query_block_size", 0)) <= 0:
        raise BenchmarkConfigurationError("query_block_size must be positive")
    if int(config.get("warmup_queries", 0)) < 0:
        raise BenchmarkConfigurationError("warmup_queries cannot be negative")
    phase = str(config.get("phase", ""))
    if phase not in {"smoke", "validation", "final"}:
        raise BenchmarkConfigurationError("phase must be smoke, validation, or final")
    if phase == "validation" and int(config.get("process_sessions", 0)) != 1:
        raise BenchmarkConfigurationError(
            "formal validation requires process_sessions == 1"
        )
    experiments = config.get("experiments")
    if not isinstance(experiments, list) or not experiments:
        raise BenchmarkConfigurationError("experiments must be a non-empty list")
    identifiers: set[str] = set()
    for experiment in experiments:
        if not isinstance(experiment, dict):
            raise BenchmarkConfigurationError("experiment entries must be objects")
        identifier = str(experiment.get("id", ""))
        if not identifier or identifier in identifiers:
            raise BenchmarkConfigurationError("experiment IDs must be unique and non-empty")
        identifiers.add(identifier)
        dataset = experiment.get("dataset")
        if not isinstance(dataset, dict):
            raise BenchmarkConfigurationError(f"{identifier}: dataset is required")
        for field in ("n_base", "n_delta", "n_validation", "n_test"):
            if field not in dataset or int(dataset[field]) < 0:
                raise BenchmarkConfigurationError(f"{identifier}: invalid dataset.{field}")
        if int(dataset["n_base"]) <= 0 or int(dataset["n_test"]) <= 0:
            raise BenchmarkConfigurationError(f"{identifier}: Base/test must be non-empty")
        for field in ("k", "candidate_count", "n_groups"):
            if int(experiment.get(field, 0)) <= 0:
                raise BenchmarkConfigurationError(f"{identifier}: {field} must be positive")
        if int(experiment["n_groups"]) > int(dataset["n_base"]):
            raise BenchmarkConfigurationError(f"{identifier}: too many groups")
        raw = int(experiment.get("raw_pending_count", 0))
        if raw < 0 or raw > int(dataset["n_delta"]):
            raise BenchmarkConfigurationError(f"{identifier}: invalid raw_pending_count")
        factors = [float(value) for value in experiment.get("beta_factors", [0.0])]
        if 0.0 not in factors:
            raise BenchmarkConfigurationError(f"{identifier}: beta_factors must include 0")
        if any(not math.isfinite(value) or value < 0.0 for value in factors):
            raise BenchmarkConfigurationError(f"{identifier}: invalid beta factor")
        if len(set(factors)) != len(factors):
            raise BenchmarkConfigurationError(f"{identifier}: duplicate beta factor")


def apply_native_resource_limits(
    config: Mapping[str, Any],
    *,
    only_experiment: str | None = None,
    max_experiments: int | None = None,
    max_base: int | None = None,
    max_delta: int | None = None,
    max_validation_queries: int | None = None,
    max_test_queries: int | None = None,
    max_repetitions: int | None = None,
) -> dict[str, Any]:
    """Apply explicit smoke/debug reductions and relabel them calibration-only."""

    effective = _materialize_defaults(config)
    experiments = list(effective.get("experiments", []))
    if only_experiment is not None:
        experiments = [row for row in experiments if row.get("id") == only_experiment]
        if not experiments:
            raise BenchmarkConfigurationError(f"unknown experiment {only_experiment!r}")
    if max_experiments is not None:
        if max_experiments <= 0:
            raise BenchmarkConfigurationError("max_experiments must be positive")
        experiments = experiments[:max_experiments]
    for row in experiments:
        dataset = row["dataset"]
        for field, limit in (
            ("n_base", max_base),
            ("n_delta", max_delta),
            ("n_validation", max_validation_queries),
            ("n_test", max_test_queries),
        ):
            if limit is not None:
                if limit < 0 or (field != "n_delta" and limit == 0):
                    raise BenchmarkConfigurationError(f"invalid limit for {field}")
                dataset[field] = min(int(dataset[field]), int(limit))
        row["n_groups"] = min(int(row["n_groups"]), int(dataset["n_base"]))
        row["raw_pending_count"] = min(
            int(row.get("raw_pending_count", 0)), int(dataset["n_delta"])
        )
        if "center_training_size" in row:
            row["center_training_size"] = min(
                int(row["center_training_size"]), int(dataset["n_base"])
            )
    effective["experiments"] = experiments
    if max_repetitions is not None:
        if max_repetitions <= 0:
            raise BenchmarkConfigurationError("max_repetitions must be positive")
        effective["repetitions"] = min(
            int(effective["repetitions"]), int(max_repetitions)
        )
    limits = {
        "only_experiment": only_experiment,
        "max_experiments": max_experiments,
        "max_base": max_base,
        "max_delta": max_delta,
        "max_validation_queries": max_validation_queries,
        "max_test_queries": max_test_queries,
        "max_repetitions": max_repetitions,
    }
    effective["applied_resource_limits"] = limits
    if any(value is not None for value in limits.values()):
        effective["source_evidence_role_before_resource_limits"] = effective.get(
            "evidence_role"
        )
        effective["evidence_role"] = "calibration"
        effective["phase"] = "smoke"
    return effective


def _new_run_id(output_root: Path, run_name: str, config_hash: str) -> str:
    # Keep generated IDs inside the exact alphabet accepted by _safe_name so
    # an explicit resume/immutability check resolves the identical directory.
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    stem = f"{_safe_name(run_name)}-{stamp}-{config_hash[:10]}"
    result = stem
    suffix = 0
    while (output_root / result).exists():
        suffix += 1
        result = f"{stem}-{suffix}"
    return result


def _select_run_dir(
    config: Mapping[str, Any],
    config_hash: str,
    implementation_hash: str,
    *,
    resume: bool,
    run_id: str | None,
) -> tuple[Path, str]:
    root = Path(str(config["output_root"]))
    root.mkdir(parents=True, exist_ok=True)
    if run_id is not None:
        selected = _safe_name(run_id)
        path = root / selected
        if (path / "COMPLETED.json").exists():
            raise CompletedRunError(f"completed run is immutable: {path}")
        if path.exists() and not resume:
            raise FileExistsError(f"run directory exists: {path}")
        return path, selected
    if resume:
        matching: list[tuple[int, Path]] = []
        completed_matching: list[Path] = []
        for path in root.glob("*/run_manifest.json"):
            if (path.parent / "COMPLETED.json").exists():
                if str(config.get("phase")) == "validation":
                    try:
                        manifest = _read_object(path)
                    except RuntimeError:
                        continue
                    if (
                        manifest.get("config_hash") == config_hash
                        and manifest.get("implementation_tree_sha256")
                        == implementation_hash
                    ):
                        completed_matching.append(path.parent)
                continue
            try:
                manifest = _read_object(path)
            except RuntimeError:
                continue
            if manifest.get("config_hash") == config_hash:
                matching.append((path.stat().st_mtime_ns, path.parent))
        if completed_matching:
            raise CompletedRunError(
                "completed formal validation already exists for this config/code; "
                "refusing to create a duplicate evidence pool: "
                + ", ".join(str(path) for path in sorted(completed_matching))
            )
        if matching:
            selected_path = max(matching)[1]
            return selected_path, selected_path.name
    selected = _new_run_id(root, str(config["run_name"]), config_hash)
    return root / selected, selected


def _has_disqualifying_failures(manifest: Mapping[str, Any]) -> bool:
    """Return true unless every persisted failure is a recoverable orphan event."""

    failures = manifest.get("failures", [])
    return not isinstance(failures, list) or any(
        not isinstance(failure, dict)
        or failure.get("kind") != "uncheckpointed_raw_preserved_on_resume"
        for failure in failures
    )


def _split_payload(split: DatasetSplit, experiment_id: str, dataset_hash: str) -> dict[str, Any]:
    payload = {
        "schema_version": NATIVE_BENCHMARK_SCHEMA_VERSION,
        "experiment_id": experiment_id,
        "dataset_name": split.name,
        "dataset_hash": dataset_hash,
        "dimension": split.dimension,
        "counts": {
            "base": len(split.base),
            "delta": len(split.delta),
            "validation_queries": len(split.validation_queries),
            "test_queries": len(split.test_queries),
        },
        "id_ranges": {
            "base": [int(split.base_ids[0]), int(split.base_ids[-1])] if len(split.base_ids) else None,
            "delta": [int(split.delta_ids[0]), int(split.delta_ids[-1])] if len(split.delta_ids) else None,
            "validation": [int(value) for value in split.validation_query_ids],
            "test": [int(value) for value in split.test_query_ids],
        },
        "metadata": dict(split.metadata),
    }
    payload["split_id"] = object_sha256(payload)
    return payload


def _prepare_experiment(
    split: DatasetSplit, specification: Mapping[str, Any]
) -> _Prepared:
    """Build only the indexes used by F/A/N/P and record each build cost."""

    build_start_rss = _current_rss_bytes()
    build_start_peak = _peak_rss_bytes()
    base_records = _records(split.base, split.base_ids)
    delta_records = _records(split.delta, split.delta_ids)
    raw_count = int(specification.get("raw_pending_count", 0))
    grouped_records = delta_records if raw_count == 0 else delta_records[:-raw_count]
    raw_records = () if raw_count == 0 else delta_records[-raw_count:]
    hnsw_m = int(specification.get("hnsw_m", 32))
    ef_construction = int(specification.get("hnsw_ef_construction", 200))
    ef_search = int(specification.get("base_hnsw_ef_search", 128))

    base_start = time.perf_counter_ns()
    base = BaseIndex(
        base_records,
        dimension=split.dimension,
        generation_id=f"native-recheck-{specification['id']}",
        covered_commit_seq=0,
        m=hnsw_m,
        ef_construction=ef_construction,
        ef_search=ef_search,
        threads=1,
    )
    base_build_ns = time.perf_counter_ns() - base_start

    center_count = min(
        len(split.base), int(specification.get("center_training_size", len(split.base)))
    )
    if center_count <= 0:
        raise BenchmarkConfigurationError("center training population must be positive")
    directory = GroupDirectory.build(
        split.base[:center_count],
        grouped_records,
        int(specification["n_groups"]),
        seed=int(specification.get("group_seed", 0)),
        iterations=int(specification.get("kmeans_iterations", 20)),
        threads=1,
    )
    delta = DeltaStore(raw_records=raw_records, grouped=directory)
    engine = SearchEngine(base, delta)

    flat_start = time.perf_counter_ns()
    delta_flat = PreparedFaissIndex(
        delta_records,
        dimension=split.dimension,
        kind="flat",
        threads=1,
    )
    flat_wall_ns = time.perf_counter_ns() - flat_start

    # This first request is the cold packed-build measurement.  Later requests
    # reuse the immutable SearchEngine cache and are query-timing warm state.
    packed = engine._native_packed_view(0)
    packed_warm_start = time.perf_counter_ns()
    packed_warm = engine._native_packed_view(0)
    packed_warm_lookup_ns = time.perf_counter_ns() - packed_warm_start
    if packed_warm is not packed:
        raise RuntimeError("native packed-view cache did not preserve object identity")

    records_by_key = {record.key: record for record in base_records + delta_records}
    full_vectors = as_float32_matrix(
        np.vstack((split.base, split.delta)), name="full exact truth vectors"
    )
    build_end_rss = _current_rss_bytes()
    build_manifest = {
        "experiment_id": str(specification["id"]),
        "base": {
            "kind": "faiss_hnsw",
            "vectors": len(base_records),
            "dimension": split.dimension,
            "m": hnsw_m,
            "ef_construction": ef_construction,
            "ef_search": ef_search,
            "build_wall_ns": base_build_ns,
            "universe_hash": base.universe_hash,
        },
        "groups": {
            "n_groups": len(directory.groups),
            "grouped_vectors": len(grouped_records),
            "raw_pending_vectors": len(raw_records),
            "raw_ratio": (len(raw_records) / len(delta_records)) if delta_records else 0.0,
            "center_training_source": "base_only",
            "center_training_vectors": center_count,
            "center_training_ns": directory.build_stats.center_training_ns,
            "assignment_ns": directory.build_stats.assignment_ns,
            "packing_and_radius_ns": directory.build_stats.packing_and_radius_ns,
            "member_bytes": directory.build_stats.member_bytes,
            "metadata_bytes": directory.build_stats.metadata_bytes,
            "member_counts": [len(group.members) for group in directory.groups],
            "radius_upper_l2": [group.radius_upper for group in directory.groups],
        },
        "delta_flat": {
            **asdict(delta_flat.build_stats),
            "build_wall_ns": flat_wall_ns,
            "distance_returned_by_faiss": "squared_L2",
            "reported_metric": "ordinary_L2_after_exact_shortlist_rerank",
        },
        "native_packed_view": {
            "hash": packed.packed_view_hash,
            "rows": packed.size,
            "groups": packed.group_count,
            "raw_rows": packed.raw_count,
            "build_ns": packed.build_ns,
            "warm_cache_lookup_ns": packed_warm_lookup_ns,
            "python_owned_bytes": packed.python_owned_bytes,
            "native_owned_bytes": packed.native_owned_bytes,
            "force_full_reason": packed.force_full_reason,
        },
        "native_build": dict(native_build_info()),
        "base_visibility_cache_after_build": base.visible_view_cache_stats(),
        "full_exact_truth_buffer_bytes": int(full_vectors.nbytes),
        "memory": {
            "rss_bytes_before": build_start_rss,
            "rss_bytes_after": build_end_rss,
            "rss_change_bytes": (
                None
                if build_start_rss is None or build_end_rss is None
                else build_end_rss - build_start_rss
            ),
            "peak_rss_bytes_before": build_start_peak,
            "peak_rss_bytes_after": _peak_rss_bytes(),
            "numpy_base_bytes": int(split.base.nbytes),
            "numpy_delta_bytes": int(split.delta.nbytes),
            "full_exact_truth_buffer_bytes": int(full_vectors.nbytes),
            "group_member_bytes": int(directory.build_stats.member_bytes),
            "group_metadata_bytes_estimate": int(directory.build_stats.metadata_bytes),
            "packed_python_owned_bytes": int(packed.python_owned_bytes),
            "packed_native_owned_bytes": int(packed.native_owned_bytes),
        },
    }
    return _Prepared(
        engine=engine,
        base=base,
        delta_flat=delta_flat,
        base_records=base_records,
        delta_records=delta_records,
        grouped_records=grouped_records,
        raw_records=raw_records,
        full_vectors=full_vectors,
        records_by_key=records_by_key,
        delta_logical_ids=frozenset(record.logical_id for record in delta_records),
        build_manifest=build_manifest,
    )


def _validation_betas(
    split: DatasetSplit,
    specification: Mapping[str, Any],
    prepared: _Prepared,
) -> dict[str, Any]:
    """Select absolute beta solely from registered validation queries."""

    k = int(specification["k"])
    candidate_count = int(specification["candidate_count"])
    factors = [float(value) for value in specification.get("beta_factors", [0.0])]
    positive = [value for value in factors if value > 0.0]
    rows: list[dict[str, Any]] = []
    taus: list[float] = []
    for query_id, query in zip(
        split.validation_query_ids.tolist(), split.validation_queries
    ):
        candidates = prepared.engine.prepare_candidates(
            query, snapshot_id=0, k=k, candidate_count=candidate_count
        )
        native_query = prepared.engine.prepare_native_query(
            query,
            snapshot_id=0,
            k=k,
            candidate_count=candidate_count,
            candidate_set=candidates,
        )
        result = prepared.engine.search_native_prepared(
            native_query, k=k, beta=0.0, mode="F"
        )
        tau = result.receipt.tau_returned
        if tau is not None:
            taus.append(float(tau))
        rows.append(
            {
                "query_id": int(query_id),
                "candidate_set_id_or_hash": candidates.candidate_hash,
                "native_F_tau_l2": tau,
                "native_call_index": result.receipt.certificate_details.get(
                    "native_call_index"
                ) if result.receipt.certificate_details else None,
            }
        )
    if positive and not taus:
        raise BenchmarkConfigurationError("positive beta selection produced no kth tau")
    median = float(np.median(np.asarray(taus, dtype=np.float64))) if taus else None
    floor = float(specification.get("positive_beta_min", 1.0e-12))
    if not math.isfinite(floor) or floor <= 0.0:
        raise BenchmarkConfigurationError("positive_beta_min must be finite and positive")
    selected = {
        format(factor, ".17g"): max(floor, factor * float(median))
        for factor in positive
    }
    return {
        "schema_version": NATIVE_BENCHMARK_SCHEMA_VERSION,
        "selection_source": "validation_queries_only",
        "reference_method": "native_F_after_correctness_gate",
        "statistic": "median_kth_ordinary_L2",
        "median_reference_tau_l2": median,
        "positive_beta_floor": floor,
        "factors_to_absolute_beta": selected,
        "validation_query_count": len(rows),
        "validation_rows": rows,
        "test_queries_used_for_selection": 0,
    }


def _methods(
    specification: Mapping[str, Any], beta_selection: Mapping[str, Any]
) -> list[_Method]:
    methods = [
        _Method("native_F", "F", native_mode="F"),
        _Method("faiss_A", "A"),
        _Method("faiss_A_reference", "A-reference"),
        _Method("native_N", "N", native_mode="N"),
    ]
    absolute = beta_selection["factors_to_absolute_beta"]
    for factor in [float(value) for value in specification.get("beta_factors", [0.0])]:
        beta = 0.0 if factor == 0.0 else float(absolute[format(factor, ".17g")])
        name = "native_P_beta0" if factor == 0.0 else f"native_P_factor_{_float_name(factor)}"
        methods.append(
            _Method(
                name,
                "P",
                beta=beta,
                beta_factor=factor,
                native_mode="P",
            )
        )
    if bool(specification.get("kernel_ablations", False)):
        methods.extend(
            (
                _Method(
                    "ablation_F_rescan_all_exact",
                    "ablation",
                    native_mode="F",
                    threshold_mode="rescan",
                    rank_strategy="all_boundary_exact",
                ),
                _Method(
                    "ablation_F_heap_all_exact",
                    "ablation",
                    native_mode="F",
                    threshold_mode="heap",
                    rank_strategy="all_boundary_exact",
                ),
                _Method(
                    "ablation_P_beta0_rescan_adaptive",
                    "ablation",
                    native_mode="P",
                    threshold_mode="rescan",
                    rank_strategy="adaptive",
                ),
            )
        )
    if len({method.name for method in methods}) != len(methods):
        raise BenchmarkConfigurationError("method names are not unique")
    return methods


def _keys(hits: Sequence[SearchHit]) -> list[list[int]]:
    return [[int(hit.logical_id), int(hit.version_id)] for hit in hits]


def _lightweight_faiss_merge(
    query: np.ndarray,
    *,
    prepared: _Prepared,
    candidates: CandidateSet,
    k: int,
    delta_overfetch: int,
) -> tuple[tuple[SearchHit, ...], int, int, int]:
    """Non-certified A-reference using Faiss/float distance order directly."""

    requested = min(len(prepared.delta_records), max(k, delta_overfetch))
    faiss.omp_set_num_threads(1)
    search_start = time.perf_counter_ns()
    if requested:
        squared, labels = prepared.delta_flat.index.search(query[None, :], requested)
        delta_rows = [
            (prepared.delta_records[int(label)], float(distance), "faiss_A_reference")
            for distance, label in zip(squared[0].tolist(), labels[0].tolist())
            if int(label) >= 0
        ]
    else:
        delta_rows = []
    search_ns = time.perf_counter_ns() - search_start

    merge_start = time.perf_counter_ns()
    if len(candidates.records):
        difference = candidates.vectors - query[None, :]
        base_squared = np.sum(difference * difference, axis=1, dtype=np.float32)
    else:
        base_squared = np.empty(0, dtype=np.float32)
    rows: list[tuple[VectorRecord, float, str, int]] = [
        (record, float(distance), "base_A_reference", position)
        for position, (record, distance) in enumerate(
            zip(candidates.records, base_squared.tolist())
        )
    ]
    rows.extend(
        (record, distance, source, len(rows) + offset)
        for offset, (record, distance, source) in enumerate(delta_rows)
    )
    chosen: dict[int, tuple[VectorRecord, float, str, int]] = {}
    for item in rows:
        record, _, _, position = item
        current = chosen.get(record.logical_id)
        if current is None or (record.begin_seq, record.version_id, position) >= (
            current[0].begin_seq,
            current[0].version_id,
            current[3],
        ):
            chosen[record.logical_id] = item
    ranked = sorted(
        chosen.values(),
        key=lambda item: (item[1], item[0].logical_id, item[0].version_id, item[3]),
    )[:k]
    hits = tuple(
        SearchHit(
            logical_id=record.logical_id,
            version_id=record.version_id,
            distance=math.sqrt(max(0.0, squared_distance)),
            source=source,
        )
        for record, squared_distance, source, _ in ranked
    )
    merge_ns = time.perf_counter_ns() - merge_start
    return hits, search_ns, merge_ns, requested


def _measure_api(
    method: _Method,
    *,
    query: np.ndarray,
    prepared: _Prepared,
    k: int,
    candidate_count: int,
    delta_overfetch: int,
) -> _ApiMeasurement:

    if method.role in {"A", "A-reference"}:
        if method.role == "A-reference":
            api_start = time.perf_counter_ns()
            api_candidates = prepared.engine.prepare_candidates(
                query, snapshot_id=0, k=k, candidate_count=candidate_count
            )
            api_hits, _, _, _ = _lightweight_faiss_merge(
                query,
                prepared=prepared,
                candidates=api_candidates,
                k=k,
                delta_overfetch=delta_overfetch,
            )
            api_ns = time.perf_counter_ns() - api_start
            return _ApiMeasurement(
                hits=api_hits,
                wall_ns=api_ns,
                candidate_hash=api_candidates.candidate_hash,
                native_calls=0,
                backend="faiss_squared_L2_order_noncertified",
            )
        api_start = time.perf_counter_ns()
        api_candidates = prepared.engine.prepare_candidates(
            query, snapshot_id=0, k=k, candidate_count=candidate_count
        )
        api_result = prepared.delta_flat.search_and_merge_candidates_performance(
            query,
            k=k,
            base_candidates=api_candidates,
            ann_candidate_count=delta_overfetch,
            source="faiss_A",
        )
        api_ns = time.perf_counter_ns() - api_start
        return _ApiMeasurement(
            hits=api_result.hits,
            wall_ns=api_ns,
            candidate_hash=api_candidates.candidate_hash,
            native_calls=0,
            backend="faiss.IndexFlatL2",
        )

    if method.native_mode is None:
        raise RuntimeError(f"method {method.name} has no executable implementation")
    before = native_call_counter()
    api_start = time.perf_counter_ns()
    if method.native_mode == "F":
        api_result = prepared.engine.search_native_full_scan(
            query,
            k=k,
            beta=method.beta,
            snapshot_id=0,
            candidate_count=candidate_count,
            threshold_mode=method.threshold_mode,
            rank_strategy=method.rank_strategy,
        )
    else:
        api_result = prepared.engine.search_native_pruned(
            query,
            k=k,
            beta=method.beta,
            snapshot_id=0,
            candidate_count=candidate_count,
            no_pruning=method.native_mode == "N",
            threshold_mode=method.threshold_mode,
            rank_strategy=method.rank_strategy,
        )
    api_ns = time.perf_counter_ns() - api_start
    calls = native_call_counter() - before
    if calls != 1:
        raise RuntimeError(f"expected one API native call for {method.name}, observed {calls}")
    details = api_result.receipt.certificate_details or {}
    backend = str(details.get("native_backend") or "")
    if not backend or backend.lower().startswith("python"):
        raise RuntimeError(f"{method.name} API did not report a compiled backend")
    return _ApiMeasurement(
        hits=api_result.hits,
        wall_ns=api_ns,
        candidate_hash=api_result.receipt.candidate_set_id_or_hash,
        native_calls=calls,
        backend=backend,
    )


def _measure_micro(
    method: _Method,
    *,
    query: np.ndarray,
    candidates: CandidateSet,
    native_query: object,
    prepared: _Prepared,
    k: int,
    delta_overfetch: int,
) -> _MicroMeasurement:
    if method.role == "A-reference":
        start = time.perf_counter_ns()
        hits, search_ns, merge_ns, requested = _lightweight_faiss_merge(
            query,
            prepared=prepared,
            candidates=candidates,
            k=k,
            delta_overfetch=delta_overfetch,
        )
        wall_ns = time.perf_counter_ns() - start
        return _MicroMeasurement(
            hits,
            wall_ns,
            0,
            "faiss_squared_L2_order_noncertified",
            None,
            {
                "delta_faiss_search_ns": search_ns,
                "lightweight_float_merge_ns": merge_ns,
            },
            requested,
        )
    if method.role == "A":
        start = time.perf_counter_ns()
        result = prepared.delta_flat.search_and_merge_candidates_performance(
            query,
            k=k,
            base_candidates=candidates,
            ann_candidate_count=delta_overfetch,
            source="faiss_A",
        )
        wall_ns = time.perf_counter_ns() - start
        return _MicroMeasurement(
            result.hits,
            wall_ns,
            0,
            "faiss.IndexFlatL2",
            None,
            {
                "delta_faiss_search_ns": int(result.search_ns),
                "shortlist_exact_merge_ns": int(result.merge_ns),
            },
            result.candidate_count,
            result.exact_rechecks,
        )
    if method.native_mode is None:
        raise RuntimeError(f"method {method.name} has no executable implementation")
    before = native_call_counter()
    micro_start = time.perf_counter_ns()
    micro_result = prepared.engine.search_native_prepared(
        native_query,
        k=k,
        beta=method.beta,
        mode=method.native_mode,
        threshold_mode=method.threshold_mode,
        rank_strategy=method.rank_strategy,
    )
    micro_ns = time.perf_counter_ns() - micro_start
    calls = native_call_counter() - before
    if calls != 1:
        raise RuntimeError(f"expected one micro native call for {method.name}, observed {calls}")
    receipt = micro_result.receipt.to_dict()
    details = receipt.get("certificate_details") or {}
    backend = str(details.get("native_backend") or "")
    if not backend or backend.lower().startswith("python"):
        raise RuntimeError(f"{method.name} did not report a compiled backend")
    return _MicroMeasurement(
        hits=micro_result.hits,
        wall_ns=micro_ns,
        native_calls=calls,
        backend=backend,
        receipt=receipt,
        components={
            str(name): int(value)
            for name, value in micro_result.receipt.component_timings.items()
        },
    )


def _combine_measurements(
    method: _Method, api: _ApiMeasurement, micro: _MicroMeasurement
) -> _Measured:
    if _keys(api.hits) != _keys(micro.hits):
        raise RuntimeError(f"{method.name} micro/API results differ")
    if api.backend != micro.backend:
        raise RuntimeError(f"{method.name} micro/API backend identity differs")
    calls = api.native_calls + micro.native_calls
    if method.native_mode is not None and calls != 2:
        raise RuntimeError(f"expected two total native calls for {method.name}, observed {calls}")
    return _Measured(
        hits=micro.hits,
        micro_ns=micro.wall_ns,
        api_wall_ns=api.wall_ns,
        api_candidate_hash=api.candidate_hash,
        native_calls=calls,
        backend=micro.backend,
        receipt=micro.receipt,
        components=micro.components,
        candidate_count=micro.candidate_count,
        exact_rechecks=micro.exact_rechecks,
    )


def _exact_gap(
    query: np.ndarray,
    measured: Sequence[SearchHit],
    reference: Sequence[SearchHit],
    records_by_key: Mapping[tuple[int, int], VectorRecord],
) -> tuple[float | None, float | None, bool | None]:
    if not measured or not reference:
        return None, None, None
    proposed_record = records_by_key[measured[-1].key]
    reference_record = records_by_key[reference[-1].key]
    proposed = oracle_exact_squared_l2(query, proposed_record.vector)
    expected = oracle_exact_squared_l2(query, reference_record.vector)
    bracket = observed_gap_bracket(proposed, expected)
    return max(0.0, float(bracket.lower)), max(0.0, float(bracket.upper)), proposed >= expected


def _native_contract(
    method: _Method,
    measured: _Measured,
    reference_hits: Sequence[SearchHit],
    *,
    query: np.ndarray,
    records_by_key: Mapping[tuple[int, int], VectorRecord],
    population: int,
    k: int,
) -> dict[str, Any]:
    keys = _keys(measured.hits)
    reference_keys = _keys(reference_hits)
    ordered_match = keys == reference_keys
    gap_lower, gap_upper, gap_nonnegative = _exact_gap(
        query, measured.hits, reference_hits, records_by_key
    )
    certified = None if measured.receipt is None else measured.receipt.get("certified_beta")
    receipt_beta = None if measured.receipt is None else measured.receipt.get("requested_beta")
    receipt_candidate = (
        None
        if measured.receipt is None
        else measured.receipt.get("candidate_set_id_or_hash")
    )
    if population < k:
        valid = (
            len(keys) == population
            and ordered_match
            and certified is None
            and measured.receipt is not None
            and measured.receipt.get("tau_returned") is None
        )
    elif method.beta == 0.0:
        valid = ordered_match and certified is not None and float(certified) == 0.0
    else:
        valid = (
            gap_nonnegative is True
            and gap_upper is not None
            and certified is not None
            and 0.0 <= gap_upper <= float(certified) <= method.beta
        )
    return {
        "contract_valid": bool(valid),
        "contract_violation": not bool(valid),
        "same_c_order_match": ordered_match,
        "observed_beta_lower_l2": gap_lower,
        "observed_beta_upper_l2": gap_upper,
        "observed_gap_nonnegative": gap_nonnegative,
        "certified_beta_l2": certified,
        "receipt_requested_beta_l2": receipt_beta,
        "receipt_candidate_set_id_or_hash": receipt_candidate,
    }


def _row(
    *,
    run_id: str,
    config_hash: str,
    implementation_hash: str,
    evidence_role: str,
    phase: str,
    session_id: str,
    dataset_hash: str,
    split_id: str,
    split: DatasetSplit,
    specification: Mapping[str, Any],
    query_id: int,
    query_position: int,
    repetition: int,
    api_order: Sequence[str],
    micro_order: Sequence[str],
    method: _Method,
    measured: _Measured,
    reference_hits: Sequence[SearchHit],
    candidates: CandidateSet,
    base_prepare_wall_ns: int,
    native_query_prepare_wall_ns: int,
    prepared: _Prepared,
    query: np.ndarray,
) -> dict[str, Any]:
    result_keys = _keys(measured.hits)
    reference_keys = _keys(reference_hits)
    delta_influence = any(
        hit.logical_id in prepared.delta_logical_ids for hit in reference_hits
    )
    is_native = method.native_mode is not None
    composed = base_prepare_wall_ns + measured.micro_ns
    if is_native:
        composed += native_query_prepare_wall_ns
    population = len({record.logical_id for record in candidates.records + prepared.delta_records})
    contract = (
        _native_contract(
            method,
            measured,
            reference_hits,
            query=query,
            records_by_key=prepared.records_by_key,
            population=population,
            k=int(specification["k"]),
        )
        if is_native
        else {}
    )
    a_match = (
        result_keys == reference_keys
        if method.role in {"A", "A-reference"}
        else None
    )
    candidate_hash_match = measured.api_candidate_hash == candidates.candidate_hash
    if not candidate_hash_match:
        raise RuntimeError(f"{method.name} regenerated a different C in API timing")
    receipt_details = (
        {}
        if measured.receipt is None
        else dict(measured.receipt.get("certificate_details") or {})
    )
    native_binary_sha256 = (
        receipt_details.get("native_build", {}).get("shared_object_sha256")
        if isinstance(receipt_details.get("native_build"), dict)
        else None
    )
    return {
        "schema_version": NATIVE_BENCHMARK_SCHEMA_VERSION,
        "study_id": "issue-3-native-recheck",
        "run_id": run_id,
        "config_hash": config_hash,
        "implementation_tree_sha256": implementation_hash,
        "evidence_role": evidence_role,
        "phase": phase,
        "session_id": session_id,
        "dataset_hash": dataset_hash,
        "split_id": split_id,
        "dataset_type": str(specification["dataset"]["type"]),
        "dataset_name": split.name,
        "experiment_id": str(specification["id"]),
        "axis": specification.get("axis"),
        "query_id": int(query_id),
        "query_position": int(query_position),
        "repetition": int(repetition),
        "method": method.name,
        "method_role": method.role,
        "method_order": list(api_order),
        "method_order_position": list(api_order).index(method.name),
        "api_method_order": list(api_order),
        "api_method_order_position": list(api_order).index(method.name),
        "micro_method_order": list(micro_order),
        "micro_method_order_position": list(micro_order).index(method.name),
        "requested_beta_l2": method.beta,
        "beta_factor": method.beta_factor,
        "threshold_mode": method.threshold_mode if is_native else None,
        "rank_strategy": method.rank_strategy if is_native else None,
        "k": int(specification["k"]),
        "candidate_count_requested": int(specification["candidate_count"]),
        "candidate_count_returned": len(candidates.records),
        "candidate_set_id_or_hash": candidates.candidate_hash,
        "same_frozen_candidate_object_used": True,
        "api_candidate_hash_match": candidate_hash_match,
        "base_prepare_wall_ns": int(base_prepare_wall_ns),
        "native_query_prepare_wall_ns": (
            int(native_query_prepare_wall_ns) if is_native else 0
        ),
        "micro_latency_ns": int(measured.micro_ns),
        "composed_e2e_latency_ns": int(composed),
        "api_wall_latency_ns": int(measured.api_wall_ns),
        "execution_backend": measured.backend,
        "native_binary_sha256": native_binary_sha256 if is_native else None,
        "native_call_count": int(measured.native_calls),
        "native_call_index": receipt_details.get("native_call_index") if is_native else None,
        "python_fallback_used": False if is_native else None,
        "result_keys": result_keys,
        "reference_keys": reference_keys,
        "delta_influence": delta_influence,
        "baseline_quality_match": a_match,
        "baseline_validation_failure": (not a_match) if a_match is not None else False,
        "comparison_valid": a_match if a_match is not None else True,
        "n_base": len(split.base),
        "n_delta": len(split.delta),
        "delta_count": len(split.delta),
        "n_groups": int(specification["n_groups"]),
        "raw_pending_count": len(prepared.raw_records),
        "raw_ratio": len(prepared.raw_records) / len(split.delta) if len(split.delta) else 0.0,
        "component_timings_ns": measured.components,
        "exact_boundary_rechecks": (
            measured.receipt.get("exact_boundary_rechecks")
            if measured.receipt is not None
            else measured.exact_rechecks
        ),
        "faiss_delta_candidate_count": measured.candidate_count,
        "receipt": measured.receipt,
        **contract,
    }


def _oracle_row(
    *,
    common: Mapping[str, Any],
    query_id: int,
    query_position: int,
    reference_hits: Sequence[SearchHit],
    candidates: CandidateSet,
    delta_influence: bool,
    fraction_checked: bool,
    fraction_match: bool | None,
) -> dict[str, Any]:
    return {
        **dict(common),
        "query_id": int(query_id),
        "query_position": int(query_position),
        "repetition": -1,
        "method": "old_O",
        "method_role": "O",
        "requested_beta_l2": 0.0,
        "micro_latency_ns": None,
        "composed_e2e_latency_ns": None,
        "api_wall_latency_ns": None,
        "execution_backend": "python_reference_not_timed",
        "native_call_count": 0,
        "python_fallback_used": None,
        "candidate_set_id_or_hash": candidates.candidate_hash,
        "result_keys": _keys(reference_hits),
        "reference_keys": _keys(reference_hits),
        "same_c_order_match": True,
        "contract_valid": True,
        "contract_violation": False,
        "delta_influence": delta_influence,
        "fraction_oracle_checked": fraction_checked,
        "fraction_oracle_order_match": fraction_match,
    }


def _stable_order(
    methods: Sequence[_Method],
    *,
    seed: int,
    experiment_id: str,
    query_id: int,
    repetition: int,
    scope: str,
) -> list[_Method]:
    digest = object_sha256(
        {
            "purpose": "issue3_method_order_v2_independent_scopes",
            "scope": scope,
            "seed": int(seed),
            "experiment_id": experiment_id,
            "query_id": int(query_id),
            "repetition": int(repetition),
        }
    )
    ordered = list(methods)
    random.Random(int(digest[:16], 16)).shuffle(ordered)
    return ordered


def _run_query(
    *,
    run_id: str,
    config_hash: str,
    implementation_hash: str,
    config: Mapping[str, Any],
    measurement_session_id: str,
    dataset_hash: str,
    split_id: str,
    split: DatasetSplit,
    specification: Mapping[str, Any],
    prepared: _Prepared,
    methods: Sequence[_Method],
    query: np.ndarray,
    query_id: int,
    query_position: int,
    fraction_oracle_positions: set[int],
    full_truth_cache: dict[str, tuple[SearchHit, ...]],
) -> list[dict[str, Any]]:
    k = int(specification["k"])
    candidate_count = int(specification["candidate_count"])
    delta_overfetch = int(specification.get("delta_flat_overfetch", max(k, 64)))
    base_start = time.perf_counter_ns()
    candidates = prepared.engine.prepare_candidates(
        query, snapshot_id=0, k=k, candidate_count=candidate_count
    )
    base_prepare_wall_ns = time.perf_counter_ns() - base_start
    native_prepare_start = time.perf_counter_ns()
    native_query = prepared.engine.prepare_native_query(
        query,
        snapshot_id=0,
        k=k,
        candidate_count=candidate_count,
        candidate_set=candidates,
    )
    native_query_prepare_wall_ns = time.perf_counter_ns() - native_prepare_start

    # Collect the two independently randomized timing passes before O or the
    # full-visible truth touches a different representation of the whole Delta.
    trial_results: list[tuple[int, list[str], list[str], _Method, _Measured]] = []
    repetitions = int(specification.get("repetitions", config["repetitions"]))
    for repetition in range(repetitions):
        api_ordered = _stable_order(
            methods,
            seed=int(config.get("method_order_seed", 0)),
            experiment_id=str(specification["id"]),
            query_id=query_id,
            repetition=repetition,
            scope="api_wall",
        )
        micro_ordered = _stable_order(
            methods,
            seed=int(config.get("method_order_seed", 0)),
            experiment_id=str(specification["id"]),
            query_id=query_id,
            repetition=repetition,
            scope="micro",
        )
        api_results = {
            method.name: _measure_api(
                method,
                query=query,
                prepared=prepared,
                k=k,
                candidate_count=candidate_count,
                delta_overfetch=delta_overfetch,
            )
            for method in api_ordered
        }
        micro_results = {
            method.name: _measure_micro(
                method,
                query=query,
                candidates=candidates,
                native_query=native_query,
                prepared=prepared,
                k=k,
                delta_overfetch=delta_overfetch,
            )
            for method in micro_ordered
        }
        api_names = [method.name for method in api_ordered]
        micro_names = [method.name for method in micro_ordered]
        for method in methods:
            trial_results.append(
                (
                    repetition,
                    api_names,
                    micro_names,
                    method,
                    _combine_measurements(
                        method, api_results[method.name], micro_results[method.name]
                    ),
                )
            )

    # O is generated only after timing, from the already frozen C.  It remains
    # the compatibility reference used to validate every saved measured row.
    reference = prepared.engine.search_full_scan(
        query,
        k=k,
        beta=0.0,
        snapshot_id=0,
        candidate_count=candidate_count,
        candidate_set=candidates,
    )
    fraction_checked = query_position in fraction_oracle_positions
    fraction_match: bool | None = None
    if fraction_checked:
        oracle = reference_topk(
            query,
            candidates,
            raw_records=prepared.raw_records,
            grouped_records=prepared.grouped_records,
            snapshot_id=0,
            k=k,
        )
        fraction_match = [list(key) for key in oracle.keys] == _keys(reference.hits)
        if not fraction_match:
            raise RuntimeError("old same-C reference differs from independent Fraction oracle")

    delta_influence = any(
        hit.logical_id in prepared.delta_logical_ids for hit in reference.hits
    )
    common = {
        "schema_version": NATIVE_BENCHMARK_SCHEMA_VERSION,
        "study_id": "issue-3-native-recheck",
        "run_id": run_id,
        "config_hash": config_hash,
        "implementation_tree_sha256": implementation_hash,
        "evidence_role": str(config.get("evidence_role", "validation")),
        "phase": str(config["phase"]),
        "session_id": measurement_session_id,
        "dataset_hash": dataset_hash,
        "split_id": split_id,
        "dataset_type": str(specification["dataset"]["type"]),
        "dataset_name": split.name,
        "experiment_id": str(specification["id"]),
        "axis": specification.get("axis"),
        "n_base": len(split.base),
        "n_delta": len(split.delta),
        "delta_count": len(split.delta),
        "n_groups": int(specification["n_groups"]),
        "k": k,
        "candidate_count_requested": candidate_count,
        "raw_pending_count": len(prepared.raw_records),
        "raw_ratio": len(prepared.raw_records) / len(split.delta) if len(split.delta) else 0.0,
    }
    rows = [
        _oracle_row(
            common=common,
            query_id=query_id,
            query_position=query_position,
            reference_hits=reference.hits,
            candidates=candidates,
            delta_influence=delta_influence,
            fraction_checked=fraction_checked,
            fraction_match=fraction_match,
        )
    ]
    for repetition, api_names, micro_names, method, measured in trial_results:
        rows.append(
                _row(
                    run_id=run_id,
                    config_hash=config_hash,
                    implementation_hash=implementation_hash,
                    evidence_role=str(config.get("evidence_role", "validation")),
                    phase=str(config["phase"]),
                    session_id=measurement_session_id,
                    dataset_hash=dataset_hash,
                    split_id=split_id,
                    split=split,
                    specification=specification,
                    query_id=query_id,
                    query_position=query_position,
                    repetition=repetition,
                    api_order=api_names,
                    micro_order=micro_names,
                    method=method,
                    measured=measured,
                    reference_hits=reference.hits,
                    candidates=candidates,
                    base_prepare_wall_ns=base_prepare_wall_ns,
                    native_query_prepare_wall_ns=native_query_prepare_wall_ns,
                    prepared=prepared,
                    query=query,
                )
            )
    truth_key = object_sha256(
        {
            "purpose": "full_visible_exact_truth_v1",
            "dataset_hash": dataset_hash,
            "query_id": int(query_id),
            "k": k,
        }
    )
    exact_hits = full_truth_cache.get(truth_key)
    truth_cache_hit = exact_hits is not None
    if exact_hits is None:
        all_records = prepared.base_records + prepared.delta_records
        bounds = distance_intervals(query, prepared.full_vectors)
        exact_hits = stable_topk(
            query,
            all_records,
            prepared.full_vectors,
            ["full_visible_exact"] * len(all_records),
            k,
            bounds,
        ).hits
        full_truth_cache[truth_key] = exact_hits
    exact_keys = {hit.key for hit in exact_hits}
    exact_delta_keys = {
        hit.key for hit in exact_hits if hit.logical_id in prepared.delta_logical_ids
    }
    for row in rows:
        result_key_set = {tuple(value) for value in row["result_keys"]}
        row["full_visible_exact_keys"] = _keys(exact_hits)
        row["full_visible_exact_recall"] = (
            len(result_key_set.intersection(exact_keys)) / len(exact_keys)
            if exact_keys
            else None
        )
        row["full_visible_exact_delta_neighbors"] = len(exact_delta_keys)
        row["delta_neighbor_capture"] = (
            len(result_key_set.intersection(exact_delta_keys)) / len(exact_delta_keys)
            if exact_delta_keys
            else None
        )
        row["full_truth_cache_hit"] = truth_cache_hit
        row["full_truth_cache_key"] = truth_key
    return rows


def _warmup(
    split: DatasetSplit,
    specification: Mapping[str, Any],
    prepared: _Prepared,
    methods: Sequence[_Method],
    count: int,
) -> dict[str, Any]:
    count = min(count, len(split.validation_queries))
    calls_before = native_call_counter()
    for query in split.validation_queries[:count]:
        k = int(specification["k"])
        candidate_count = int(specification["candidate_count"])
        candidates = prepared.engine.prepare_candidates(
            query, snapshot_id=0, k=k, candidate_count=candidate_count
        )
        native_query = prepared.engine.prepare_native_query(
            query,
            snapshot_id=0,
            k=k,
            candidate_count=candidate_count,
            candidate_set=candidates,
        )
        api_results = {}
        for method in methods:
            api_results[method.name] = _measure_api(
                method,
                query=query,
                prepared=prepared,
                k=k,
                candidate_count=candidate_count,
                delta_overfetch=int(
                    specification.get("delta_flat_overfetch", max(k, 64))
                ),
            )
        for method in reversed(methods):
            micro = _measure_micro(
                method,
                query=query,
                candidates=candidates,
                native_query=native_query,
                prepared=prepared,
                k=k,
                delta_overfetch=int(
                    specification.get("delta_flat_overfetch", max(k, 64))
                ),
            )
            _combine_measurements(method, api_results[method.name], micro)
    return {
        "validation_queries": count,
        "methods_per_query": len(methods),
        "recorded_in_raw": False,
        "native_calls": native_call_counter() - calls_before,
        "test_queries_used": 0,
    }


def _base_cache_ablation(
    split: DatasetSplit,
    specification: Mapping[str, Any],
    prepared: _Prepared,
    count: int,
) -> dict[str, Any]:
    count = min(count, len(split.validation_queries))
    k = int(specification["k"])
    candidate_count = int(specification["candidate_count"])
    rows: list[dict[str, Any]] = []
    for position, (query_id, query) in enumerate(
        zip(split.validation_query_ids[:count].tolist(), split.validation_queries[:count])
    ):
        observations: dict[str, CandidateSet] = {}
        for mode in ("cached", "legacy_rebuild"):
            start = time.perf_counter_ns()
            result = prepared.base.prepare_candidates(
                query,
                snapshot_id=0,
                k=k,
                candidate_count=candidate_count,
                use_cached_view=mode == "cached",
            )
            wall = time.perf_counter_ns() - start
            observations[mode] = result
            rows.append(
                {
                    "query_id": int(query_id),
                    "query_position": position,
                    "mode": mode,
                    "wall_ns": wall,
                    "candidate_hash": result.candidate_hash,
                    "candidate_keys": [list(record.key) for record in result.records],
                }
            )
        if observations["cached"].candidate_hash != observations["legacy_rebuild"].candidate_hash:
            raise RuntimeError("Base visibility cache ablation changed frozen C")
    medians = {
        mode: (
            float(np.median([row["wall_ns"] for row in rows if row["mode"] == mode]))
            if rows
            else None
        )
        for mode in ("cached", "legacy_rebuild")
    }
    return {
        "scope": "validation_queries_only_not_in_performance_rows",
        "query_count": count,
        "medians_ns": medians,
        "cache_stats": prepared.base.visible_view_cache_stats(),
        "rows": rows,
    }


def _native_lb_audit(
    split: DatasetSplit,
    specification: Mapping[str, Any],
    prepared: _Prepared,
    count: int,
) -> dict[str, Any]:
    """Audit every native group decision against its exact true minimum.

    Bulk intervals identify the only members that can realize the minimum;
    the independent Fraction oracle then resolves those candidates exactly.
    This keeps the audit exhaustive without putting per-vector Fraction work
    in either the search kernel or the measured path.
    """

    count = min(count, len(split.validation_queries))
    checks: list[dict[str, Any]] = []
    directory = prepared.engine.delta.grouped
    groups = {} if directory is None else {group.group_id: group for group in directory.groups}
    for position, (query_id, query) in enumerate(
        zip(split.validation_query_ids[:count].tolist(), split.validation_queries[:count])
    ):
        k = int(specification["k"])
        candidate_count = int(specification["candidate_count"])
        candidates = prepared.engine.prepare_candidates(
            query, snapshot_id=0, k=k, candidate_count=candidate_count
        )
        native_query = prepared.engine.prepare_native_query(
            query,
            snapshot_id=0,
            k=k,
            candidate_count=candidate_count,
            candidate_set=candidates,
        )
        result = prepared.engine.search_native_prepared(
            native_query, k=k, beta=0.0, mode="P", audit=True
        )
        audit_rows = tuple(result.receipt.audit or ())
        expected_group_ids = {
            group_id
            for group_id, group in groups.items()
            if any(member.visible_at(0) for member in group.members)
        }
        observed_group_ids = {int(row["group_id"]) for row in audit_rows}
        if observed_group_ids != expected_group_ids:
            raise RuntimeError(
                "native LB audit did not emit exactly one decision for every visible group"
            )
        for row in audit_rows:
            group_id = int(row["group_id"])
            group = groups[group_id]
            visible, visible_vectors = group.visible(0)
            if not visible:
                continue
            intervals = distance_intervals(query, visible_vectors)
            minimum_upper = float(intervals.upper.min())
            possible_minimum_indices = np.flatnonzero(intervals.lower <= minimum_upper)
            if not len(possible_minimum_indices):
                raise AssertionError("group minimum interval candidate set is empty")
            exact_candidates = [
                oracle_exact_squared_l2(query, visible[int(index)].vector)
                for index in possible_minimum_indices.tolist()
            ]
            exact_min_squared = min(exact_candidates)
            exact_min_lower, exact_min_upper = oracle_sqrt_bracket(exact_min_squared)
            lb = float(row["lb_lower"])
            lb_sound = lb >= 0.0 and Fraction.from_float(lb) ** 2 <= exact_min_squared
            lb_fraction = Fraction(str(row["lb_lower_fraction"]))
            threshold_text = row.get("threshold_tau_minus_beta_fraction")
            skip_predicate = (
                threshold_text is not None
                and lb_fraction > Fraction(str(threshold_text))
            )
            action = str(row["action"])
            decision_matches = (action == "skip") == skip_predicate
            checks.append(
                {
                    "query_id": int(query_id),
                    "query_position": position,
                    "group_id": group_id,
                    "visible_members": len(visible),
                    "action": action,
                    "radius_upper": float(row["radius_upper"]),
                    "radius_upper_fraction": row["radius_upper_fraction"],
                    "center_distance_lower": float(row["center_distance_lower"]),
                    "center_distance_upper": float(row["center_distance_upper"]),
                    "lb_lower": lb,
                    "lb_lower_fraction": row["lb_lower_fraction"],
                    "tau_upper_at_decision": row.get("tau_upper_at_decision"),
                    "tau_upper_at_decision_fraction": row.get(
                        "tau_upper_at_decision_fraction"
                    ),
                    "threshold_tau_minus_beta": row.get(
                        "threshold_tau_minus_beta"
                    ),
                    "threshold_tau_minus_beta_fraction": threshold_text,
                    "exact_min_squared_fraction": (
                        f"{exact_min_squared.numerator}/{exact_min_squared.denominator}"
                    ),
                    "exact_min_l2_lower": exact_min_lower,
                    "exact_min_l2_upper": exact_min_upper,
                    "interval_min_candidates_exactly_rechecked": len(
                        possible_minimum_indices
                    ),
                    "lower_bound_sound": lb_sound,
                    "strict_skip_predicate": skip_predicate,
                    "decision_matches_strict_predicate": decision_matches,
                }
            )
    failures = [
        row
        for row in checks
        if not row["lower_bound_sound"]
        or not row["decision_matches_strict_predicate"]
    ]
    return {
        "scope": "validation_queries_only_outside_timing",
        "query_count": count,
        "groups_checked": len(checks),
        "skipped_groups_checked": sum(row["action"] == "skip" for row in checks),
        "scanned_groups_checked": sum(row["action"] == "scan" for row in checks),
        "passed": not failures,
        "failures": failures[:100],
        "rows": checks,
    }


def _verify_correctness_prerequisite(
    config: Mapping[str, Any], implementation_hash: str
) -> dict[str, Any] | None:
    path_value = config.get("required_correctness_evidence")
    if path_value is None:
        if str(config["phase"]) in {"validation", "final"}:
            raise BenchmarkConfigurationError(
                "validation/final requires required_correctness_evidence"
            )
        return None
    path = Path(str(path_value))
    evidence = _read_object(path)
    reasons: list[str] = []
    if evidence.get("status") != "passed":
        reasons.append("status_not_passed")
    if evidence.get("pytest_exit_code") != 0:
        reasons.append("pytest_exit_code_not_zero")
    if int(evidence.get("fixed_seed_cases", 0)) < 10_000:
        reasons.append("fewer_than_10000_fixed_seed_cases")
    if evidence.get("implementation_tree_sha256") != implementation_hash:
        reasons.append("implementation_tree_hash_mismatch")
    build = dict(native_build_info())
    if evidence.get("native_shared_object_sha256") != build.get("shared_object_sha256"):
        reasons.append("native_binary_hash_mismatch")
    junit_value = evidence.get("junit_path")
    junit_path = None if junit_value is None else Path(str(junit_value))
    if junit_path is not None and not junit_path.is_absolute():
        junit_path = REPOSITORY_ROOT / junit_path
    if (
        junit_path is None
        or not junit_path.is_file()
        or evidence.get("junit_sha256") != file_sha256(junit_path)
    ):
        reasons.append("junit_hash_mismatch_or_missing")
    if evidence.get("test_tree_sha256") != _native_test_tree_sha256():
        reasons.append("test_tree_hash_mismatch")
    if reasons:
        raise RuntimeError(
            "native correctness prerequisite rejected: " + ",".join(reasons)
        )
    return {
        "path": str(path),
        "sha256": file_sha256(path),
        "status": evidence.get("status"),
        "fixed_seed_cases": int(evidence["fixed_seed_cases"]),
        "implementation_tree_sha256": evidence["implementation_tree_sha256"],
        "native_shared_object_sha256": evidence["native_shared_object_sha256"],
        "junit_sha256": evidence["junit_sha256"],
        "test_tree_sha256": evidence["test_tree_sha256"],
    }


def _native_test_tree_sha256() -> str:
    return native_test_tree_sha256(REPOSITORY_ROOT)


def _verify_holdout_registration(config: Mapping[str, Any]) -> dict[str, Any] | None:
    """Fail closed before loading a query vector reserved by the registration."""

    phase = str(config["phase"])
    if phase not in {"validation", "final"}:
        return None
    policy = config.get("holdout_policy")
    if not isinstance(policy, dict) or not policy.get("manifest"):
        raise BenchmarkConfigurationError("validation/final requires holdout_policy.manifest")
    path = Path(str(policy["manifest"]))
    manifest = _read_object(path)
    if manifest.get("study_id") != "issue-3-native-recheck":
        raise BenchmarkConfigurationError("holdout manifest study identity mismatch")
    selection = manifest.get("selection_integrity")
    if not isinstance(selection, dict):
        raise BenchmarkConfigurationError("holdout manifest has no selection integrity")
    source_hash = config.get("source_config_object_hash")
    source_file_hash = config.get("source_config_file_sha256")
    if phase == "validation" and source_hash != selection.get(
        "validation_source_config_object_sha256"
    ):
        raise BenchmarkConfigurationError(
            "validation source config differs from the pre-registered object hash"
        )
    if phase == "validation" and source_file_hash != selection.get(
        "validation_source_config_file_sha256"
    ):
        raise BenchmarkConfigurationError(
            "validation source config bytes differ from the pre-registered file hash"
        )
    configured_final = policy.get("pre_registered_final_policy")
    registered_final = manifest.get("pre_registered_final_policy")
    if not isinstance(configured_final, dict) or not isinstance(registered_final, dict):
        raise BenchmarkConfigurationError("final policy was not pre-registered")
    identity = {
        "path": registered_final.get("path"),
        "file_sha256": registered_final.get("file_sha256"),
        "object_sha256": registered_final.get("object_sha256"),
    }
    if configured_final != identity:
        raise BenchmarkConfigurationError(
            "configured final policy identity differs from holdout registration"
        )
    try:
        validate_issue_fixed_decision_values(
            registered_final.get("issue_fixed_decision_values")
        )
    except FinalPolicyError as error:
        raise BenchmarkConfigurationError(str(error)) from error
    if (
        selection.get("final_policy_path") != identity["path"]
        or selection.get("final_policy_file_sha256") != identity["file_sha256"]
        or selection.get("final_policy_object_sha256") != identity["object_sha256"]
    ):
        raise BenchmarkConfigurationError("holdout final-policy registration is inconsistent")
    final_policy_path = Path(str(identity["path"]))
    if not final_policy_path.is_absolute():
        final_policy_path = REPOSITORY_ROOT / final_policy_path
    current_final_policy = _read_object(final_policy_path)
    try:
        validate_final_policy(current_final_policy)
    except FinalPolicyError as error:
        raise BenchmarkConfigurationError(str(error)) from error
    if (
        file_sha256(final_policy_path) != identity["file_sha256"]
        or object_sha256(current_final_policy) != identity["object_sha256"]
    ):
        raise BenchmarkConfigurationError(
            "current final policy differs from its pre-registered hashes"
        )
    sift = manifest.get("sift")
    if not isinstance(sift, dict):
        raise BenchmarkConfigurationError("holdout manifest has no SIFT registration")
    final_registration = sift.get("pre_registered_fresh_final_holdout")
    if not isinstance(final_registration, dict):
        raise BenchmarkConfigurationError("holdout manifest has no final interval")
    final_interval = final_registration.get("interval")
    if not isinstance(final_interval, list) or len(final_interval) != 2:
        raise BenchmarkConfigurationError("malformed SIFT final holdout interval")
    final_start, final_stop = (int(final_interval[0]), int(final_interval[1]))
    checked: list[dict[str, Any]] = []
    for experiment in config["experiments"]:
        dataset = experiment["dataset"]
        if dataset.get("type") != "texmex" or dataset.get("name") != "sift":
            continue
        validation_start = int(dataset.get("validation_query_offset", 0))
        validation_stop = validation_start + int(dataset["n_validation"])
        test_start = int(dataset.get("test_query_offset", validation_stop))
        test_stop = test_start + int(dataset["n_test"])
        intervals = (("validation", validation_start, validation_stop), ("test", test_start, test_stop))
        if phase == "validation" and any(stop > final_start for _, _, stop in intervals):
            raise BenchmarkConfigurationError(
                f"{experiment['id']}: validation would load the registered SIFT holdout"
            )
        if phase == "final" and not (
            final_start <= test_start < test_stop <= final_stop
        ):
            raise BenchmarkConfigurationError(
                f"{experiment['id']}: final SIFT test range is outside the registered holdout"
            )
        checked.append(
            {
                "experiment_id": str(experiment["id"]),
                "validation_interval": [validation_start, validation_stop],
                "test_interval": [test_start, test_stop],
            }
        )
    if not checked:
        raise BenchmarkConfigurationError(f"{phase} config has no registered SIFT experiment")
    return {
        "path": str(path),
        "sha256": file_sha256(path),
        "source_config_object_hash": source_hash,
        "source_config_file_sha256": source_file_hash,
        "final_policy_path": str(identity["path"]),
        "final_policy_file_sha256": str(identity["file_sha256"]),
        "final_policy_object_sha256": str(identity["object_sha256"]),
        "issue_fixed_decision_values": issue_fixed_decision_values(),
        "phase": phase,
        "registered_final_interval": [final_start, final_stop],
        "checked_sift_intervals": checked,
    }


def _block_count(config: Mapping[str, Any]) -> int:
    block_size = int(config["query_block_size"])
    return sum(
        math.ceil(int(row["dataset"]["n_test"]) / block_size)
        for row in config["experiments"]
    )


def _verify_raw_block(
    path: Path,
    entry: Mapping[str, Any],
    *,
    run_id: str,
    config_hash: str,
    implementation_hash: str,
) -> None:
    if not path.is_file() or file_sha256(path) != entry.get("sha256"):
        raise RuntimeError(f"raw block checksum mismatch: {path}")
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise RuntimeError(f"non-object raw row in {path}")
                rows.append(value)
    if len(rows) != int(entry.get("rows", -1)):
        raise RuntimeError(f"raw row count mismatch: {path}")
    for row in rows:
        if (
            row.get("run_id") != run_id
            or row.get("config_hash") != config_hash
            or row.get("implementation_tree_sha256") != implementation_hash
        ):
            raise RuntimeError(f"raw identity mismatch: {path}")


def _verify_native_lb_audit_file(
    path: Path,
    *,
    experiment_id: str,
    dataset_hash: str,
    split_id: str,
    expected_query_count: int,
) -> None:
    audit = _read_object(path)
    if (
        audit.get("experiment_id") != experiment_id
        or audit.get("dataset_hash") != dataset_hash
        or audit.get("split_id") != split_id
        or int(audit.get("query_count", -1)) != expected_query_count
        or audit.get("passed") is not True
        or audit.get("failures") != []
    ):
        raise RuntimeError(
            f"native LB audit identity/query-count/pass mismatch: {path}"
        )


def _ancillary_inventory(run_dir: Path) -> list[dict[str, Any]]:
    paths = [run_dir / "effective_config.json", run_dir / "build_manifest.json"]
    for directory in ("splits", "validation", "audits", "failures"):
        root = run_dir / directory
        if root.is_dir():
            paths.extend(path for path in root.rglob("*") if path.is_file())
    inventory = []
    for path in sorted(set(paths), key=lambda item: item.relative_to(run_dir).as_posix()):
        if not path.is_file():
            raise RuntimeError(f"completion ancillary artifact is missing: {path}")
        inventory.append(
            {
                "path": path.relative_to(run_dir).as_posix(),
                "sha256": file_sha256(path),
                "bytes": path.stat().st_size,
            }
        )
    return inventory


def run_native_benchmark(
    config: Mapping[str, Any] | str | Path,
    *,
    resume: bool = False,
    run_id: str | None = None,
    stop_after_blocks: int | None = None,
) -> RunOutcome:
    """Run/resume native recheck measurements and publish an immutable marker."""

    effective = _materialize_defaults(
        load_json_config(config) if isinstance(config, (str, Path)) else config
    )
    _validate_config(effective)
    if stop_after_blocks is not None and stop_after_blocks <= 0:
        raise BenchmarkConfigurationError("stop_after_blocks must be positive")
    for variable in THREAD_VARIABLES:
        os.environ[variable] = "1"
    faiss.omp_set_num_threads(1)
    require_native()
    implementation_hash = implementation_tree_sha256(REPOSITORY_ROOT)
    correctness = _verify_correctness_prerequisite(effective, implementation_hash)
    holdout_registration = _verify_holdout_registration(effective)
    config_hash = object_sha256(effective)
    run_dir, selected_id = _select_run_dir(
        effective,
        config_hash,
        implementation_hash,
        resume=resume,
        run_id=run_id,
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    for directory in ("raw", "splits", "validation", "audits", "failures"):
        (run_dir / directory).mkdir(exist_ok=True)
    effective_path = run_dir / "effective_config.json"
    manifest_path = run_dir / "run_manifest.json"
    checkpoint_path = run_dir / "checkpoint.json"
    build_path = run_dir / "build_manifest.json"
    if effective_path.exists():
        if object_sha256(_read_object(effective_path)) != config_hash:
            raise RuntimeError("effective config changed on resume")
    else:
        atomic_write_json(effective_path, effective)

    total_blocks = _block_count(effective)
    if manifest_path.exists():
        manifest = _read_object(manifest_path)
        if manifest.get("config_hash") != config_hash:
            raise RuntimeError("run manifest config hash mismatch")
        if manifest.get("implementation_tree_sha256") != implementation_hash:
            raise RuntimeError("implementation changed; refusing mixed-code resume")
        if manifest.get("holdout_registration") != holdout_registration:
            raise RuntimeError("holdout registration changed; refusing resume")
        if _has_disqualifying_failures(manifest):
            raise RuntimeError("correctness-failed run cannot be resumed as performance evidence")
        events = list(manifest.get("resume_events", []))
        segment_index = len(manifest.get("process_segments", []))
        measurement_session_id = (
            f"{effective.get('session_id', 'session-0')}:process-{segment_index:04d}"
        )
        segment = {
            "measurement_session_id": measurement_session_id,
            "started_at_utc": _utc_now(),
            "environment": collect_environment(1),
        }
        events.append({"resumed_at_utc": segment["started_at_utc"], **segment})
        manifest["resume_events"] = events
        manifest["process_segments"] = [*manifest.get("process_segments", []), segment]
        manifest["status"] = "running_resumed"
    else:
        measurement_session_id = (
            f"{effective.get('session_id', 'session-0')}:process-0000"
        )
        first_segment = {
            "measurement_session_id": measurement_session_id,
            "started_at_utc": _utc_now(),
            "environment": collect_environment(1),
        }
        manifest = {
            "schema_version": NATIVE_BENCHMARK_SCHEMA_VERSION,
            "study_id": "issue-3-native-recheck",
            "run_id": selected_id,
            "run_name": str(effective["run_name"]),
            "phase": str(effective["phase"]),
            "session_id": str(effective.get("session_id", "session-0")),
            "evidence_role": str(effective.get("evidence_role", "validation")),
            "config_hash": config_hash,
            "implementation_tree_sha256": implementation_hash,
            "native_build": dict(native_build_info()),
            "correctness_prerequisite": correctness,
            "holdout_registration": holdout_registration,
            "status": "running",
            "started_at_utc": _utc_now(),
            "environment": collect_environment(1),
            "process_segments": [first_segment],
            "dataset_hashes": {},
            "split_ids": {},
            "warmup": {},
            "failures": [],
            "completed_blocks": 0,
            "total_blocks": total_blocks,
            "measurement_notes": {
                "metric": "ordinary L2; Faiss scan output is squared L2",
                "frozen_C": "same CandidateSet object for all frozen-C micro calls",
                "micro": "prepared frozen C/native query to completed result+Receipt",
                "composed_e2e": "one common Base preparation plus method-specific preparation and micro",
                "api_wall": (
                    "fresh public-path Base preparation through completed result; "
                    "all methods measured in an independent randomized API pass"
                ),
                "truth": (
                    "old O, sampled Fraction oracle, and full-visible truth run only "
                    "after both timing passes"
                ),
                "method_order": (
                    "API and micro are separate deterministically shuffled passes "
                    "per query and repetition"
                ),
                "baseline_mismatch": "A mismatches are retained in all timing aggregates",
                "packed_state": "cold build recorded; all query measurements use warmed immutable view",
            },
        }
    atomic_write_json(manifest_path, manifest)
    if checkpoint_path.exists():
        checkpoint = _read_object(checkpoint_path)
        for field, expected in (
            ("run_id", selected_id),
            ("config_hash", config_hash),
            ("implementation_tree_sha256", implementation_hash),
        ):
            if checkpoint.get(field) != expected:
                raise RuntimeError(f"checkpoint {field} mismatch")
    else:
        checkpoint = {
            "schema_version": NATIVE_BENCHMARK_SCHEMA_VERSION,
            "run_id": selected_id,
            "config_hash": config_hash,
            "implementation_tree_sha256": implementation_hash,
            "completed_blocks": {},
        }
        atomic_write_json(checkpoint_path, checkpoint)
    builds = _read_object(build_path) if build_path.exists() else {}
    newly_completed = 0
    initial_native_calls = native_call_counter()
    full_truth_cache: dict[str, tuple[SearchHit, ...]] = {}
    try:
        for specification in effective["experiments"]:
            experiment_id = str(specification["id"])
            safe_id = _safe_name(experiment_id)
            split = load_dataset(specification["dataset"])
            dataset_hash = split.sha256()
            split_payload = _split_payload(split, experiment_id, dataset_hash)
            split_id = str(split_payload["split_id"])
            split_path = run_dir / "splits" / f"{safe_id}.json"
            if split_path.exists():
                prior_split = _read_object(split_path)
                if prior_split.get("split_id") != split_id:
                    raise RuntimeError(f"split changed on resume: {experiment_id}")
            else:
                atomic_write_json(split_path, split_payload)
            manifest["dataset_hashes"][experiment_id] = dataset_hash
            manifest["split_ids"][experiment_id] = split_id
            atomic_write_json(manifest_path, manifest)

            prepared = _prepare_experiment(split, specification)
            current_build = {
                **prepared.build_manifest,
                "dataset_hash": dataset_hash,
                "split_id": split_id,
                "built_at_utc": _utc_now(),
            }
            if experiment_id in builds:
                prior = builds[experiment_id]
                if (
                    prior.get("dataset_hash") != dataset_hash
                    or prior.get("split_id") != split_id
                    or prior.get("native_build", {}).get("shared_object_sha256")
                    != current_build["native_build"].get("shared_object_sha256")
                ):
                    raise RuntimeError(f"build identity changed on resume: {experiment_id}")
                rebuilds = list(prior.get("resume_rebuilds", []))
                rebuilds.append(current_build)
                prior["resume_rebuilds"] = rebuilds
            else:
                builds[experiment_id] = current_build
            atomic_write_json(build_path, builds)

            beta_path = run_dir / "validation" / f"{safe_id}.json"
            if beta_path.exists():
                beta_selection = _read_object(beta_path)
                if (
                    beta_selection.get("dataset_hash") != dataset_hash
                    or beta_selection.get("split_id") != split_id
                ):
                    raise RuntimeError(f"beta selection changed: {experiment_id}")
            else:
                beta_selection = _validation_betas(split, specification, prepared)
                beta_selection.update(
                    {
                        "experiment_id": experiment_id,
                        "dataset_hash": dataset_hash,
                        "split_id": split_id,
                        "implementation_tree_sha256": implementation_hash,
                    }
                )
                atomic_write_json(beta_path, beta_selection)
            methods = _methods(specification, beta_selection)

            warmup = _warmup(
                split,
                specification,
                prepared,
                methods,
                int(effective.get("warmup_queries", 0)),
            )
            manifest["warmup"][experiment_id] = warmup
            atomic_write_json(manifest_path, manifest)

            cache_path = run_dir / "audits" / f"{safe_id}.base-cache.json"
            if not cache_path.exists():
                atomic_write_json(
                    cache_path,
                    {
                        "schema_version": NATIVE_BENCHMARK_SCHEMA_VERSION,
                        "experiment_id": experiment_id,
                        "dataset_hash": dataset_hash,
                        "split_id": split_id,
                        **_base_cache_ablation(
                            split,
                            specification,
                            prepared,
                            int(effective.get("base_cache_ablation_queries", 0)),
                        ),
                    },
                )
            audit_path = run_dir / "audits" / f"{safe_id}.native-lb.json"
            expected_audit_queries = min(
                int(effective.get("lb_audit_queries", 0)),
                len(split.validation_queries),
            )
            if not audit_path.exists():
                audit = _native_lb_audit(
                    split,
                    specification,
                    prepared,
                    int(effective.get("lb_audit_queries", 0)),
                )
                atomic_write_json(
                    audit_path,
                    {
                        "schema_version": NATIVE_BENCHMARK_SCHEMA_VERSION,
                        "experiment_id": experiment_id,
                        "dataset_hash": dataset_hash,
                        "split_id": split_id,
                        **audit,
                    },
                )
                if not audit["passed"]:
                    raise RuntimeError(f"native LB audit failed: {experiment_id}")
            _verify_native_lb_audit_file(
                audit_path,
                experiment_id=experiment_id,
                dataset_hash=dataset_hash,
                split_id=split_id,
                expected_query_count=expected_audit_queries,
            )

            fraction_count = min(
                int(effective.get("independent_oracle_queries", 0)),
                len(split.test_queries),
            )
            fraction_positions = set(range(fraction_count))
            block_size = int(effective["query_block_size"])
            for start in range(0, len(split.test_queries), block_size):
                stop = min(len(split.test_queries), start + block_size)
                key = f"{safe_id}:{start:08d}:{stop:08d}"
                filename = f"{safe_id}.q{start:08d}-{stop:08d}.jsonl"
                raw_path = run_dir / "raw" / filename
                prior_entry = checkpoint["completed_blocks"].get(key)
                if prior_entry is not None:
                    _verify_raw_block(
                        raw_path,
                        prior_entry,
                        run_id=selected_id,
                        config_hash=config_hash,
                        implementation_hash=implementation_hash,
                    )
                    continue
                if raw_path.exists():
                    orphan_hash = file_sha256(raw_path)
                    orphan_name = (
                        f"orphan-{filename}-{orphan_hash[:12]}-"
                        f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')}"
                    )
                    orphan_path = run_dir / "failures" / orphan_name
                    os.replace(raw_path, orphan_path)
                    manifest["failures"].append(
                        {
                            "kind": "uncheckpointed_raw_preserved_on_resume",
                            "original_path": f"raw/{filename}",
                            "preserved_path": f"failures/{orphan_name}",
                            "sha256": orphan_hash,
                        }
                    )
                    atomic_write_json(manifest_path, manifest)
                rows: list[dict[str, Any]] = []
                for position in range(start, stop):
                    try:
                        query_rows = _run_query(
                            run_id=selected_id,
                            config_hash=config_hash,
                            implementation_hash=implementation_hash,
                            config=effective,
                            measurement_session_id=measurement_session_id,
                            dataset_hash=dataset_hash,
                            split_id=split_id,
                            split=split,
                            specification=specification,
                            prepared=prepared,
                            methods=methods,
                            query=split.test_queries[position],
                            query_id=int(split.test_query_ids[position]),
                            query_position=position,
                            fraction_oracle_positions=fraction_positions,
                            full_truth_cache=full_truth_cache,
                        )
                    except BaseException as query_error:
                        failure_name = (
                            f"{safe_id}.{_safe_name(measurement_session_id)}."
                            f"q{position:08d}.json"
                        )
                        failure_path = run_dir / "failures" / failure_name
                        failure = {
                            "schema_version": NATIVE_BENCHMARK_SCHEMA_VERSION,
                            "run_id": selected_id,
                            "config_hash": config_hash,
                            "implementation_tree_sha256": implementation_hash,
                            "measurement_session_id": measurement_session_id,
                            "experiment_id": experiment_id,
                            "dataset_hash": dataset_hash,
                            "split_id": split_id,
                            "query_id": int(split.test_query_ids[position]),
                            "query_position": position,
                            "query_float32_values": split.test_queries[position].tolist(),
                            "error_type": type(query_error).__name__,
                            "error_message": str(query_error),
                            "traceback": traceback.format_exc(),
                        }
                        atomic_write_json(failure_path, failure)
                        manifest["failures"].append(
                            {
                                "kind": "query_counterexample",
                                "path": f"failures/{failure_name}",
                                "sha256": file_sha256(failure_path),
                            }
                        )
                        atomic_write_json(manifest_path, manifest)
                        raise
                    rows.extend(query_rows)
                atomic_write_jsonl(raw_path, rows)
                entry = {
                    "path": f"raw/{filename}",
                    "sha256": file_sha256(raw_path),
                    "rows": len(rows),
                    "experiment_id": experiment_id,
                    "query_start": start,
                    "query_stop": stop,
                    "completed_at_utc": _utc_now(),
                }
                checkpoint["completed_blocks"][key] = entry
                atomic_write_json(checkpoint_path, checkpoint)
                newly_completed += 1
                violations = [row for row in rows if row.get("contract_violation") is True]
                if violations:
                    manifest["failures"].append(
                        {
                            "kind": "native_contract_violation",
                            "block": key,
                            "count": len(violations),
                            "examples": violations[:10],
                        }
                    )
                manifest["completed_blocks"] = len(checkpoint["completed_blocks"])
                manifest["native_calls_so_far"] = native_call_counter() - initial_native_calls
                atomic_write_json(manifest_path, manifest)
                if violations:
                    raise RuntimeError(
                        f"native contract violation persisted in block {key}; stopping"
                    )
                if stop_after_blocks is not None and newly_completed >= stop_after_blocks:
                    manifest["status"] = "incomplete_controlled_stop"
                    manifest["stopped_at_utc"] = _utc_now()
                    atomic_write_json(manifest_path, manifest)
                    return RunOutcome(
                        run_dir,
                        selected_id,
                        False,
                        len(checkpoint["completed_blocks"]),
                        total_blocks,
                    )

        if len(checkpoint["completed_blocks"]) != total_blocks:
            raise RuntimeError("run ended without every configured query block")
        if _has_disqualifying_failures(manifest):
            raise RuntimeError("run has persisted scientific failures; refusing completion")
        manifest["status"] = "completed"
        manifest["completed_at_utc"] = _utc_now()
        manifest["completed_blocks"] = total_blocks
        manifest["total_blocks"] = total_blocks
        manifest["native_calls_total"] = native_call_counter() - initial_native_calls
        atomic_write_json(manifest_path, manifest)
        completion = {
            "schema_version": NATIVE_BENCHMARK_SCHEMA_VERSION,
            "study_id": "issue-3-native-recheck",
            "run_id": selected_id,
            "config_hash": config_hash,
            "implementation_tree_sha256": implementation_hash,
            "completed_at_utc": _utc_now(),
            "raw_shards": total_blocks,
            "checkpoint_sha256": file_sha256(checkpoint_path),
            "run_manifest_sha256": file_sha256(manifest_path),
            "ancillary_files": _ancillary_inventory(run_dir),
        }
        if holdout_registration is not None:
            completion.update(
                {
                    "holdout_manifest_sha256": holdout_registration["sha256"],
                    "final_policy_file_sha256": holdout_registration[
                        "final_policy_file_sha256"
                    ],
                    "final_policy_object_sha256": holdout_registration[
                        "final_policy_object_sha256"
                    ],
                }
            )
        atomic_write_json(run_dir / "COMPLETED.json", completion)
        return RunOutcome(run_dir, selected_id, True, total_blocks, total_blocks)
    except BaseException as error:
        manifest["status"] = "failed_incomplete"
        manifest["failed_at_utc"] = _utc_now()
        manifest["failures"].append(
            {
                "kind": type(error).__name__,
                "message": str(error),
                "traceback": traceback.format_exc(),
            }
        )
        atomic_write_json(manifest_path, manifest)
        raise


__all__ = [
    "NATIVE_BENCHMARK_SCHEMA_VERSION",
    "apply_native_resource_limits",
    "run_native_benchmark",
]

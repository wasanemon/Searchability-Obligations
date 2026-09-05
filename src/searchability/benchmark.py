"""Reproducible, resumable benchmark harness for Issue #1.

The harness intentionally keeps measurement orchestration separate from the
search implementations.  In particular, one :class:`CandidateSet` is frozen
per test query and the *same Python object* is handed to every coupled method
and every measured repetition for that query.

Raw output is sharded at query-block boundaries.  A shard is atomically
published before its checksum is added to ``checkpoint.json``; this makes an
interrupted run resumable without ever rewriting a completed run.
"""

from __future__ import annotations

from contextlib import redirect_stdout
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from fractions import Fraction
import io
import json
import math
import os
from pathlib import Path
import platform
import random
import resource
import subprocess
import sys
import time
from typing import Any, Callable, Iterable, Mapping, Sequence

import faiss
import numpy as np

from .artifacts import (
    atomic_write_json,
    atomic_write_jsonl,
    file_sha256,
    implementation_tree_sha256,
    object_sha256,
)
from .base import BaseIndex
from .baselines import BaselineResult, PreparedFaissIndex, visible_unique_records
from .datasets import DatasetSplit, load_dataset
from .groups import DeltaStore, GroupDirectory
from .models import CandidateSet, SearchHit, SearchResult, VectorRecord
from .oracle import (
    exact_squared_l2 as oracle_exact_squared_l2,
    observed_gap_bracket,
    reference_topk,
)
from .search import SearchEngine


SCHEMA_VERSION = 1
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
COUPLED_KINDS = {
    "delta_flat",
    "certified_full",
    "pruned",
    "pruned_recompute",
    "delta_hnsw",
    "no_pruning",
}
CERTIFIED_KINDS = {"certified_full", "pruned", "pruned_recompute", "no_pruning"}
TRUTH_KINDS = {"certified_full", "full_exact_truth"}
THREAD_VARIABLES = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)


class BenchmarkConfigurationError(ValueError):
    """The JSON experiment specification is inconsistent or unsafe."""


class CompletedRunError(FileExistsError):
    """Raised rather than modifying an already completed run directory."""


@dataclass(frozen=True, slots=True)
class RunOutcome:
    run_dir: Path
    run_id: str
    completed: bool
    completed_blocks: int
    total_blocks: int


@dataclass(frozen=True, slots=True)
class _Method:
    name: str
    kind: str
    requested_beta: float | None = None
    beta_factor: float | None = None
    ef_search: int | None = None


@dataclass(slots=True)
class _Measured:
    hits: tuple[SearchHit, ...]
    wall_ns: int
    micro_latency_ns: int | None
    end_to_end_latency_ns: int
    component_timings: dict[str, int]
    receipt: dict[str, Any] | None
    index_candidate_count: int | None
    peak_rss_bytes: int
    truth_cache_hit: bool = False
    truth_cache_key: str | None = None


@dataclass(slots=True)
class _PreparedExperiment:
    engine: SearchEngine
    base: BaseIndex
    delta_records: tuple[VectorRecord, ...]
    grouped_records: tuple[VectorRecord, ...]
    raw_records: tuple[VectorRecord, ...]
    delta_flat: PreparedFaissIndex
    delta_hnsw: PreparedFaissIndex
    full_flat: PreparedFaissIndex
    full_hnsw: PreparedFaissIndex
    delta_logical_ids: frozenset[int]
    delta_by_key: dict[tuple[int, int], VectorRecord]
    build_manifest: dict[str, Any]


def _json_copy(value: Mapping[str, Any]) -> dict[str, Any]:
    try:
        return json.loads(json.dumps(value, allow_nan=False))
    except (TypeError, ValueError) as error:
        raise BenchmarkConfigurationError(
            "configuration must contain only finite JSON values"
        ) from error


def load_json_config(path: str | Path) -> dict[str, Any]:
    location = Path(path)
    try:
        value = json.loads(location.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise BenchmarkConfigurationError(f"cannot read JSON config {location}") from error
    if not isinstance(value, dict):
        raise BenchmarkConfigurationError("top-level config must be a JSON object")
    return _json_copy(value)


def apply_resource_limits(
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
    """Return the exact effective config, including all CLI scale reductions."""

    effective = _json_copy(config)
    experiments = effective.get("experiments")
    if not isinstance(experiments, list) or not experiments:
        raise BenchmarkConfigurationError("config.experiments must be a non-empty list")
    if only_experiment is not None:
        experiments = [row for row in experiments if row.get("id") == only_experiment]
        if not experiments:
            raise BenchmarkConfigurationError(
                f"experiment {only_experiment!r} is not present in the config"
            )
    if max_experiments is not None:
        if max_experiments <= 0:
            raise BenchmarkConfigurationError("max_experiments must be positive")
        experiments = experiments[:max_experiments]

    positive_limits = {
        "n_base": max_base,
        "n_delta": max_delta,
        "n_validation": max_validation_queries,
        "n_test": max_test_queries,
    }
    for experiment in experiments:
        if not isinstance(experiment, dict) or not isinstance(experiment.get("dataset"), dict):
            raise BenchmarkConfigurationError("each experiment needs a dataset object")
        dataset = experiment["dataset"]
        for key, limit in positive_limits.items():
            if limit is None:
                continue
            if limit < 0 or (key != "n_delta" and limit == 0):
                raise BenchmarkConfigurationError(f"invalid limit for {key}: {limit}")
            dataset[key] = min(int(dataset[key]), int(limit))
        if int(dataset["n_base"]) <= 0:
            raise BenchmarkConfigurationError("the effective base size must be positive")
        experiment["n_groups"] = min(
            int(experiment.get("n_groups", 1)), int(dataset["n_base"])
        )
        if "center_training_size" in experiment:
            experiment["center_training_size"] = min(
                int(experiment["center_training_size"]), int(dataset["n_base"])
            )
        experiment.setdefault("base_hnsw_ef_search", 128)
    effective["experiments"] = experiments
    if max_repetitions is not None:
        if max_repetitions <= 0:
            raise BenchmarkConfigurationError("max_repetitions must be positive")
        effective["repetitions"] = min(
            int(effective.get("repetitions", 1)), int(max_repetitions)
        )
        for experiment in experiments:
            if "repetitions" in experiment:
                experiment["repetitions"] = min(
                    int(experiment["repetitions"]), int(max_repetitions)
                )
    if max_test_queries is not None and "throughput_queries" in effective:
        effective["throughput_queries"] = min(
            int(effective["throughput_queries"]), int(max_test_queries)
        )
    if max_test_queries is not None and "independent_oracle_queries" in effective:
        effective["independent_oracle_queries"] = min(
            int(effective["independent_oracle_queries"]), int(max_test_queries)
        )
    if max_test_queries is not None and "lb_audit_queries" in effective:
        effective["lb_audit_queries"] = min(
            int(effective["lb_audit_queries"]), int(max_test_queries)
        )
    effective["applied_resource_limits"] = {
        "only_experiment": only_experiment,
        "max_experiments": max_experiments,
        "max_base": max_base,
        "max_delta": max_delta,
        "max_validation_queries": max_validation_queries,
        "max_test_queries": max_test_queries,
        "max_repetitions": max_repetitions,
    }
    if any(
        value is not None
        for value in (
            only_experiment,
            max_experiments,
            max_base,
            max_delta,
            max_validation_queries,
            max_test_queries,
            max_repetitions,
        )
    ):
        effective["source_evidence_role_before_resource_limits"] = str(
            effective.get("evidence_role", "calibration")
        )
        effective["evidence_role"] = "calibration"
    return effective


def _validate_config(config: Mapping[str, Any]) -> None:
    if int(config.get("schema_version", -1)) != SCHEMA_VERSION:
        raise BenchmarkConfigurationError(
            f"schema_version must be {SCHEMA_VERSION}"
        )
    if not str(config.get("run_name", "")).strip():
        raise BenchmarkConfigurationError("run_name is required")
    if not str(config.get("output_root", "")).strip():
        raise BenchmarkConfigurationError("output_root is required")
    threads = int(config.get("threads", 1))
    if threads <= 0:
        raise BenchmarkConfigurationError("threads must be positive")
    if int(config.get("query_block_size", 0)) <= 0:
        raise BenchmarkConfigurationError("query_block_size must be positive")
    if int(config.get("repetitions", 0)) <= 0:
        raise BenchmarkConfigurationError("repetitions must be positive")
    if int(config.get("warmup_queries", 0)) < 0:
        raise BenchmarkConfigurationError("warmup_queries cannot be negative")
    if int(config.get("throughput_queries", 0)) < 0:
        raise BenchmarkConfigurationError("throughput_queries cannot be negative")
    if int(config.get("throughput_repetitions", 1)) <= 0:
        raise BenchmarkConfigurationError("throughput_repetitions must be positive")
    if int(config.get("independent_oracle_queries", 0)) < 0:
        raise BenchmarkConfigurationError("independent_oracle_queries cannot be negative")
    if int(config.get("lb_audit_queries", 0)) < 0:
        raise BenchmarkConfigurationError("lb_audit_queries cannot be negative")
    if int(config.get("independent_oracle_max_population", 0)) < 0:
        raise BenchmarkConfigurationError(
            "independent_oracle_max_population cannot be negative"
        )
    if str(config.get("evidence_role", "calibration")) not in {
        "final",
        "calibration",
        "obsolete",
    }:
        raise BenchmarkConfigurationError(
            "evidence_role must be final, calibration, or obsolete"
        )

    experiments = config.get("experiments")
    if not isinstance(experiments, list) or not experiments:
        raise BenchmarkConfigurationError("experiments must be non-empty")
    identifiers: set[str] = set()
    for row in experiments:
        if not isinstance(row, dict):
            raise BenchmarkConfigurationError("experiment entries must be objects")
        identifier = str(row.get("id", ""))
        if not identifier or identifier in identifiers:
            raise BenchmarkConfigurationError("experiment IDs must be unique and non-empty")
        identifiers.add(identifier)
        dataset = row.get("dataset")
        if not isinstance(dataset, dict):
            raise BenchmarkConfigurationError(f"{identifier}: dataset must be an object")
        for key in ("n_base", "n_delta", "n_validation", "n_test"):
            if key not in dataset or int(dataset[key]) < 0:
                raise BenchmarkConfigurationError(f"{identifier}: invalid dataset.{key}")
        if int(dataset["n_base"]) <= 0 or int(dataset["n_test"]) <= 0:
            raise BenchmarkConfigurationError(f"{identifier}: base and test must be non-empty")
        k = int(row.get("k", 0))
        candidate_count = int(row.get("candidate_count", 0))
        groups = int(row.get("n_groups", 0))
        if k <= 0 or candidate_count <= 0 or groups <= 0:
            raise BenchmarkConfigurationError(
                f"{identifier}: k, candidate_count and n_groups must be positive"
            )
        if groups > int(dataset["n_base"]):
            raise BenchmarkConfigurationError(
                f"{identifier}: n_groups exceeds base-only center training population"
            )
        if int(row.get("raw_pending_count", 0)) < 0:
            raise BenchmarkConfigurationError(f"{identifier}: raw_pending_count is negative")
        if int(row.get("repetitions", config["repetitions"])) <= 0:
            raise BenchmarkConfigurationError(
                f"{identifier}: repetitions must be positive"
            )
        factors = row.get("positive_beta_factors", [])
        if not isinstance(factors, list) or not factors:
            raise BenchmarkConfigurationError(
                f"{identifier}: at least one positive_beta_factor is required"
            )
        parsed_factors = [float(value) for value in factors]
        if any(not math.isfinite(value) or value <= 0.0 for value in parsed_factors):
            raise BenchmarkConfigurationError(
                f"{identifier}: positive_beta_factors must be finite and > 0"
            )
        if len(set(parsed_factors)) != len(parsed_factors):
            raise BenchmarkConfigurationError(f"{identifier}: duplicate beta factors")
        ef_values = [int(value) for value in row.get("hnsw_ef_search", [])]
        if len(ef_values) < 2 or any(value <= 0 for value in ef_values):
            raise BenchmarkConfigurationError(
                f"{identifier}: hnsw_ef_search needs at least two positive values"
            )
        if int(row.get("base_hnsw_ef_search", 128)) <= 0:
            raise BenchmarkConfigurationError(
                f"{identifier}: base_hnsw_ef_search must be positive"
            )


def _safe_name(value: str) -> str:
    rendered = "".join(character if character.isalnum() or character in "-_" else "-" for character in value)
    rendered = rendered.strip("-")
    if not rendered:
        raise BenchmarkConfigurationError("identifier has no filesystem-safe characters")
    return rendered


def _float_name(value: float) -> str:
    return format(value, ".8g").replace("-", "m").replace("+", "p").replace(".", "p")


def _run_command(command: Sequence[str]) -> str | None:
    try:
        completed = subprocess.run(
            list(command), check=False, capture_output=True, text=True, timeout=15
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return completed.stdout.strip() or completed.stderr.strip() or None


def _peak_rss_bytes() -> int:
    # Linux reports KiB; macOS reports bytes.  Experiments are currently Linux,
    # but keeping the conversion explicit makes the manifest unambiguous.
    observed = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return observed if sys.platform == "darwin" else observed * 1024


def _current_rss_bytes() -> int | None:
    try:
        pages = int(Path("/proc/self/statm").read_text(encoding="ascii").split()[1])
        return pages * os.sysconf("SC_PAGE_SIZE")
    except (OSError, ValueError, IndexError):
        return None


def _cpu_affinity() -> list[int] | None:
    """Return the scheduler-visible CPU set when the host exposes it."""

    getter = getattr(os, "sched_getaffinity", None)
    if getter is None:
        return None
    try:
        return sorted(int(value) for value in getter(0))
    except OSError:
        return None


def collect_environment(threads: int) -> dict[str, Any]:
    output = io.StringIO()
    with redirect_stdout(output):
        np.__config__.show()
    return {
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_count_logical": os.cpu_count(),
        "cpu_affinity": _cpu_affinity(),
        "faiss_version": faiss.__version__,
        "faiss_compile_options": faiss.get_compile_options(),
        "faiss_threads": faiss.omp_get_max_threads(),
        "numpy_version": np.__version__,
        "numpy_build_configuration": output.getvalue(),
        "thread_environment": {name: os.environ.get(name) for name in THREAD_VARIABLES},
        "configured_threads": threads,
        "git_commit": _run_command(("git", "rev-parse", "HEAD")),
        "git_status_porcelain": _run_command(("git", "status", "--porcelain")),
        "lscpu": _run_command(("lscpu",)),
        "memory_info": (
            Path("/proc/meminfo").read_text(encoding="ascii")
            if Path("/proc/meminfo").exists()
            else None
        ),
        "rss_bytes_at_start": _current_rss_bytes(),
        "peak_rss_bytes_at_start": _peak_rss_bytes(),
        "shared_host_warning": (
            "No exclusive CPU reservation was asserted; latency can include shared-host noise."
        ),
    }


def _new_run_id(root: Path, run_name: str, config_hash: str) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    stem = f"{_safe_name(run_name)}-{stamp}-{config_hash[:10]}"
    candidate = stem
    suffix = 0
    while (root / candidate).exists():
        suffix += 1
        candidate = f"{stem}-{suffix}"
    return candidate


def _select_run_dir(
    config: Mapping[str, Any], config_hash: str, *, resume: bool, run_id: str | None
) -> tuple[Path, str, bool]:
    root = Path(str(config["output_root"]))
    root.mkdir(parents=True, exist_ok=True)
    if run_id is not None:
        selected_id = _safe_name(run_id)
        directory = root / selected_id
        if (directory / "COMPLETED.json").exists():
            raise CompletedRunError(f"completed run is immutable: {directory}")
        if directory.exists() and not resume:
            raise FileExistsError(f"run directory already exists: {directory}")
        return directory, selected_id, directory.exists()

    if resume:
        candidates: list[tuple[int, Path, str]] = []
        for manifest_path in root.glob("*/run_manifest.json"):
            directory = manifest_path.parent
            if (directory / "COMPLETED.json").exists():
                continue
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if manifest.get("config_hash") == config_hash:
                candidates.append((manifest_path.stat().st_mtime_ns, directory, directory.name))
        if candidates:
            _, directory, selected_id = max(candidates)
            return directory, selected_id, True

    selected_id = _new_run_id(root, str(config["run_name"]), config_hash)
    return root / selected_id, selected_id, False


def _records(vectors: np.ndarray, identifiers: np.ndarray) -> tuple[VectorRecord, ...]:
    return tuple(
        VectorRecord(int(identifier), 0, vector)
        for identifier, vector in zip(identifiers.tolist(), vectors)
    )


def _serialize_size(index: faiss.Index) -> tuple[int, int]:
    start = time.perf_counter_ns()
    serialized = faiss.serialize_index(index)
    return int(serialized.nbytes), time.perf_counter_ns() - start


def _prepare_experiment(
    split: DatasetSplit, specification: Mapping[str, Any], threads: int
) -> _PreparedExperiment:
    base_records = _records(split.base, split.base_ids)
    delta_records = _records(split.delta, split.delta_ids)
    if len({record.key for record in delta_records}) != len(delta_records):
        raise BenchmarkConfigurationError("Delta contains duplicate version keys")
    dimension = split.dimension
    m = int(specification.get("hnsw_m", 32))
    ef_construction = int(specification.get("hnsw_ef_construction", 200))
    ef_values = [int(value) for value in specification["hnsw_ef_search"]]
    comparator_initial_ef = max(ef_values)
    base_ef_search = int(specification.get("base_hnsw_ef_search", 128))
    build_start_rss = _current_rss_bytes()
    build_start_peak = _peak_rss_bytes()

    start = time.perf_counter_ns()
    base = BaseIndex(
        base_records,
        dimension=dimension,
        generation_id=f"benchmark-{specification['id']}",
        covered_commit_seq=0,
        m=m,
        ef_construction=ef_construction,
        ef_search=base_ef_search,
        threads=threads,
    )
    base_build_ns = time.perf_counter_ns() - start
    base_serialized_bytes, base_serialize_ns = _serialize_size(base.index)

    raw_count = min(int(specification.get("raw_pending_count", 0)), len(delta_records))
    grouped_records = delta_records[: len(delta_records) - raw_count]
    raw_records = delta_records[len(delta_records) - raw_count :]
    training_count = min(
        len(base_records), int(specification.get("center_training_size", len(base_records)))
    )
    if training_count <= 0:
        raise BenchmarkConfigurationError("center training needs at least one base vector")
    n_groups = int(specification["n_groups"])
    centers, center_training_ns = GroupDirectory.train_centers(
        split.base[:training_count],
        n_groups,
        seed=int(specification.get("group_seed", 0)),
        iterations=int(specification.get("kmeans_iterations", 20)),
        threads=threads,
    )
    directory = GroupDirectory.from_centers(
        centers,
        grouped_records,
        center_training_ns=center_training_ns,
        threads=threads,
    )
    delta_store = DeltaStore(raw_records=raw_records, grouped=directory)
    engine = SearchEngine(base, delta_store)
    visibility_view_start = time.perf_counter_ns()
    engine._visible_raw(0)
    engine._visible_groups(0)
    visibility_view_build_ns = time.perf_counter_ns() - visibility_view_start

    delta_flat = PreparedFaissIndex(
        delta_records, dimension=dimension, kind="flat", threads=threads
    )
    delta_hnsw = PreparedFaissIndex(
        delta_records,
        dimension=dimension,
        kind="hnsw",
        m=m,
        ef_construction=ef_construction,
        ef_search=comparator_initial_ef,
        threads=threads,
    )
    # The exact full-population comparator is built only from records visible
    # at the benchmark snapshot, with the same logical-ID resolution used by
    # the rest of the artifact.  Dataset splits currently contain one version
    # per ID, but keeping this explicit prevents a future MVCC split from
    # silently turning the ground truth into an all-version scan.
    full_records = visible_unique_records(base_records + delta_records, 0)
    full_flat = PreparedFaissIndex(
        full_records, dimension=dimension, kind="flat", threads=threads
    )
    full_hnsw = PreparedFaissIndex(
        full_records,
        dimension=dimension,
        kind="hnsw",
        m=m,
        ef_construction=ef_construction,
        ef_search=comparator_initial_ef,
        threads=threads,
    )

    radii = [float(group.radius_upper) for group in directory.groups]
    member_counts = [len(group.members) for group in directory.groups]
    build_end_rss = _current_rss_bytes()
    build_manifest = {
        "base": {
            "kind": "hnsw",
            "build_wall_ns": base_build_ns,
            "serialize_ns": base_serialize_ns,
            "serialized_bytes": base_serialized_bytes,
            "vectors": len(base_records),
            "dimension": dimension,
            "m": m,
            "ef_construction": ef_construction,
            "ef_search": base_ef_search,
            "ef_search_role": "frozen_base_candidate_generation",
        },
        "groups": {
            **asdict(directory.build_stats),
            "n_groups": len(directory.groups),
            "training_vectors": training_count,
            "training_source": "base_only",
            "group_seed": int(specification.get("group_seed", 0)),
            "kmeans_iterations": int(specification.get("kmeans_iterations", 20)),
            "grouped_vectors": len(grouped_records),
            "raw_pending_vectors": len(raw_records),
            "radius_upper_values": radii,
            "member_counts": member_counts,
            "snapshot_zero_visible_view_cache_build_ns": visibility_view_build_ns,
            "snapshot_zero_visible_view_cache_scope": (
                "immutable raw/group MVCC filtering and visible group matrix packing"
            ),
        },
        "delta_flat": asdict(delta_flat.build_stats),
        "delta_hnsw": asdict(delta_hnsw.build_stats),
        "full_flat": asdict(full_flat.build_stats),
        "full_hnsw": asdict(full_hnsw.build_stats),
        "comparator_hnsw_query_sweep": {
            "ef_search_values": ef_values,
            "initial_ef_search_before_per_method_override": comparator_initial_ef,
            "scope": "delta_hnsw_and_full_population_hnsw_only",
        },
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
            "group_member_bytes": int(directory.build_stats.member_bytes),
            "group_metadata_bytes_estimate": int(directory.build_stats.metadata_bytes),
        },
        "maintenance_scope_note": (
            "Index construction/add and grouping costs are component costs, not an ACID "
            "synchronous-update throughput measurement."
        ),
    }
    return _PreparedExperiment(
        engine=engine,
        base=base,
        delta_records=delta_records,
        grouped_records=grouped_records,
        raw_records=raw_records,
        delta_flat=delta_flat,
        delta_hnsw=delta_hnsw,
        full_flat=full_flat,
        full_hnsw=full_hnsw,
        delta_logical_ids=frozenset(
            record.logical_id for record in delta_records
        ),
        delta_by_key={record.key: record for record in delta_records},
        build_manifest=build_manifest,
    )


def _validation_betas(
    split: DatasetSplit,
    specification: Mapping[str, Any],
    prepared: _PreparedExperiment,
) -> dict[str, Any]:
    k = int(specification["k"])
    candidate_count = int(specification["candidate_count"])
    factors = [float(value) for value in specification["positive_beta_factors"]]
    reference_taus: list[float] = []
    query_rows: list[dict[str, Any]] = []
    for query_id, query in zip(
        split.validation_query_ids.tolist(), split.validation_queries
    ):
        candidates = prepared.engine.prepare_candidates(
            query, snapshot_id=0, k=k, candidate_count=candidate_count
        )
        result = prepared.engine.search_full_scan(
            query,
            k=k,
            beta=0.0,
            snapshot_id=0,
            candidate_count=candidate_count,
            candidate_set=candidates,
            audit=False,
        )
        tau = None if not result.hits else float(result.hits[-1].distance)
        if tau is not None:
            reference_taus.append(tau)
        query_rows.append(
            {
                "query_id": int(query_id),
                "candidate_set_id_or_hash": candidates.candidate_hash,
                "reference_tau_l2": tau,
            }
        )
    if not reference_taus:
        raise BenchmarkConfigurationError(
            f"{specification['id']}: validation produced no reference tau"
        )
    median = float(np.median(np.asarray(reference_taus, dtype=np.float64)))
    floor = float(specification.get("positive_beta_min", 1.0e-12))
    if not math.isfinite(floor) or floor <= 0.0:
        raise BenchmarkConfigurationError("positive_beta_min must be finite and > 0")
    selected = {
        format(factor, ".17g"): max(floor, factor * median) for factor in factors
    }
    return {
        "selection_source": "validation_queries_only",
        "reference_method": "base_frozen_C_plus_certified_full_delta_scan",
        "statistic": "median_kth_ordinary_l2",
        "median_reference_tau_l2": median,
        "positive_beta_floor": floor,
        "factors_to_absolute_beta": selected,
        "validation_query_count": len(query_rows),
        "validation_rows": query_rows,
        "test_queries_used_for_selection": 0,
    }


def _methods(
    specification: Mapping[str, Any], beta_selection: Mapping[str, Any]
) -> list[_Method]:
    methods = [
        _Method("delta_flat_full_scan", "delta_flat", requested_beta=0.0),
        _Method("certified_full_delta_reference", "certified_full", requested_beta=0.0),
        _Method("group_pruning_beta0", "pruned", requested_beta=0.0),
        _Method(
            "group_pruning_beta0_recompute_intervals_ablation",
            "pruned_recompute",
            requested_beta=0.0,
        ),
    ]
    values = beta_selection["factors_to_absolute_beta"]
    for factor in [float(value) for value in specification["positive_beta_factors"]]:
        methods.append(
            _Method(
                f"group_pruning_beta_factor_{_float_name(factor)}",
                "pruned",
                requested_beta=float(values[format(factor, ".17g")]),
                beta_factor=factor,
            )
        )
    for ef_search in [int(value) for value in specification["hnsw_ef_search"]]:
        methods.append(
            _Method(
                f"delta_hnsw_ef{ef_search}", "delta_hnsw", ef_search=ef_search
            )
        )
    for ef_search in [int(value) for value in specification["hnsw_ef_search"]]:
        methods.append(
            _Method(
                f"full_base_delta_hnsw_ef{ef_search}",
                "full_hnsw",
                ef_search=ef_search,
            )
        )
    methods.extend(
        (
            _Method("full_base_delta_flat_scan_performance", "full_flat_performance"),
            _Method("full_base_delta_exact_flat", "full_exact_truth"),
            _Method("grouping_no_pruning_ablation", "no_pruning", requested_beta=0.0),
        )
    )
    if len({method.name for method in methods}) != len(methods):
        raise BenchmarkConfigurationError("method names are not unique")
    return methods


def _measure(
    method: _Method,
    prepared: _PreparedExperiment,
    query: np.ndarray,
    *,
    k: int,
    candidate_count: int,
    candidates: CandidateSet,
    base_prepare_wall_ns: int,
    audit: bool,
) -> _Measured:
    # Switching an immutable comparator's configured search effort is setup,
    # not query service work.  Exclude it consistently from per-query timing.
    if method.kind == "delta_hnsw":
        assert method.ef_search is not None
        prepared.delta_hnsw.set_ef_search(method.ef_search)
    elif method.kind == "full_hnsw":
        assert method.ef_search is not None
        prepared.full_hnsw.set_ef_search(method.ef_search)

    start = time.perf_counter_ns()
    receipt_object = None
    baseline: BaselineResult | None = None
    index_candidate_count: int | None = None
    if method.kind == "delta_flat":
        baseline = prepared.delta_flat.search_and_merge_candidates_performance(
            query,
            k=k,
            base_candidates=candidates,
            ann_candidate_count=max(k, candidate_count),
            source="delta_flat_performance",
        )
        hits = baseline.hits
        index_candidate_count = baseline.candidate_count
    elif method.kind == "certified_full":
        result = prepared.engine.search_full_scan(
            query,
            k=k,
            beta=0.0,
            snapshot_id=0,
            candidate_count=candidate_count,
            candidate_set=candidates,
            audit=audit,
        )
        hits = result.hits
        receipt_object = result.receipt
    elif method.kind in {"pruned", "pruned_recompute", "no_pruning"}:
        result = prepared.engine.search_pruned(
            query,
            k=k,
            beta=float(method.requested_beta or 0.0),
            snapshot_id=0,
            candidate_count=candidate_count,
            candidate_set=candidates,
            no_pruning=method.kind == "no_pruning",
            recompute_final_intervals=method.kind == "pruned_recompute",
            audit=audit,
        )
        hits = result.hits
        receipt_object = result.receipt
    elif method.kind == "delta_hnsw":
        baseline = prepared.delta_hnsw.search_and_merge_candidates(
            query,
            k=k,
            base_candidates=candidates,
            ann_candidate_count=max(k, candidate_count),
            source=f"delta_hnsw_ef{method.ef_search}",
        )
        hits = baseline.hits
        index_candidate_count = baseline.candidate_count
    elif method.kind in {"full_flat_performance", "full_exact_truth", "full_hnsw"}:
        if method.kind == "full_hnsw":
            baseline = prepared.full_hnsw.search(query, k=k)
        elif method.kind == "full_flat_performance":
            baseline = prepared.full_flat.search_performance(
                query, k=k, candidate_count=max(k, candidate_count)
            )
        else:
            baseline = prepared.full_flat.search(query, k=k)
        hits = baseline.hits
        index_candidate_count = baseline.candidate_count
    else:  # pragma: no cover - guarded by _methods
        raise AssertionError(method.kind)
    # Stop before materializing the saved Receipt dict.  SearchEngine already
    # includes one structured Receipt conversion in its receipt_ns component;
    # the benchmark writer's deep copy is artifact bookkeeping and must not
    # contaminate only the certified methods' latency.
    wall_ns = time.perf_counter_ns() - start
    receipt = None if receipt_object is None else receipt_object.to_dict()
    if receipt_object is not None:
        component_timings = dict(receipt_object.component_timings)
    elif baseline is not None and method.kind in {
        "full_flat_performance",
        "full_exact_truth",
        "full_hnsw",
    }:
        component_timings = {
            (
                "exact_truth_generation_ns"
                if method.kind == "full_exact_truth"
                else "full_index_search_ns"
            ): baseline.search_ns
        }
        if baseline.merge_ns:
            component_timings["exact_boundary_rerank_ns"] = baseline.merge_ns
    elif baseline is not None:
        component_timings = {
            "delta_index_search_ns": baseline.search_ns,
            "merge_ns": baseline.merge_ns,
        }
    else:  # pragma: no cover - every branch above sets one result object
        raise AssertionError("measurement produced no result")
    coupled = method.kind in COUPLED_KINDS
    return _Measured(
        hits=tuple(hits),
        wall_ns=wall_ns,
        micro_latency_ns=wall_ns if coupled else None,
        end_to_end_latency_ns=(wall_ns + base_prepare_wall_ns if coupled else wall_ns),
        component_timings=component_timings,
        receipt=receipt,
        index_candidate_count=index_candidate_count,
        peak_rss_bytes=_peak_rss_bytes(),
    )


def _hit_ids(measured: _Measured) -> list[int]:
    return [int(hit.logical_id) for hit in measured.hits]


def _hit_keys(measured: _Measured) -> list[list[int]]:
    return [[int(hit.logical_id), int(hit.version_id)] for hit in measured.hits]


def _recall(ids: Sequence[int], truth: Sequence[int]) -> float | None:
    if not truth:
        return None
    return len(set(ids).intersection(truth)) / len(set(truth))


def _exact_observed_gap(
    query: np.ndarray,
    measured: _Measured,
    reference: _Measured,
    records_by_key: Mapping[tuple[int, int], VectorRecord],
    distance_cache: dict[tuple[int, int], Fraction],
) -> tuple[float | None, float | None, bool | None]:
    if not measured.hits or not reference.hits:
        return (None, None, None)

    def cached(key: tuple[int, int]) -> Fraction:
        value = distance_cache.get(key)
        if value is not None:
            return value
        try:
            record = records_by_key[key]
        except KeyError as error:
            raise KeyError(f"result key {key!r} is outside frozen C union Delta") from error
        value = oracle_exact_squared_l2(query, record.vector)
        distance_cache[key] = value
        return value

    proposed_squared = cached(measured.hits[-1].key)
    reference_squared = cached(reference.hits[-1].key)
    bracket = observed_gap_bracket(proposed_squared, reference_squared)
    nonnegative = proposed_squared >= reference_squared
    # The exact squared-distance comparison proves monotonic ordinary-L2
    # ordering.  Intersect the outward numeric bracket with that proved domain
    # so the saved lower endpoint cannot misleadingly be negative.
    lower = max(0.0, float(bracket.lower)) if nonnegative else float(bracket.lower)
    return (lower, float(bracket.upper), nonnegative)


def _row_for_method(
    *,
    run_id: str,
    config_hash: str,
    implementation_hash: str,
    evidence_role: str,
    dataset_hash: str,
    split_id: str,
    experiment_id: str,
    query_id: int,
    query_position: int,
    repetition: int,
    order: Sequence[str],
    method: _Method,
    measured: _Measured,
    reference: _Measured,
    exact: _Measured,
    query: np.ndarray,
    candidates: CandidateSet,
    base_prepare_wall_ns: int,
    prepared: _PreparedExperiment,
    k: int,
    records_by_key: Mapping[tuple[int, int], VectorRecord],
    distance_cache: dict[tuple[int, int], Fraction],
    audit_evidence: Sequence[Mapping[str, Any]] | None,
) -> dict[str, Any]:
    ids = _hit_ids(measured)
    reference_ids = _hit_ids(reference)
    exact_ids = _hit_ids(exact)
    exact_delta_ids = [
        identifier for identifier in exact_ids if identifier in prepared.delta_logical_ids
    ]
    reference_delta_ids = [
        identifier
        for identifier in reference_ids
        if identifier in prepared.delta_logical_ids
    ]
    coupled = method.kind in COUPLED_KINDS
    observed_lower: float | None = None
    observed_upper: float | None = None
    observed_nonnegative: bool | None = None
    if coupled:
        observed_lower, observed_upper, observed_nonnegative = _exact_observed_gap(
            query,
            measured,
            reference,
            records_by_key,
            distance_cache,
        )

    receipt = measured.receipt or {}
    certified_beta = receipt.get("certified_beta")
    requested_beta = (
        receipt.get("requested_beta")
        if measured.receipt is not None
        else method.requested_beta
    )
    violation_reasons: list[str] = []
    baseline_validation_reasons: list[str] = []
    if coupled and method.kind in CERTIFIED_KINDS:
        if measured.receipt is None:
            violation_reasons.append("certified_method_missing_receipt")
        required_receipt_fields = {
            "snapshot_id",
            "base_generation",
            "covered_commit_seq",
            "candidate_set_id_or_hash",
            "metric",
            "k",
            "requested_beta",
            "certified_beta",
            "certificate_status",
            "tau_returned",
            "min_skipped_lb",
            "raw_pending_scanned",
            "groups_scanned",
            "groups_skipped",
            "vectors_scanned",
            "visibility_rejections",
            "fallback_reason",
            "numeric_mode",
            "component_timings",
            "certificate_details",
        }
        missing_fields = sorted(required_receipt_fields.difference(receipt))
        if missing_fields:
            violation_reasons.append(
                "missing_receipt_fields:" + ",".join(missing_fields)
            )
        status = receipt.get("certificate_status")
        allowed_statuses = {
            "certified",
            "certified_no_skip",
            "full_scan_reference",
            "fallback_full_scan_duplicate_visible_logical_id",
            "not_applicable_population_below_k",
        }
        if status not in allowed_statuses:
            violation_reasons.append("missing_or_unknown_certificate_status")
        if not isinstance(receipt.get("numeric_mode"), str) or not receipt.get(
            "numeric_mode"
        ):
            violation_reasons.append("missing_numeric_mode")
        details = receipt.get("certificate_details")
        if not isinstance(details, dict):
            violation_reasons.append("missing_certificate_details")
        elif status != "not_applicable_population_below_k" and not details.get(
            "authoritative_gap"
        ):
            violation_reasons.append("missing_authoritative_gap")
        if receipt.get("candidate_set_id_or_hash") != candidates.candidate_hash:
            violation_reasons.append("candidate_set_hash_changed")
        if receipt.get("snapshot_id") != 0:
            violation_reasons.append("receipt_snapshot_mismatch")
        if receipt.get("base_generation") != prepared.base.generation_id:
            violation_reasons.append("receipt_base_generation_mismatch")
        if receipt.get("metric") != "L2" or receipt.get("k") != k:
            violation_reasons.append("receipt_metric_or_k_mismatch")
        if not isinstance(receipt.get("component_timings"), dict):
            violation_reasons.append("missing_component_timings")
        for count_field in (
            "raw_pending_scanned",
            "groups_scanned",
            "groups_skipped",
            "vectors_scanned",
            "visibility_rejections",
        ):
            value = receipt.get(count_field)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                violation_reasons.append(f"invalid_receipt_count:{count_field}")
        if status != "not_applicable_population_below_k" and certified_beta is None:
            violation_reasons.append("missing_certified_beta")
        if requested_beta is None:
            violation_reasons.append("missing_requested_beta")
        elif not math.isfinite(float(requested_beta)) or float(requested_beta) < 0.0:
            violation_reasons.append("invalid_requested_beta")
        if observed_nonnegative is False:
            violation_reasons.append("observed_gap_is_negative")
        if observed_lower is not None and observed_lower < 0.0:
            violation_reasons.append("observed_gap_lower_endpoint_is_negative")
        certificate_is_finite = certified_beta is not None and math.isfinite(
            float(certified_beta)
        )
        if certified_beta is not None and (
            not certificate_is_finite or float(certified_beta) < 0.0
        ):
            violation_reasons.append("invalid_certified_beta")
        if certificate_is_finite and requested_beta is not None and math.isfinite(
            float(requested_beta)
        ):
            if Fraction.from_float(float(certified_beta)) > Fraction.from_float(
                float(requested_beta)
            ):
                violation_reasons.append("certified_beta_exceeds_requested")
            if observed_upper is not None and Fraction.from_float(
                observed_upper
            ) > Fraction.from_float(float(certified_beta)):
                violation_reasons.append("observed_beta_exceeds_certificate")
        if float(requested_beta or 0.0) == 0.0 and _hit_keys(measured) != _hit_keys(reference):
            violation_reasons.append("beta_zero_id_mismatch")
    if method.kind == "delta_flat" and _hit_keys(measured) != _hit_keys(reference):
        baseline_validation_reasons.append(
            "optimized_delta_flat_differs_from_certified_same_c_reference"
        )
    if (
        method.kind == "full_flat_performance"
        and _hit_keys(measured) != _hit_keys(exact)
    ):
        baseline_validation_reasons.append(
            "optimized_full_flat_differs_from_exact_full_visible_truth"
        )

    groups_scanned = receipt.get("groups_scanned")
    groups_skipped = receipt.get("groups_skipped")
    vectors_scanned = receipt.get("vectors_scanned")
    if method.kind == "delta_flat":
        vectors_scanned = len(prepared.delta_records)
    elif method.kind in {"full_flat_performance", "full_exact_truth"}:
        vectors_scanned = len(prepared.full_flat.records)
    elif method.kind in {"delta_hnsw", "full_hnsw"}:
        # Faiss does not expose a reliable per-query visited-vector count here.
        vectors_scanned = None

    lb_values: list[float] | None = None
    if method.name == "group_pruning_beta0" and audit_evidence is not None:
        lb_values = [float(item["lb_lower"]) for item in audit_evidence]

    rank_position_diff = sum(
        left != right for left, right in zip(ids, exact_ids)
    ) + abs(len(ids) - len(exact_ids))
    return {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "config_hash": config_hash,
        "implementation_tree_sha256": implementation_hash,
        "evidence_role": evidence_role,
        "dataset_hash": dataset_hash,
        "split_id": split_id,
        "experiment_id": experiment_id,
        "query_id": int(query_id),
        "query_position": int(query_position),
        "repetition": int(repetition),
        "method": method.name,
        "method_kind": method.kind,
        "method_order": list(order),
        "method_order_position": (
            list(order).index(method.name) if method.name in order else None
        ),
        "timing_scope": (
            (
                "validation_truth_reused_from_run_cache"
                if measured.truth_cache_hit
                else "validation_truth_generation_once_per_unique_dataset_query_k"
            )
            if method.kind in TRUTH_KINDS
            else (
                "coupled_delta_micro_and_end_to_end"
                if coupled
                else "end_to_end_only"
            )
        ),
        "warmup_recorded": False,
        "threads": prepared.base.threads,
        "k": k,
        "candidate_set_id_or_hash": candidates.candidate_hash if coupled else None,
        "candidate_object_reused_within_query": coupled,
        "base_prepare_wall_ns": base_prepare_wall_ns if coupled else None,
        "base_prepare_measurement_scope": (
            "actual_same-query_search_this_repetition; content hash/key equality "
            "verified; original frozen C object remains reused"
            if coupled and method.kind not in TRUTH_KINDS
            else None
        ),
        "latency_wall_ns": measured.wall_ns,
        "micro_latency_ns": measured.micro_latency_ns,
        "end_to_end_latency_ns": measured.end_to_end_latency_ns,
        "end_to_end_composition": (
            "sum_of_disjoint_actual_base_search_and_delta_method_wall_intervals"
            if coupled
            else "one_contiguous_method_wall_interval"
        ),
        "component_timings_ns": measured.component_timings,
        "peak_rss_bytes": measured.peak_rss_bytes,
        "truth_cache_hit": measured.truth_cache_hit,
        "truth_cache_key": measured.truth_cache_key,
        "result_ids": ids,
        "result_keys": _hit_keys(measured),
        "result_distances_l2": [float(hit.distance) for hit in measured.hits],
        "result_sources": [hit.source for hit in measured.hits],
        "tau_returned_l2": None if not measured.hits else float(measured.hits[-1].distance),
        "reference_ids": reference_ids,
        "exact_full_visible_ids": exact_ids,
        "recall_at_k_exact_full_visible": _recall(ids, exact_ids),
        "recall_at_k_same_c_reference": _recall(ids, reference_ids),
        "id_symmetric_difference_exact": len(set(ids).symmetric_difference(exact_ids)),
        "rank_position_difference_exact": rank_position_diff,
        "delta_in_same_c_reference_topk": bool(reference_delta_ids),
        "delta_in_exact_full_visible_topk": bool(exact_delta_ids),
        "delta_exact_neighbor_count": len(exact_delta_ids),
        "delta_exact_neighbors_captured": len(set(ids).intersection(exact_delta_ids)),
        "delta_exact_neighbor_capture_rate": _recall(ids, exact_delta_ids),
        "requested_beta_l2": requested_beta,
        "certified_beta_l2": certified_beta,
        "observed_beta_lower_l2": observed_lower,
        "observed_beta_upper_l2": observed_upper,
        "certificate_status": receipt.get("certificate_status"),
        "contract_violation": bool(violation_reasons),
        "violation_reasons": violation_reasons,
        "baseline_validation_failure": bool(baseline_validation_reasons),
        "baseline_validation_reasons": baseline_validation_reasons,
        "comparison_valid": not violation_reasons and not baseline_validation_reasons,
        "groups_scanned": groups_scanned,
        "groups_skipped": groups_skipped,
        "vectors_scanned_or_read": vectors_scanned,
        "distance_evaluations": receipt.get("distance_evaluations"),
        "raw_pending_scanned": receipt.get("raw_pending_scanned"),
        "visibility_rejections": receipt.get("visibility_rejections"),
        "fallback": receipt.get("fallback_reason") is not None,
        "fallback_reason": receipt.get("fallback_reason"),
        "min_skipped_lb_l2": receipt.get("min_skipped_lb"),
        "lb_lower_values_l2": lb_values,
        "lb_audit_non_timed": (
            list(audit_evidence)
            if method.name == "group_pruning_beta0" and audit_evidence is not None
            else None
        ),
        "exact_boundary_rechecks": receipt.get("exact_boundary_rechecks"),
        "numeric_mode": receipt.get("numeric_mode"),
        "index_returned_candidate_count": measured.index_candidate_count,
        "hnsw_ef_search": method.ef_search,
        "base_hnsw_ef_search": prepared.base.ef_search,
        "beta_factor_of_validation_median": method.beta_factor,
        "receipt": measured.receipt,
    }


def _stable_order_seed(base_seed: int, experiment_id: str, query_id: int, repetition: int) -> int:
    digest = object_sha256(
        {
            "base_seed": base_seed,
            "experiment_id": experiment_id,
            "query_id": query_id,
            "repetition": repetition,
        }
    )
    return int(digest[:16], 16)


def _run_query(
    *,
    run_id: str,
    config_hash: str,
    implementation_hash: str,
    evidence_role: str,
    dataset_hash: str,
    split_id: str,
    experiment_id: str,
    query_id: int,
    query_position: int,
    query: np.ndarray,
    repetition_count: int,
    method_order_seed: int,
    methods: Sequence[_Method],
    prepared: _PreparedExperiment,
    k: int,
    candidate_count: int,
    collect_lb_audit: bool,
    exact_truth_cache: dict[str, _Measured],
) -> list[dict[str, Any]]:
    candidate_start = time.perf_counter_ns()
    candidates = prepared.engine.prepare_candidates(
        query, snapshot_id=0, k=k, candidate_count=candidate_count
    )
    base_prepare_wall_ns = time.perf_counter_ns() - candidate_start
    records_by_key = dict(prepared.delta_by_key)
    for record in candidates.records:
        prior = records_by_key.get(record.key)
        if prior is not None and prior.vector.tobytes(order="C") != record.vector.tobytes(
            order="C"
        ):
            raise RuntimeError("one version key maps to inconsistent C/Delta vectors")
        records_by_key[record.key] = record
    distance_cache: dict[tuple[int, int], Fraction] = {}

    truth_by_kind = {method.kind: method for method in methods if method.kind in TRUTH_KINDS}
    if set(truth_by_kind) != TRUTH_KINDS:
        raise AssertionError("benchmark requires both same-C and full-visible truth methods")
    timed_methods = [method for method in methods if method.kind not in TRUTH_KINDS]
    trials: list[tuple[int, int, list[_Method], dict[str, _Measured]]] = []
    for repetition in range(repetition_count):
        if repetition == 0:
            repetition_base_wall_ns = base_prepare_wall_ns
        else:
            repeated_base_start = time.perf_counter_ns()
            repeated_candidates = prepared.engine.prepare_candidates(
                query, snapshot_id=0, k=k, candidate_count=candidate_count
            )
            repetition_base_wall_ns = time.perf_counter_ns() - repeated_base_start
            if (
                repeated_candidates.candidate_hash != candidates.candidate_hash
                or tuple(record.key for record in repeated_candidates.records)
                != tuple(record.key for record in candidates.records)
            ):
                raise AssertionError("repeated E2E base search changed frozen C")
        ordered = list(timed_methods)
        random.Random(
            _stable_order_seed(
                method_order_seed, experiment_id, int(query_id), repetition
            )
        ).shuffle(ordered)
        measured: dict[str, _Measured] = {}
        for method in ordered:
            measured[method.name] = _measure(
                method,
                prepared,
                query,
                k=k,
                candidate_count=candidate_count,
                candidates=candidates,
                base_prepare_wall_ns=repetition_base_wall_ns,
                audit=False,
            )
        trials.append((repetition, repetition_base_wall_ns, ordered, measured))

    # Correctness references are generated exactly once per query, after every
    # randomized timed trial, so they neither receive artificial repetitions
    # nor prime only one measured method's caches.
    reference_method = truth_by_kind["certified_full"]
    exact_method = truth_by_kind["full_exact_truth"]
    reference = _measure(
        reference_method,
        prepared,
        query,
        k=k,
        candidate_count=candidate_count,
        candidates=candidates,
        base_prepare_wall_ns=base_prepare_wall_ns,
        audit=False,
    )
    exact_cache_key = object_sha256(
        {
            "dataset_hash": dataset_hash,
            "query_id": int(query_id),
            "query_position": query_position,
            "query_bytes_sha256": object_sha256(query.tolist()),
            "k": k,
            "truth": "all_visible_base_plus_delta_exact_v1",
        }
    )
    cached_exact = exact_truth_cache.get(exact_cache_key)
    if cached_exact is None:
        exact = _measure(
            exact_method,
            prepared,
            query,
            k=k,
            candidate_count=candidate_count,
            candidates=candidates,
            base_prepare_wall_ns=base_prepare_wall_ns,
            audit=False,
        )
        exact.truth_cache_key = exact_cache_key
        exact_truth_cache[exact_cache_key] = exact
    else:
        lookup_start = time.perf_counter_ns()
        cached_hits = cached_exact.hits
        lookup_ns = time.perf_counter_ns() - lookup_start
        exact = _Measured(
            hits=cached_hits,
            wall_ns=lookup_ns,
            micro_latency_ns=None,
            end_to_end_latency_ns=lookup_ns,
            component_timings={"truth_cache_lookup_ns": lookup_ns},
            receipt=None,
            index_candidate_count=cached_exact.index_candidate_count,
            peak_rss_bytes=_peak_rss_bytes(),
            truth_cache_hit=True,
            truth_cache_key=exact_cache_key,
        )

    # LB distributions and decisions are sampled outside all latency trials.
    # Member-key lists are already represented by the immutable build manifest,
    # so omit their O(Delta) repetition from per-query evidence.
    audit_evidence: list[dict[str, Any]] | None = None
    if collect_lb_audit:
        audited = prepared.engine.search_pruned(
            query,
            k=k,
            beta=0.0,
            snapshot_id=0,
            candidate_count=candidate_count,
            candidate_set=candidates,
            no_pruning=False,
            audit=True,
        )
        audit_evidence = [
            {key: value for key, value in item.items() if key != "member_version_keys"}
            for item in (audited.receipt.audit or ())
        ]
        beta0_measured = next(
            (
                measured[method.name]
                for _, _, _, measured in trials
                for method in timed_methods
                if method.name == "group_pruning_beta0"
            ),
            None,
        )
        if beta0_measured is None or _hit_keys(beta0_measured) != [
            [hit.logical_id, hit.version_id] for hit in audited.hits
        ]:
            raise AssertionError("non-timed LB audit disagrees with timed beta=0 search")

    rows: list[dict[str, Any]] = []
    for repetition, repetition_base_wall_ns, ordered, measured in trials:
        order_names = [method.name for method in ordered]
        for method in ordered:
            rows.append(
                _row_for_method(
                    run_id=run_id,
                    config_hash=config_hash,
                    implementation_hash=implementation_hash,
                    evidence_role=evidence_role,
                    dataset_hash=dataset_hash,
                    split_id=split_id,
                    experiment_id=experiment_id,
                    query_id=int(query_id),
                    query_position=query_position,
                    repetition=repetition,
                    order=order_names,
                    method=method,
                    measured=measured[method.name],
                    reference=reference,
                    exact=exact,
                    query=query,
                    candidates=candidates,
                    base_prepare_wall_ns=repetition_base_wall_ns,
                    prepared=prepared,
                    k=k,
                    records_by_key=records_by_key,
                    distance_cache=distance_cache,
                    audit_evidence=(
                        audit_evidence
                        if method.name == "group_pruning_beta0" and repetition == 0
                        else None
                    ),
                )
            )
    for truth_method, truth_measurement in (
        (reference_method, reference),
        (exact_method, exact),
    ):
        rows.append(
            _row_for_method(
                run_id=run_id,
                config_hash=config_hash,
                implementation_hash=implementation_hash,
                evidence_role=evidence_role,
                dataset_hash=dataset_hash,
                split_id=split_id,
                experiment_id=experiment_id,
                query_id=int(query_id),
                query_position=query_position,
                repetition=-1,
                order=(),
                method=truth_method,
                measured=truth_measurement,
                reference=reference,
                exact=exact,
                query=query,
                candidates=candidates,
                base_prepare_wall_ns=base_prepare_wall_ns,
                prepared=prepared,
                k=k,
                records_by_key=records_by_key,
                distance_cache=distance_cache,
                audit_evidence=None,
            )
        )
    return rows


def _warmup(
    split: DatasetSplit,
    specification: Mapping[str, Any],
    prepared: _PreparedExperiment,
    methods: Sequence[_Method],
    count: int,
) -> int:
    if count <= 0:
        return 0
    queries = split.test_queries[: min(count, len(split.test_queries))]
    k = int(specification["k"])
    candidate_count = int(specification["candidate_count"])
    for query in queries:
        start = time.perf_counter_ns()
        candidates = prepared.engine.prepare_candidates(
            query, snapshot_id=0, k=k, candidate_count=candidate_count
        )
        base_wall = time.perf_counter_ns() - start
        for method in methods:
            if method.kind in TRUTH_KINDS:
                continue
            _measure(
                method,
                prepared,
                query,
                k=k,
                candidate_count=candidate_count,
                candidates=candidates,
                base_prepare_wall_ns=base_wall,
                audit=False,
            )
    return len(queries)


def _sample_positions(total: int, count: int, identity: Mapping[str, Any]) -> list[int]:
    """Choose a deterministic, recorded sample without taking a prefix."""

    take = min(max(0, int(count)), total)
    if take == 0:
        return []
    seed = int(object_sha256(identity)[:16], 16)
    return sorted(random.Random(seed).sample(range(total), take))


def _independent_oracle_rows(
    split: DatasetSplit,
    specification: Mapping[str, Any],
    prepared: _PreparedExperiment,
    *,
    count: int,
    max_population: int,
    dataset_hash: str,
    split_id: str,
) -> list[dict[str, Any]]:
    """Systematically sample same-C results with the independent Fraction oracle."""

    positions = _sample_positions(
        len(split.test_queries),
        count,
        {
            "purpose": "independent_fraction_oracle",
            "experiment_id": specification["id"],
            "dataset_hash": dataset_hash,
        },
    )
    rows: list[dict[str, Any]] = []
    k = int(specification["k"])
    candidate_count = int(specification["candidate_count"])
    for position in positions:
        query = split.test_queries[position]
        candidates = prepared.engine.prepare_candidates(
            query, snapshot_id=0, k=k, candidate_count=candidate_count
        )
        population_upper = len(candidates.records) + len(prepared.delta_records)
        common = {
            "schema_version": SCHEMA_VERSION,
            "experiment_id": str(specification["id"]),
            "dataset_hash": dataset_hash,
            "split_id": split_id,
            "query_position": position,
            "query_id": int(split.test_query_ids[position]),
            "candidate_set_id_or_hash": candidates.candidate_hash,
            "population_upper_before_logical_id_deduplication": population_upper,
            "max_population": max_population,
            "selection": "deterministic_hash_seeded_sample_without_replacement",
        }
        if population_upper > max_population:
            rows.append(
                {
                    **common,
                    "status": "skipped_population_above_explicit_limit",
                    "match": None,
                    "oracle_wall_ns": None,
                    "production_wall_ns": None,
                }
            )
            continue
        oracle_start = time.perf_counter_ns()
        oracle = reference_topk(
            query,
            candidates,
            raw_records=prepared.raw_records,
            grouped_records=prepared.grouped_records,
            snapshot_id=0,
            k=k,
        )
        oracle_wall_ns = time.perf_counter_ns() - oracle_start
        production_start = time.perf_counter_ns()
        production = prepared.engine.search_full_scan(
            query,
            k=k,
            beta=0.0,
            snapshot_id=0,
            candidate_count=candidate_count,
            candidate_set=candidates,
            audit=False,
        )
        production_wall_ns = time.perf_counter_ns() - production_start
        production_keys = tuple(hit.key for hit in production.hits)
        match = production_keys == oracle.keys
        rows.append(
            {
                **common,
                "status": "executed_match" if match else "executed_mismatch",
                "match": match,
                "oracle_wall_ns": oracle_wall_ns,
                "production_wall_ns": production_wall_ns,
                "oracle_result_keys": [list(key) for key in oracle.keys],
                "production_result_keys": [list(key) for key in production_keys],
                "oracle_tau_squared_fraction": (
                    None
                    if oracle.tau_squared is None
                    else f"{oracle.tau_squared.numerator}/{oracle.tau_squared.denominator}"
                ),
                "production_receipt": production.receipt.to_dict(),
            }
        )
    return rows


def _throughput_rows(
    split: DatasetSplit,
    specification: Mapping[str, Any],
    prepared: _PreparedExperiment,
    methods: Sequence[_Method],
    *,
    count: int,
    repetitions: int,
    dataset_hash: str,
    split_id: str,
) -> list[dict[str, Any]]:
    """Measure actual single-thread sequential batches without artifact writes."""

    positions = _sample_positions(
        len(split.test_queries),
        count,
        {
            "purpose": "actual_batch_throughput",
            "experiment_id": specification["id"],
            "dataset_hash": dataset_hash,
        },
    )
    if not positions:
        return []
    queries = [split.test_queries[position] for position in positions]
    query_ids = [int(split.test_query_ids[position]) for position in positions]
    k = int(specification["k"])
    candidate_count = int(specification["candidate_count"])
    candidates = [
        prepared.engine.prepare_candidates(
            query, snapshot_id=0, k=k, candidate_count=candidate_count
        )
        for query in queries
    ]
    timed_methods = [method for method in methods if method.kind not in TRUTH_KINDS]
    rows: list[dict[str, Any]] = []
    for repetition in range(repetitions):
        ordered = list(timed_methods)
        random.Random(
            _stable_order_seed(
                0x5450524F55474850,
                str(specification["id"]),
                query_ids[0],
                repetition,
            )
        ).shuffle(ordered)
        order_names = [method.name for method in ordered]
        for method in ordered:
            result_keys: list[list[list[int]]] = []
            micro_start = time.perf_counter_ns()
            for query, frozen in zip(queries, candidates):
                measured = _measure(
                    method,
                    prepared,
                    query,
                    k=k,
                    candidate_count=candidate_count,
                    candidates=frozen,
                    base_prepare_wall_ns=0,
                    audit=False,
                )
                result_keys.append(_hit_keys(measured))
            micro_wall_ns = time.perf_counter_ns() - micro_start

            if method.kind in COUPLED_KINDS:
                e2e_keys: list[list[list[int]]] = []
                regenerated_hashes: list[str] = []
                end_start = time.perf_counter_ns()
                for query in queries:
                    regenerated = prepared.engine.prepare_candidates(
                        query, snapshot_id=0, k=k, candidate_count=candidate_count
                    )
                    regenerated_hashes.append(regenerated.candidate_hash)
                    measured = _measure(
                        method,
                        prepared,
                        query,
                        k=k,
                        candidate_count=candidate_count,
                        candidates=regenerated,
                        base_prepare_wall_ns=0,
                        audit=False,
                    )
                    e2e_keys.append(_hit_keys(measured))
                end_wall_ns = time.perf_counter_ns() - end_start
                if regenerated_hashes != [item.candidate_hash for item in candidates]:
                    raise AssertionError("throughput rerun changed frozen C content")
                if e2e_keys != result_keys:
                    raise AssertionError("throughput micro/E2E batches returned different keys")
            else:
                end_wall_ns = micro_wall_ns

            rows.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "experiment_id": str(specification["id"]),
                    "dataset_hash": dataset_hash,
                    "split_id": split_id,
                    "method": method.name,
                    "method_kind": method.kind,
                    "repetition": repetition,
                    "method_batch_order": order_names,
                    "method_batch_order_position": order_names.index(method.name),
                    "query_positions": positions,
                    "query_ids": query_ids,
                    "queries": len(queries),
                    "micro_batch_wall_ns": micro_wall_ns,
                    "micro_actual_batch_qps": len(queries) * 1e9 / micro_wall_ns,
                    "end_to_end_batch_wall_ns": end_wall_ns,
                    "end_to_end_actual_batch_qps": len(queries) * 1e9 / end_wall_ns,
                    "coupled_candidate_hashes": (
                        [item.candidate_hash for item in candidates]
                        if method.kind in COUPLED_KINDS
                        else None
                    ),
                    "result_keys_sha256": object_sha256(result_keys),
                    "timing_scope": (
                        "single_thread_sequential_batch; micro reuses precomputed C; "
                        "E2E regenerates C; excludes JSON/artifact writes and truth checks"
                    ),
                    "threads": prepared.base.threads,
                    "base_hnsw_ef_search": prepared.base.ef_search,
                    "hnsw_ef_search": method.ef_search,
                }
            )
    return rows


def _split_payload(
    split: DatasetSplit, experiment_id: str, dataset_hash: str
) -> dict[str, Any]:
    id_payload = {
        "base_ids": split.base_ids.tolist(),
        "delta_ids": split.delta_ids.tolist(),
        "validation_query_ids": split.validation_query_ids.tolist(),
        "test_query_ids": split.test_query_ids.tolist(),
    }
    # An ID layout is not a split identity on its own: two generated datasets
    # with different vectors/seeds deliberately reuse the same ID ranges.
    # Bind the content hash without changing the existing ``ids`` object.
    identity_payload = {"dataset_hash": dataset_hash, "ids": id_payload}
    return {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": experiment_id,
        "dataset_name": split.name,
        "dataset_hash": dataset_hash,
        "split_id": object_sha256(identity_payload),
        "ids": id_payload,
        "metadata": dict(split.metadata),
        "shapes": {
            "base": list(split.base.shape),
            "delta": list(split.delta.shape),
            "validation_queries": list(split.validation_queries.shape),
            "test_queries": list(split.test_queries.shape),
        },
        "dtypes": {
            "base": str(split.base.dtype),
            "delta": str(split.delta.dtype),
            "queries": str(split.test_queries.dtype),
        },
    }


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _load_or_create_checkpoint(
    run_dir: Path, run_id: str, config_hash: str, implementation_hash: str
) -> dict[str, Any]:
    path = run_dir / "checkpoint.json"
    if path.exists():
        checkpoint = _read_json(path)
        if checkpoint.get("run_id") != run_id or checkpoint.get("config_hash") != config_hash:
            raise RuntimeError("checkpoint identity does not match the requested run")
        if checkpoint.get("implementation_tree_sha256") != implementation_hash:
            raise RuntimeError(
                "implementation changed since checkpoint; refusing mixed-code resume"
            )
        if not isinstance(checkpoint.get("completed_blocks"), dict):
            raise RuntimeError("checkpoint completed_blocks is malformed")
        return checkpoint
    checkpoint = {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "config_hash": config_hash,
        "implementation_tree_sha256": implementation_hash,
        "completed_blocks": {},
    }
    atomic_write_json(path, checkpoint)
    return checkpoint


def _verify_completed_block(run_dir: Path, entry: Mapping[str, Any]) -> None:
    location = run_dir / str(entry["path"])
    if not location.is_file():
        raise RuntimeError(f"checkpoint shard is missing: {location}")
    observed = file_sha256(location)
    if observed != entry.get("sha256"):
        raise RuntimeError(f"checkpoint shard checksum mismatch: {location}")
    with location.open("r", encoding="utf-8") as handle:
        rows = sum(1 for line in handle if line.strip())
    if rows != int(entry.get("rows", -1)):
        raise RuntimeError(f"checkpoint shard row count mismatch: {location}")


def _adopt_orphan_block(
    path: Path,
    *,
    expected_rows: int,
    experiment_id: str,
    start: int,
    stop: int,
) -> dict[str, Any]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    if len(rows) != expected_rows:
        raise RuntimeError(f"orphan shard has unexpected row count: {path}")
    if any(
        row.get("experiment_id") != experiment_id
        or not start <= int(row.get("query_position", -1)) < stop
        for row in rows
    ):
        raise RuntimeError(f"orphan shard has unexpected query identity: {path}")
    return {
        "path": str(path.parent.name + "/" + path.name),
        "sha256": file_sha256(path),
        "rows": len(rows),
        "adopted_after_checkpoint_gap": True,
    }


def _total_blocks(config: Mapping[str, Any]) -> int:
    block_size = int(config["query_block_size"])
    return sum(
        math.ceil(int(experiment["dataset"]["n_test"]) / block_size)
        for experiment in config["experiments"]
    )


def run_benchmark(
    config: Mapping[str, Any] | str | Path,
    *,
    resume: bool = False,
    run_id: str | None = None,
    stop_after_blocks: int | None = None,
) -> RunOutcome:
    """Execute all experiments and preserve a resumable per-query raw run.

    ``stop_after_blocks`` is a controlled interruption hook used by the
    checkpoint test and by operators validating resume behavior.  Such a run
    is explicitly left incomplete and is never labelled successful.
    """

    effective = load_json_config(config) if isinstance(config, (str, Path)) else _json_copy(config)
    for experiment in effective.get("experiments", []):
        if isinstance(experiment, dict):
            experiment.setdefault("base_hnsw_ef_search", 128)
    _validate_config(effective)
    if stop_after_blocks is not None and stop_after_blocks <= 0:
        raise BenchmarkConfigurationError("stop_after_blocks must be positive")
    threads = int(effective.get("threads", 1))
    for variable in THREAD_VARIABLES:
        os.environ[variable] = str(threads)
    faiss.omp_set_num_threads(threads)
    config_hash = object_sha256(effective)
    implementation_hash = implementation_tree_sha256(REPOSITORY_ROOT)
    run_dir, selected_id, existed = _select_run_dir(
        effective, config_hash, resume=resume, run_id=run_id
    )
    run_dir.mkdir(parents=True, exist_ok=existed)
    (run_dir / "raw").mkdir(exist_ok=True)
    (run_dir / "splits").mkdir(exist_ok=True)
    (run_dir / "validation").mkdir(exist_ok=True)
    (run_dir / "oracle_checks").mkdir(exist_ok=True)
    (run_dir / "throughput").mkdir(exist_ok=True)

    config_path = run_dir / "effective_config.json"
    if config_path.exists():
        if object_sha256(_read_json(config_path)) != config_hash:
            raise RuntimeError("existing effective config does not match config hash")
    else:
        atomic_write_json(config_path, effective)
    manifest_path = run_dir / "run_manifest.json"
    if manifest_path.exists():
        manifest = _read_json(manifest_path)
        if manifest.get("config_hash") != config_hash:
            raise RuntimeError("run manifest config hash mismatch")
        if manifest.get("implementation_tree_sha256") != implementation_hash:
            raise RuntimeError(
                "implementation changed since run creation; refusing mixed-code resume"
            )
        resume_events = list(manifest.get("resume_events", []))
        resume_events.append(
            {
                "resumed_at_utc": datetime.now(timezone.utc).isoformat(),
                "environment": collect_environment(threads),
                "completed_blocks_on_entry": len(
                    _read_json(run_dir / "checkpoint.json").get("completed_blocks", {})
                )
                if (run_dir / "checkpoint.json").exists()
                else 0,
            }
        )
        manifest["resume_events"] = resume_events
        manifest["status"] = "running_resumed"
        atomic_write_json(manifest_path, manifest)
    else:
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "run_id": selected_id,
            "run_name": effective["run_name"],
            "config_hash": config_hash,
            "implementation_tree_sha256": implementation_hash,
            "evidence_role": str(effective.get("evidence_role", "calibration")),
            "status": "running",
            "started_at_utc": datetime.now(timezone.utc).isoformat(),
            "environment": collect_environment(threads),
            "dataset_hashes": {},
            "split_ids": {},
            "warmup": {},
            "experiment_repetitions": {},
            "failures": [],
            "measurement_notes": {
                "distance_metric": "ordinary L2; Faiss squared-L2 is square-rooted",
                "candidate_policy": "one frozen C object per query, reused across coupled methods and repetitions",
                "timing_policy": "coupled Delta micro excludes base preparation; end-to-end includes it",
                "truth_policy": (
                    "certified same-C and exact full-visible truth are generated once per "
                    "query after timed trials, not repeated as performance methods; "
                    "exact full-visible truth is reused within this process across "
                    "experiments with identical dataset/query/k identities"
                ),
                "audit_policy": "beta=0 LB decision audit runs once after timed trials",
                "optimized_flat_policy": (
                    "one Faiss exhaustive scan plus exact rerank of returned overfetch; "
                    "checked against separately generated certified truth"
                ),
                "method_order": "deterministically shuffled for every query/repetition",
                "p99_note": "1,000 test queries provide only a finite empirical tail sample",
            },
        }
        atomic_write_json(manifest_path, manifest)
    manifest.setdefault("experiment_repetitions", {})
    checkpoint = _load_or_create_checkpoint(
        run_dir, selected_id, config_hash, implementation_hash
    )
    total_blocks = _total_blocks(effective)
    newly_completed = 0
    build_path = run_dir / "build_manifest.json"
    build_all: dict[str, Any] = _read_json(build_path) if build_path.exists() else {}
    # Several matrix experiments intentionally share the same immutable dataset,
    # test query and k.  The mathematical all-visible exact truth is independent
    # of index/search parameters, so retain it once for this process.  The cache
    # key includes the dataset and query content; it is deliberately not restored
    # across resume processes, where recomputation is safer than trusting memory-
    # only state that cannot be checksummed in the checkpoint.
    exact_truth_cache: dict[str, _Measured] = {}

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
                prior = _read_json(split_path)
                if prior.get("dataset_hash") != dataset_hash or prior.get("split_id") != split_id:
                    raise RuntimeError(f"dataset/split changed while resuming {experiment_id}")
            else:
                atomic_write_json(split_path, split_payload)
            manifest["dataset_hashes"][experiment_id] = dataset_hash
            manifest["split_ids"][experiment_id] = split_id
            atomic_write_json(manifest_path, manifest)

            prepared = _prepare_experiment(split, specification, threads)
            prepared.build_manifest.update(
                {
                    "experiment_id": experiment_id,
                    "dataset_hash": dataset_hash,
                    "split_id": split_id,
                }
            )
            if experiment_id in build_all:
                primary = build_all[experiment_id]
                rebuilds = list(primary.get("resume_rebuilds", []))
                rebuilds.append(
                    {
                        **prepared.build_manifest,
                        "rebuilt_at_utc": datetime.now(timezone.utc).isoformat(),
                        "reason": "resume_process_reconstructed_immutable_indexes",
                    }
                )
                primary["resume_rebuilds"] = rebuilds
                primary["raw_blocks_may_span_deterministic_rebuilds"] = True
            else:
                build_all[experiment_id] = prepared.build_manifest
            atomic_write_json(build_path, build_all)

            validation_path = run_dir / "validation" / f"{safe_id}.json"
            if validation_path.exists():
                beta_selection = _read_json(validation_path)
                if beta_selection.get("dataset_hash") != dataset_hash:
                    raise RuntimeError(f"validation selection hash changed for {experiment_id}")
            else:
                beta_selection = _validation_betas(split, specification, prepared)
                beta_selection.update(
                    {
                        "schema_version": SCHEMA_VERSION,
                        "experiment_id": experiment_id,
                        "dataset_hash": dataset_hash,
                        "split_id": split_id,
                    }
                )
                atomic_write_json(validation_path, beta_selection)
            methods = _methods(specification, beta_selection)
            experiment_repetitions = int(
                specification.get("repetitions", effective["repetitions"])
            )
            manifest["experiment_repetitions"][experiment_id] = (
                experiment_repetitions
            )
            lb_audit_positions = set(
                _sample_positions(
                    len(split.test_queries),
                    int(effective.get("lb_audit_queries", 0)),
                    {
                        "purpose": "non_timed_lb_audit",
                        "experiment_id": experiment_id,
                        "dataset_hash": dataset_hash,
                    },
                )
            )
            warmed = _warmup(
                split,
                specification,
                prepared,
                methods,
                int(effective.get("warmup_queries", 0)),
            )
            warmup_entry = {
                "queries": warmed,
                "methods_per_query": len(
                    [method for method in methods if method.kind not in TRUTH_KINDS]
                ),
                "truth_methods_excluded": sorted(
                    method.name for method in methods if method.kind in TRUTH_KINDS
                ),
                "measurement_repetitions": experiment_repetitions,
                "recorded_in_raw": False,
                "rerun_on_resume": True,
                "non_timed_lb_audit_query_positions": sorted(lb_audit_positions),
            }
            prior_warmup = manifest["warmup"].get(experiment_id)
            if prior_warmup is not None:
                sessions = list(prior_warmup.get("resume_sessions", []))
                sessions.append(
                    {
                        **warmup_entry,
                        "warmed_at_utc": datetime.now(timezone.utc).isoformat(),
                    }
                )
                prior_warmup["resume_sessions"] = sessions
            else:
                manifest["warmup"][experiment_id] = warmup_entry
            atomic_write_json(manifest_path, manifest)

            block_size = int(effective["query_block_size"])
            repetitions = experiment_repetitions
            for start in range(0, len(split.test_queries), block_size):
                stop = min(len(split.test_queries), start + block_size)
                key = f"{safe_id}:{start:08d}:{stop:08d}"
                part_name = f"{safe_id}.q{start:08d}-{stop:08d}.jsonl"
                part_path = run_dir / "raw" / part_name
                entry = checkpoint["completed_blocks"].get(key)
                if entry is not None:
                    _verify_completed_block(run_dir, entry)
                    continue
                timed_count = len(
                    [method for method in methods if method.kind not in TRUTH_KINDS]
                )
                expected_rows = (stop - start) * (
                    repetitions * timed_count + len(TRUTH_KINDS)
                )
                if part_path.exists():
                    adopted = _adopt_orphan_block(
                        part_path,
                        expected_rows=expected_rows,
                        experiment_id=experiment_id,
                        start=start,
                        stop=stop,
                    )
                    checkpoint["completed_blocks"][key] = adopted
                    atomic_write_json(run_dir / "checkpoint.json", checkpoint)
                    newly_completed += 1
                else:
                    rows: list[dict[str, Any]] = []
                    for position in range(start, stop):
                        rows.extend(
                            _run_query(
                                run_id=selected_id,
                                config_hash=config_hash,
                                implementation_hash=implementation_hash,
                                evidence_role=str(
                                    effective.get("evidence_role", "calibration")
                                ),
                                dataset_hash=dataset_hash,
                                split_id=split_id,
                                experiment_id=experiment_id,
                                query_id=int(split.test_query_ids[position]),
                                query_position=position,
                                query=split.test_queries[position],
                                repetition_count=repetitions,
                                method_order_seed=int(effective.get("method_order_seed", 0)),
                                methods=methods,
                                prepared=prepared,
                                k=int(specification["k"]),
                                candidate_count=int(specification["candidate_count"]),
                                collect_lb_audit=position in lb_audit_positions,
                                exact_truth_cache=exact_truth_cache,
                            )
                        )
                    atomic_write_jsonl(part_path, rows)
                    entry = {
                        "path": f"raw/{part_name}",
                        "sha256": file_sha256(part_path),
                        "rows": len(rows),
                        "experiment_id": experiment_id,
                        "query_start": start,
                        "query_stop": stop,
                        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
                        "adopted_after_checkpoint_gap": False,
                    }
                    checkpoint["completed_blocks"][key] = entry
                    atomic_write_json(run_dir / "checkpoint.json", checkpoint)
                    newly_completed += 1
                manifest["completed_blocks"] = len(checkpoint["completed_blocks"])
                manifest["total_blocks"] = total_blocks
                manifest["peak_rss_bytes"] = _peak_rss_bytes()
                atomic_write_json(manifest_path, manifest)
                if stop_after_blocks is not None and newly_completed >= stop_after_blocks:
                    manifest["status"] = "incomplete_controlled_stop"
                    manifest["stopped_at_utc"] = datetime.now(timezone.utc).isoformat()
                    atomic_write_json(manifest_path, manifest)
                    return RunOutcome(
                        run_dir,
                        selected_id,
                        False,
                        len(checkpoint["completed_blocks"]),
                        total_blocks,
                    )

            throughput_path = run_dir / "throughput" / f"{safe_id}.json"
            if throughput_path.exists():
                throughput_payload = _read_json(throughput_path)
                if (
                    throughput_payload.get("dataset_hash") != dataset_hash
                    or throughput_payload.get("split_id") != split_id
                    or throughput_payload.get("implementation_tree_sha256")
                    != implementation_hash
                ):
                    raise RuntimeError(
                        f"throughput artifact identity mismatch for {experiment_id}"
                    )
            else:
                throughput_rows = _throughput_rows(
                    split,
                    specification,
                    prepared,
                    methods,
                    count=int(effective.get("throughput_queries", 0)),
                    repetitions=int(effective.get("throughput_repetitions", 1)),
                    dataset_hash=dataset_hash,
                    split_id=split_id,
                )
                throughput_payload = {
                    "schema_version": SCHEMA_VERSION,
                    "experiment_id": experiment_id,
                    "dataset_hash": dataset_hash,
                    "split_id": split_id,
                    "implementation_tree_sha256": implementation_hash,
                    "rows": throughput_rows,
                }
                atomic_write_json(throughput_path, throughput_payload)

            oracle_path = run_dir / "oracle_checks" / f"{safe_id}.json"
            if oracle_path.exists():
                oracle_payload = _read_json(oracle_path)
                if (
                    oracle_payload.get("dataset_hash") != dataset_hash
                    or oracle_payload.get("split_id") != split_id
                    or oracle_payload.get("implementation_tree_sha256")
                    != implementation_hash
                ):
                    raise RuntimeError(
                        f"oracle-check artifact identity mismatch for {experiment_id}"
                    )
            else:
                oracle_rows = _independent_oracle_rows(
                    split,
                    specification,
                    prepared,
                    count=int(effective.get("independent_oracle_queries", 0)),
                    max_population=int(
                        effective.get("independent_oracle_max_population", 0)
                    ),
                    dataset_hash=dataset_hash,
                    split_id=split_id,
                )
                oracle_payload = {
                    "schema_version": SCHEMA_VERSION,
                    "experiment_id": experiment_id,
                    "dataset_hash": dataset_hash,
                    "split_id": split_id,
                    "implementation_tree_sha256": implementation_hash,
                    "rows": oracle_rows,
                }
                atomic_write_json(oracle_path, oracle_payload)
            manifest.setdefault("throughput_artifacts", {})[experiment_id] = {
                "path": f"throughput/{safe_id}.json",
                "sha256": file_sha256(throughput_path),
                "rows": len(throughput_payload.get("rows", [])),
            }
            manifest.setdefault("oracle_check_artifacts", {})[experiment_id] = {
                "path": f"oracle_checks/{safe_id}.json",
                "sha256": file_sha256(oracle_path),
                "rows": len(oracle_payload.get("rows", [])),
                "executed": sum(
                    str(row.get("status", "")).startswith("executed_")
                    for row in oracle_payload.get("rows", [])
                ),
                "skipped": sum(
                    str(row.get("status", "")).startswith("skipped_")
                    for row in oracle_payload.get("rows", [])
                ),
            }
            atomic_write_json(manifest_path, manifest)
            if any(row.get("match") is False for row in oracle_payload.get("rows", [])):
                raise AssertionError(
                    f"independent Fraction oracle mismatch for {experiment_id}"
                )

        if len(checkpoint["completed_blocks"]) != total_blocks:
            raise RuntimeError(
                "internal block count mismatch; run is deliberately left incomplete"
            )
        # Re-verify every committed shard before declaring completion.
        for entry in checkpoint["completed_blocks"].values():
            _verify_completed_block(run_dir, entry)
        manifest["status"] = "completed"
        manifest["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
        manifest["completed_blocks"] = len(checkpoint["completed_blocks"])
        manifest["total_blocks"] = total_blocks
        manifest["peak_rss_bytes"] = _peak_rss_bytes()
        atomic_write_json(manifest_path, manifest)
        completion = {
            "schema_version": SCHEMA_VERSION,
            "run_id": selected_id,
            "config_hash": config_hash,
            "implementation_tree_sha256": implementation_hash,
            "completed_at_utc": manifest["completed_at_utc"],
            "checkpoint_sha256": file_sha256(run_dir / "checkpoint.json"),
            "run_manifest_sha256": file_sha256(manifest_path),
            "raw_shards": len(checkpoint["completed_blocks"]),
        }
        atomic_write_json(run_dir / "COMPLETED.json", completion)
        return RunOutcome(
            run_dir,
            selected_id,
            True,
            len(checkpoint["completed_blocks"]),
            total_blocks,
        )
    except BaseException as error:
        manifest["status"] = "failed_incomplete"
        manifest["failed_at_utc"] = datetime.now(timezone.utc).isoformat()
        failures = list(manifest.get("failures", []))
        failures.append(
            {
                "type": type(error).__name__,
                "message": str(error),
                "completed_blocks": len(checkpoint["completed_blocks"]),
            }
        )
        manifest["failures"] = failures
        atomic_write_json(manifest_path, manifest)
        raise


__all__ = [
    "BenchmarkConfigurationError",
    "CompletedRunError",
    "RunOutcome",
    "apply_resource_limits",
    "collect_environment",
    "load_json_config",
    "run_benchmark",
]

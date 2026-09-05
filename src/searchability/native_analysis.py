"""Fail-closed analysis primitives for the Issue #3 native recheck.

The native runner writes one row per query, method and repetition.  This
module deliberately depends on method *roles* (``F``, ``A`` and ``P``), not on
one implementation's display names.  It also keeps baseline quality failures
in every timing aggregate: an ``A`` mismatch is a measured quality outcome,
not permission to remove a fast or slow observation.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import statistics
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .artifacts import file_sha256, object_sha256
from .final_policy import (
    FinalPolicyError,
    issue_fixed_decision_values,
    validate_final_policy,
    validate_issue_fixed_decision_values,
)


NATIVE_ANALYSIS_SCHEMA_VERSION = 1
MIN_BOOTSTRAP_RESAMPLES = 2_000
DEFAULT_BOOTSTRAP_SEED = 6_202_052
METRIC_FIELDS: dict[str, tuple[str, ...]] = {
    "micro": ("micro_latency_ns", "delta_micro_ns", "latency_micro_ns"),
    "composed_e2e": (
        "composed_e2e_latency_ns",
        "composed_e2e_ns",
        "end_to_end_latency_ns",
    ),
    "api_wall": (
        "api_wall_latency_ns",
        "api_wall_e2e_ns",
        "actual_api_wall_ns",
    ),
}

_DEFAULT_METHOD_ROLES = {
    "F": {
        "F",
        "native_full_delta_scan",
        "native_certified_full_delta_scan",
        "native_full_scan",
    },
    "A": {
        "A",
        "faiss_delta_flat",
        "optimized_faiss_delta_flat",
        "delta_flat_full_scan",
    },
    "P": {
        "P",
        "native_group_pruning",
        "native_group_pruning_beta0",
    },
    "N": {"N", "native_group_no_pruning", "native_no_pruning"},
    "O": {"O", "old_same_c_reference", "certified_full_delta_reference"},
    "P-old": {"P-old", "python_group_pruning", "group_pruning_beta0"},
}


class NativeEvidenceError(ValueError):
    """Raised when saved native evidence is malformed or fails identity checks."""


@dataclass(frozen=True, slots=True)
class LoadedNativeEvidence:
    rows: tuple[dict[str, Any], ...]
    runs: tuple[dict[str, Any], ...]
    excluded_incomplete_runs: tuple[dict[str, Any], ...]
    experiment_metadata: Mapping[str, Mapping[str, Any]]


def _read_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise NativeEvidenceError(f"cannot read JSON object {path}") from error
    if not isinstance(value, dict):
        raise NativeEvidenceError(f"expected JSON object: {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise NativeEvidenceError(
                        f"non-object JSONL row {path}:{line_number}"
                    )
                rows.append(value)
    except (OSError, json.JSONDecodeError) as error:
        raise NativeEvidenceError(f"invalid JSONL evidence {path}") from error
    return rows


def _first(row: Mapping[str, Any], names: Iterable[str]) -> Any:
    for name in names:
        if name in row and row[name] is not None:
            return row[name]
    return None


def _method_name(row: Mapping[str, Any]) -> str:
    value = _first(row, ("method", "method_name", "method_id"))
    if value is None or not str(value):
        raise NativeEvidenceError("native raw row is missing a method name")
    return str(value)


def method_role(
    row: Mapping[str, Any],
    method_roles: Mapping[str, str] | None = None,
) -> str:
    explicit = _first(row, ("method_role", "method_symbol"))
    if explicit is not None:
        rendered = str(explicit)
        normalized = rendered.upper().replace("_", "-")
        if normalized == "P-OLD":
            return "P-old"
        if normalized in {"O", "F", "A", "N", "P"}:
            return normalized
        # Reference-only A variants must not accidentally become the gate's A.
        return rendered

    name = _method_name(row)
    if method_roles is not None and name in method_roles:
        return str(method_roles[name])
    for role, aliases in _DEFAULT_METHOD_ROLES.items():
        if name in aliases:
            return role
    if name.startswith("native_group_pruning"):
        return "P"
    raise NativeEvidenceError(
        f"method {name!r} has no method_role and no configured role mapping"
    )


def _experiment_id(row: Mapping[str, Any]) -> str:
    value = row.get("experiment_id")
    if value is None or not str(value):
        raise NativeEvidenceError("native raw row is missing experiment_id")
    return str(value)


def _session_id(row: Mapping[str, Any]) -> str:
    value = _first(row, ("session_id", "process_session_id", "process_id"))
    return "unspecified" if value is None else str(value)


def _run_id(row: Mapping[str, Any]) -> str:
    value = row.get("run_id")
    return "unmanifested" if value is None else str(value)


def _query_identity(row: Mapping[str, Any]) -> tuple[str, int]:
    if "query_id" not in row:
        raise NativeEvidenceError("native raw row is missing query_id")
    try:
        return str(row["query_id"]), int(row.get("query_position", -1))
    except (TypeError, ValueError) as error:
        raise NativeEvidenceError("invalid query identity in native raw row") from error


def _metric_value(row: Mapping[str, Any], metric: str) -> float | None:
    try:
        fields = METRIC_FIELDS[metric]
    except KeyError as error:
        raise NativeEvidenceError(f"unknown latency metric {metric!r}") from error
    value = _first(row, fields)
    if value is None:
        return None
    try:
        converted = float(value)
    except (TypeError, ValueError) as error:
        raise NativeEvidenceError(f"non-numeric {metric} latency") from error
    if not math.isfinite(converted) or converted <= 0.0:
        raise NativeEvidenceError(f"{metric} latency must be finite and positive")
    return converted


def _optional_bool(row: Mapping[str, Any], names: Iterable[str]) -> bool | None:
    value = _first(row, names)
    if value is None:
        return None
    if not isinstance(value, bool):
        raise NativeEvidenceError(f"{next(iter(names))} must be boolean when present")
    return value


def _requested_beta(row: Mapping[str, Any]) -> float | None:
    value = _first(row, ("requested_beta_l2", "requested_beta", "beta_l2"))
    if value is None:
        return None
    try:
        converted = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise NativeEvidenceError("requested beta must be numeric") from error
    if not math.isfinite(converted) or converted < 0.0:
        raise NativeEvidenceError("requested beta must be finite and non-negative")
    return converted


def _geometric_mean(values: Sequence[float]) -> float | None:
    if not values:
        return None
    if any(not math.isfinite(value) or value <= 0.0 for value in values):
        raise NativeEvidenceError("geometric mean inputs must be finite and positive")
    return math.exp(math.fsum(math.log(value) for value in values) / len(values))


def _latency_medians(
    rows: Sequence[Mapping[str, Any]], method: str, metric: str
) -> dict[tuple[str, str, tuple[str, int]], float]:
    grouped: dict[tuple[str, str, tuple[str, int]], list[float]] = {}
    for row in rows:
        if _method_name(row) != method:
            continue
        value = _metric_value(row, metric)
        if value is None:
            continue
        # A run is one process execution.  Reusing a human-readable session ID
        # in a later completed run must not pool the two runs' repetitions into
        # one median and silently change the paired estimator.
        key = (_run_id(row), _session_id(row), _query_identity(row))
        grouped.setdefault(key, []).append(value)
    return {
        key: float(statistics.median(values)) for key, values in sorted(grouped.items())
    }


def _influence_map(
    rows: Sequence[Mapping[str, Any]], experiment_id: str
) -> dict[tuple[str, int], bool]:
    observed: dict[tuple[str, int], set[bool]] = {}
    names = (
        "delta_influence",
        "delta_in_same_c_reference_topk",
        "delta_in_reference_topk",
    )
    for row in rows:
        if _experiment_id(row) != experiment_id:
            continue
        value = _optional_bool(row, names)
        if value is None:
            continue
        observed.setdefault(_query_identity(row), set()).add(value)
    inconsistent = [identity for identity, values in observed.items() if len(values) != 1]
    if inconsistent:
        raise NativeEvidenceError(
            "Delta-influence label changed across methods/sessions for query "
            f"{inconsistent[0]!r}"
        )
    return {identity: next(iter(values)) for identity, values in observed.items()}


def _bootstrap_geometric_mean(
    ratios: Sequence[float], *, resamples: int, seed: int
) -> tuple[float | None, float | None]:
    if not ratios:
        return None, None
    if resamples < MIN_BOOTSTRAP_RESAMPLES:
        raise NativeEvidenceError(
            f"paired bootstrap requires at least {MIN_BOOTSTRAP_RESAMPLES} resamples"
        )
    logs = np.log(np.asarray(ratios, dtype=np.float64))
    generator = np.random.default_rng(seed)
    indices = generator.integers(0, len(logs), size=(resamples, len(logs)))
    samples = np.exp(np.mean(logs[indices], axis=1))
    return float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))


def paired_speedup(
    rows: Sequence[Mapping[str, Any]],
    *,
    experiment_id: str,
    proposed_method: str,
    baseline_method: str,
    baseline_role: str,
    metric: str,
    bootstrap_resamples: int = MIN_BOOTSTRAP_RESAMPLES,
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """Fold paired latency without treating repetitions/sessions as new queries."""

    experiment_rows = [
        row for row in rows if _experiment_id(row) == str(experiment_id)
    ]
    baseline = _latency_medians(experiment_rows, baseline_method, metric)
    proposed = _latency_medians(experiment_rows, proposed_method, metric)
    paired_keys = sorted(set(baseline).intersection(proposed))
    by_query: dict[tuple[str, int], list[float]] = {}
    for run_session_query in paired_keys:
        _, _, query = run_session_query
        by_query.setdefault(query, []).append(
            baseline[run_session_query] / proposed[run_session_query]
        )
    query_ratios = {
        query: float(_geometric_mean(session_ratios))
        for query, session_ratios in sorted(by_query.items())
    }
    ratios = list(query_ratios.values())
    effective_seed = int(
        object_sha256(
            {
                "base_seed": int(bootstrap_seed),
                "experiment_id": str(experiment_id),
                "proposed_method": proposed_method,
                "baseline_method": baseline_method,
                "metric": metric,
                "purpose": "native_paired_query_geomean_bootstrap_v1",
            }
        )[:16],
        16,
    )
    lower, upper = _bootstrap_geometric_mean(
        ratios, resamples=bootstrap_resamples, seed=effective_seed
    )
    influence = _influence_map(experiment_rows, str(experiment_id))
    influence_ratios = [
        ratio for query, ratio in query_ratios.items() if influence.get(query) is True
    ]
    influence_lower, influence_upper = _bootstrap_geometric_mean(
        influence_ratios,
        resamples=bootstrap_resamples,
        seed=effective_seed ^ 0xD31A1F1E,
    )
    baseline_keys = set(baseline)
    proposed_keys = set(proposed)
    quality_mismatch_rows = 0
    quality_mismatch_queries: set[tuple[str, int]] = set()
    if baseline_role == "A":
        for row in experiment_rows:
            if _method_name(row) != baseline_method:
                continue
            mismatch = bool(row.get("baseline_validation_failure", False))
            match = _optional_bool(
                row,
                ("baseline_quality_match", "same_c_order_match", "quality_match"),
            )
            if mismatch or match is False:
                quality_mismatch_rows += 1
                quality_mismatch_queries.add(_query_identity(row))

    # Keep persisted analyses small.  The raw shards remain the authoritative
    # per-query evidence; this digest binds the omitted, sorted query-ratio
    # projection so an analysis can be reproduced and compared without
    # embedding every query in each baseline/metric comparison.
    query_ratio_projection = [
        {
            "query_id": query[0],
            "query_position": query[1],
            "sessions": len(by_query[query]),
            "ratio_geometric_mean": ratio,
        }
        for query, ratio in query_ratios.items()
    ]
    influence_ratio_projection = [
        value
        for value in query_ratio_projection
        if influence.get((str(value["query_id"]), int(value["query_position"])))
        is True
    ]

    return {
        "experiment_id": str(experiment_id),
        "proposed_method": proposed_method,
        "baseline_method": baseline_method,
        "baseline_role": baseline_role,
        "metric": metric,
        "folding": (
            "median repetitions within (run,session,query), baseline/P ratio, "
            "geometric mean across process runs/sessions per query, then geometric mean "
            "across queries"
        ),
        "ratio_definition": f"{baseline_role}_latency / P_latency",
        "paired_session_query_observations": len(paired_keys),
        "paired_queries": len(ratios),
        "sessions_per_query": {
            "min": min((len(values) for values in by_query.values()), default=0),
            "max": max((len(values) for values in by_query.values()), default=0),
        },
        "unpaired_baseline_session_queries": len(baseline_keys - proposed_keys),
        "unpaired_proposed_session_queries": len(proposed_keys - baseline_keys),
        "geometric_mean": _geometric_mean(ratios),
        "bootstrap_95pct_lower": lower,
        "bootstrap_95pct_upper": upper,
        "bootstrap_resamples": bootstrap_resamples if ratios else 0,
        "bootstrap_seed": effective_seed if ratios else None,
        "query_ratios_object_sha256": object_sha256(query_ratio_projection),
        "delta_influence_subset": {
            "paired_queries": len(influence_ratios),
            "geometric_mean": _geometric_mean(influence_ratios),
            "bootstrap_95pct_lower": influence_lower,
            "bootstrap_95pct_upper": influence_upper,
            "bootstrap_resamples": bootstrap_resamples if influence_ratios else 0,
            "bootstrap_seed": (
                (effective_seed ^ 0xD31A1F1E) if influence_ratios else None
            ),
            "query_ratios_object_sha256": object_sha256(
                influence_ratio_projection
            ),
        },
        "baseline_quality_mismatch_rows_retained": quality_mismatch_rows,
        "baseline_quality_mismatch_queries_retained": len(quality_mismatch_queries),
    }


def _constant_value(
    rows: Sequence[Mapping[str, Any]], names: Iterable[str], *, label: str
) -> Any:
    values = {_first(row, names) for row in rows if _first(row, names) is not None}
    if len(values) > 1:
        raise NativeEvidenceError(f"{label} changes within one operating point")
    return next(iter(values)) if values else None


def _experiment_descriptor(
    rows: Sequence[Mapping[str, Any]], metadata: Mapping[str, Any] | None
) -> dict[str, Any]:
    metadata = dict(metadata or {})
    dataset = metadata.get("dataset") if isinstance(metadata.get("dataset"), dict) else {}
    dataset_type = _constant_value(
        rows, ("dataset_type",), label="dataset_type"
    ) or dataset.get("type")
    dataset_name = _constant_value(
        rows, ("dataset_name",), label="dataset_name"
    ) or dataset.get("name", dataset.get("kind"))
    delta_count = _constant_value(
        rows, ("delta_count", "n_delta"), label="Delta count"
    )
    if delta_count is None:
        delta_count = dataset.get("n_delta")
    test_query_count = _constant_value(
        rows, ("test_query_count", "n_test"), label="test query count"
    )
    if test_query_count is None:
        test_query_count = dataset.get("n_test")
    return {
        "dataset_type": dataset_type,
        "dataset_name": dataset_name,
        "n_delta": None if delta_count is None else int(delta_count),
        "expected_test_queries": (
            None if test_query_count is None else int(test_query_count)
        ),
        "axis": metadata.get("axis"),
    }


def _correctness_and_backend_checks(
    rows: Sequence[Mapping[str, Any]], method_roles: Mapping[str, str] | None
) -> dict[str, Any]:
    correctness_failures: list[dict[str, Any]] = []
    backend_failures: list[dict[str, Any]] = []
    native_binary_hashes: set[str] = set()
    checked_native_rows = 0
    checked_certified_rows = 0
    for row in rows:
        role = method_role(row, method_roles)
        if role not in {"F", "N", "P"}:
            continue
        identity = {
            "experiment_id": _experiment_id(row),
            "method": _method_name(row),
            "session_id": _session_id(row),
            "query_id": _query_identity(row)[0],
        }
        checked_native_rows += 1
        backend = _first(row, ("execution_backend", "backend", "backend_name"))
        calls = _first(row, ("native_call_count", "native_calls"))
        call_index = _first(
            row, ("native_call_index", "native_call_sequence", "native_call_id")
        )
        binary_hash = _first(
            row,
            (
                "native_binary_sha256",
                "native_extension_sha256",
                "backend_binary_sha256",
            ),
        )
        python_fallback = _first(
            row, ("python_fallback_used", "native_python_fallback", "python_fallback")
        )
        reasons: list[str] = []
        if backend is None or str(backend).strip().lower() in {
            "",
            "python",
            "python_fallback",
        }:
            reasons.append("missing_or_non_native_backend")
        if (
            isinstance(calls, bool)
            or not isinstance(calls, (int, np.integer))
            or int(calls) <= 0
        ):
            reasons.append("missing_or_zero_native_call_count")
        if (
            isinstance(call_index, bool)
            or not isinstance(call_index, (int, np.integer))
            or int(call_index) < 0
        ):
            reasons.append("missing_or_invalid_native_call_index")
        if (
            not isinstance(binary_hash, str)
            or len(binary_hash) != hashlib.sha256().digest_size * 2
            or any(character not in "0123456789abcdef" for character in binary_hash)
        ):
            reasons.append("missing_or_invalid_native_binary_sha256")
        else:
            native_binary_hashes.add(binary_hash)
        if python_fallback is not None and python_fallback is not False:
            reasons.append("python_fallback_used")
        if reasons:
            backend_failures.append({**identity, "reasons": reasons})

        checked_certified_rows += 1
        reasons = []
        contract_violation = _optional_bool(row, ("contract_violation",))
        contract_valid = _optional_bool(
            row, ("contract_valid", "correctness_passed")
        )
        if contract_violation is True or contract_valid is False:
            reasons.append("reported_contract_violation")
        if contract_violation is None and contract_valid is None:
            reasons.append("missing_explicit_contract_status")
        if _optional_bool(row, ("same_frozen_candidate_object_used",)) is not True:
            reasons.append("same_frozen_candidate_object_not_proved")
        if _optional_bool(row, ("api_candidate_hash_match",)) is not True:
            reasons.append("api_candidate_hash_match_not_proved")
        candidate_hash = row.get("candidate_set_id_or_hash")
        receipt_candidate_hash = row.get("receipt_candidate_set_id_or_hash")
        if (
            not isinstance(candidate_hash, str)
            or not candidate_hash
            or receipt_candidate_hash != candidate_hash
        ):
            reasons.append("receipt_candidate_binding_mismatch")
        beta = _requested_beta(row)
        if beta is None:
            reasons.append("missing_requested_beta")
        else:
            receipt_beta = row.get("receipt_requested_beta_l2")
            try:
                receipt_beta_matches = (
                    receipt_beta is not None
                    and Fraction.from_float(float(receipt_beta))
                    == Fraction.from_float(beta)
                )
            except (TypeError, ValueError, OverflowError):
                receipt_beta_matches = False
            if not receipt_beta_matches:
                reasons.append("receipt_requested_beta_mismatch")
        if beta == 0.0:
            ordered_match = _optional_bool(
                row,
                (
                    "same_c_order_match",
                    "beta_zero_order_match",
                    "ordered_reference_match",
                ),
            )
            if ordered_match is None and "result_keys" in row and "reference_keys" in row:
                ordered_match = row["result_keys"] == row["reference_keys"]
            if ordered_match is not True:
                reasons.append("beta_zero_order_not_proved")
        elif beta is not None:
            observed = _first(
                row, ("observed_beta_upper_l2", "observed_beta_l2")
            )
            certified = _first(row, ("certified_beta_l2", "certified_beta"))
            if observed is None or certified is None:
                reasons.append("positive_beta_chain_missing")
            else:
                observed_fraction = Fraction.from_float(float(observed))
                certified_fraction = Fraction.from_float(float(certified))
                requested_fraction = Fraction.from_float(beta)
                if not (
                    Fraction(0) <= observed_fraction <= certified_fraction <= requested_fraction
                ):
                    reasons.append("positive_beta_chain_invalid")
        if reasons:
            correctness_failures.append({**identity, "reasons": reasons})
    if len(native_binary_hashes) > 1:
        backend_failures.append(
            {
                "experiment_id": None,
                "method": None,
                "session_id": None,
                "query_id": None,
                "reasons": ["multiple_native_binary_hashes_in_analysis"],
                "native_binary_sha256": sorted(native_binary_hashes),
            }
        )
    return {
        "native_backend_calls": {
            "passed": checked_native_rows > 0 and not backend_failures,
            "rows_checked": checked_native_rows,
            "failure_count": len(backend_failures),
            "failure_examples": backend_failures[:100],
            "native_binary_sha256": sorted(native_binary_hashes),
        },
        "correctness": {
            "passed": checked_certified_rows > 0 and not correctness_failures,
            "rows_checked": checked_certified_rows,
            "failure_count": len(correctness_failures),
            "failure_examples": correctness_failures[:100],
        },
    }


def analyze_native_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    method_roles: Mapping[str, str] | None = None,
    experiment_metadata: Mapping[str, Mapping[str, Any]] | None = None,
    bootstrap_resamples: int = MIN_BOOTSTRAP_RESAMPLES,
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> dict[str, Any]:
    if bootstrap_resamples < MIN_BOOTSTRAP_RESAMPLES:
        raise NativeEvidenceError(
            f"bootstrap_resamples must be at least {MIN_BOOTSTRAP_RESAMPLES}"
        )
    copied = [dict(row) for row in rows]
    if not copied:
        raise NativeEvidenceError("native analysis needs at least one raw row")
    roles_by_experiment: dict[str, dict[str, set[str]]] = {}
    for row in copied:
        experiment = _experiment_id(row)
        role = method_role(row, method_roles)
        roles_by_experiment.setdefault(experiment, {}).setdefault(role, set()).add(
            _method_name(row)
        )

    comparisons: list[dict[str, Any]] = []
    operating_points: list[dict[str, Any]] = []
    for experiment, roles in sorted(roles_by_experiment.items()):
        experiment_rows = [row for row in copied if _experiment_id(row) == experiment]
        descriptor = _experiment_descriptor(
            experiment_rows,
            None if experiment_metadata is None else experiment_metadata.get(experiment),
        )
        influence = _influence_map(experiment_rows, experiment)
        for proposed in sorted(roles.get("P", set())):
            proposed_rows = [
                row for row in experiment_rows if _method_name(row) == proposed
            ]
            beta = _constant_value(
                proposed_rows,
                ("requested_beta_l2", "requested_beta", "beta_l2"),
                label="requested beta",
            )
            operating_points.append(
                {
                    "experiment_id": experiment,
                    "proposed_method": proposed,
                    "requested_beta_l2": None if beta is None else float(beta),
                    "delta_influence_queries": sum(influence.values()),
                    "labelled_queries": len(influence),
                    "proposed_queries": len(
                        {_query_identity(row) for row in proposed_rows}
                    ),
                    "proposed_rows": len(proposed_rows),
                    "proposed_rows_with_delta_influence_label": sum(
                        _optional_bool(
                            row,
                            (
                                "delta_influence",
                                "delta_in_same_c_reference_topk",
                                "delta_in_reference_topk",
                            ),
                        )
                        is not None
                        for row in proposed_rows
                    ),
                    **descriptor,
                }
            )
            for baseline_role in ("F", "A"):
                for baseline in sorted(roles.get(baseline_role, set())):
                    for metric in METRIC_FIELDS:
                        comparisons.append(
                            paired_speedup(
                                copied,
                                experiment_id=experiment,
                                proposed_method=proposed,
                                baseline_method=baseline,
                                baseline_role=baseline_role,
                                metric=metric,
                                bootstrap_resamples=bootstrap_resamples,
                                bootstrap_seed=bootstrap_seed,
                            )
                        )

    checks = _correctness_and_backend_checks(copied, method_roles)
    return {
        "schema_version": NATIVE_ANALYSIS_SCHEMA_VERSION,
        "study_id": "issue-3-native-recheck",
        "bootstrap_policy": {
            "resamples": bootstrap_resamples,
            "base_seed": bootstrap_seed,
            "independent_unit": "query",
            "session_policy": "geometric_mean_within_query_before_bootstrap",
        },
        "totals": {
            "raw_rows": len(copied),
            "experiments": len(roles_by_experiment),
            "comparisons": len(comparisons),
            "unique_queries": len(
                {(_experiment_id(row), _query_identity(row)) for row in copied}
            ),
            "sessions": len({_session_id(row) for row in copied}),
        },
        "methods_by_experiment": {
            experiment: {
                role: sorted(methods) for role, methods in sorted(roles.items())
            }
            for experiment, roles in sorted(roles_by_experiment.items())
        },
        "operating_points": operating_points,
        "comparisons": comparisons,
        "integrity_checks": checks,
        "analysis_note": (
            "A quality mismatches remain in timing pairs; repetitions and process "
            "sessions never increase the bootstrap's independent query count."
        ),
    }


def evaluate_validation_gate(
    summary: Mapping[str, Any], *, primary_metric: str = "api_wall"
) -> dict[str, Any]:
    if primary_metric not in METRIC_FIELDS:
        raise NativeEvidenceError(f"unknown gate metric {primary_metric!r}")
    checks = summary.get("integrity_checks")
    if not isinstance(checks, dict):
        raise NativeEvidenceError("analysis summary has no integrity checks")
    backend_passed = bool(checks.get("native_backend_calls", {}).get("passed"))
    correctness_passed = bool(checks.get("correctness", {}).get("passed"))
    source = summary.get("source_integrity")
    # Unit-level in-memory analysis has no source descriptor.  The command-line
    # validation decision always uses analyze_native_path(), where the absence
    # of completed validation manifests is a hard gate failure.
    source_passed = True
    if source is not None:
        if not isinstance(source, dict):
            raise NativeEvidenceError("malformed source_integrity record")
        source_passed = bool(source.get("validation_gate_source_passed"))
    comparisons = list(summary.get("comparisons", []))
    candidates: list[dict[str, Any]] = []
    for operating_point in summary.get("operating_points", []):
        experiment = str(operating_point["experiment_id"])
        proposed = str(operating_point["proposed_method"])
        relevant = [
            row
            for row in comparisons
            if row.get("experiment_id") == experiment
            and row.get("proposed_method") == proposed
            and row.get("metric") == primary_metric
        ]
        reasons: list[str] = []
        dataset_type = str(operating_point.get("dataset_type") or "")
        real_non_degenerate = (
            dataset_type == "texmex"
            and operating_point.get("n_delta") is not None
            and int(operating_point["n_delta"]) >= 1_000
        )
        dataset_name = str(operating_point.get("dataset_name") or "")
        sift_primary = dataset_name in {"sift", "texmex-sift"}
        if not real_non_degenerate:
            reasons.append("not_real_delta_ge_1000")
        if not sift_primary:
            reasons.append("not_sift_primary_for_fresh_holdout")
        if int(operating_point.get("delta_influence_queries", 0)) <= 0:
            reasons.append("no_delta_influence_query")
        if int(operating_point.get("labelled_queries", 0)) != int(
            operating_point.get("proposed_queries", -1)
        ):
            reasons.append("incomplete_delta_influence_labels")
        if int(
            operating_point.get("proposed_rows_with_delta_influence_label", -1)
        ) != int(operating_point.get("proposed_rows", -2)):
            reasons.append("incomplete_delta_influence_row_labels")
        expected_test_queries = operating_point.get("expected_test_queries")
        if expected_test_queries is None or int(expected_test_queries) <= 0:
            reasons.append("missing_expected_test_query_count")
        if not source_passed:
            reasons.append("validation_evidence_source_gate_failed")
        if not backend_passed:
            reasons.append("native_backend_or_call_gate_failed")
        if not correctness_passed:
            reasons.append("correctness_gate_failed")
        comparison_outcomes: dict[str, Any] = {}
        for baseline_role in ("F", "A"):
            role_rows = [row for row in relevant if row.get("baseline_role") == baseline_role]
            timing_passed = bool(role_rows) and all(
                row.get("paired_queries", 0) > 0
                and expected_test_queries is not None
                and row.get("paired_queries") == int(expected_test_queries)
                and row.get("unpaired_baseline_session_queries", 0) == 0
                and row.get("unpaired_proposed_session_queries", 0) == 0
                and row.get("bootstrap_95pct_lower") is not None
                and float(row["bootstrap_95pct_lower"]) > 1.0
                for row in role_rows
            )
            quality_passed = baseline_role != "A" or (
                bool(role_rows)
                and all(
                    int(row.get("baseline_quality_mismatch_queries_retained", 0))
                    == 0
                    for row in role_rows
                )
            )
            role_passed = timing_passed and quality_passed
            comparison_outcomes[baseline_role] = {
                "passed": role_passed,
                "timing_passed": timing_passed,
                "same_c_quality_passed": quality_passed,
                "comparisons": role_rows,
            }
            if not timing_passed:
                reasons.append(f"paired_ci_lower_not_above_one_vs_{baseline_role}")
            if baseline_role == "A" and not quality_passed:
                reasons.append("A_same_c_quality_mismatch")
        candidates.append(
            {
                **dict(operating_point),
                "primary_metric": primary_metric,
                "real_non_degenerate": real_non_degenerate,
                "sift_primary_for_fresh_holdout": sift_primary,
                "comparison_outcomes": comparison_outcomes,
                "passed": not reasons,
                "reasons": reasons,
            }
        )

    passed_candidates = [candidate for candidate in candidates if candidate["passed"]]

    def selection_key(candidate: Mapping[str, Any]) -> tuple[Any, ...]:
        beta = candidate.get("requested_beta_l2")
        positive_beta = beta is None or float(beta) != 0.0
        lower_bounds = [
            float(comparison["bootstrap_95pct_lower"])
            for role in ("F", "A")
            for comparison in candidate["comparison_outcomes"][role]["comparisons"]
        ]
        score = min(lower_bounds) if lower_bounds else -math.inf
        return (
            positive_beta,
            -score,
            math.inf if beta is None else float(beta),
            str(candidate["experiment_id"]),
            str(candidate["proposed_method"]),
        )

    selected = min(passed_candidates, key=selection_key) if passed_candidates else None
    status = "PASSED" if selected is not None else "NOT_PASSED"
    reasons: list[str] = []
    if not candidates:
        reasons.append("no_P_operating_point")
    if not backend_passed:
        reasons.append("native_backend_or_call_gate_failed")
    if not correctness_passed:
        reasons.append("correctness_gate_failed")
    if not source_passed:
        reasons.append("validation_evidence_source_gate_failed")
    if candidates and not passed_candidates:
        reasons.append("no_non_degenerate_candidate_beats_both_F_and_A")
    return {
        "schema_version": NATIVE_ANALYSIS_SCHEMA_VERSION,
        "study_id": "issue-3-native-recheck",
        "gate_status": status,
        "primary_metric": primary_metric,
        "selected_candidate": selected,
        "candidates": candidates,
        "requirements": {
            "native_backend_calls": checks.get("native_backend_calls"),
            "correctness": checks.get("correctness"),
            "validation_evidence_source": source,
            "real_delta_minimum": 1_000,
            "fresh_holdout_primary_dataset": "sift",
            "delta_influence_required": True,
            "paired_bootstrap_ci_lower_strictly_above": 1.0,
            "required_baselines": ["F", "A"],
        },
        "reasons": reasons,
        "lock_created": False,
        "lock_policy": (
            "This analyzer never creates a final lock. A later lock writer may act "
            "only on a PASSED decision bound to completed validation evidence."
        ),
    }


def _discover_run_dirs(input_path: Path) -> list[Path]:
    if (input_path / "run_manifest.json").is_file():
        return [input_path]
    return sorted(
        path.parent
        for path in input_path.rglob("run_manifest.json")
        if "analysis" not in path.parts
    )


def _verify_completed_run(run_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest_path = run_dir / "run_manifest.json"
    checkpoint_path = run_dir / "checkpoint.json"
    completion_path = run_dir / "COMPLETED.json"
    effective_path = run_dir / "effective_config.json"
    for required in (manifest_path, checkpoint_path, completion_path, effective_path):
        if not required.is_file():
            raise NativeEvidenceError(f"completed native run is missing {required.name}")
    manifest = _read_object(manifest_path)
    checkpoint = _read_object(checkpoint_path)
    completion = _read_object(completion_path)
    effective = _read_object(effective_path)
    run_id = str(manifest.get("run_id", ""))
    config_hash = str(manifest.get("config_hash", ""))
    implementation_hash = manifest.get("implementation_tree_sha256")
    if not run_id or not config_hash or not implementation_hash:
        raise NativeEvidenceError(f"native run identities are incomplete in {run_dir}")
    identities = (
        ("completion run", completion.get("run_id"), run_id),
        ("completion config", completion.get("config_hash"), config_hash),
        (
            "completion implementation",
            completion.get("implementation_tree_sha256"),
            implementation_hash,
        ),
        ("checkpoint run", checkpoint.get("run_id"), run_id),
        ("checkpoint config", checkpoint.get("config_hash"), config_hash),
        (
            "checkpoint implementation",
            checkpoint.get("implementation_tree_sha256"),
            implementation_hash,
        ),
    )
    for label, observed, expected in identities:
        if observed != expected:
            raise NativeEvidenceError(f"{label} identity mismatch in {run_dir}")
    if object_sha256(effective) != config_hash:
        raise NativeEvidenceError(f"effective config hash mismatch in {run_dir}")
    if completion.get("checkpoint_sha256") != file_sha256(checkpoint_path):
        raise NativeEvidenceError(f"completion checkpoint hash mismatch in {run_dir}")
    if completion.get("run_manifest_sha256") != file_sha256(manifest_path):
        raise NativeEvidenceError(f"completion manifest hash mismatch in {run_dir}")
    holdout_registration = manifest.get("holdout_registration")
    if holdout_registration is not None and (
        not isinstance(holdout_registration, dict)
        or completion.get("holdout_manifest_sha256")
        != holdout_registration.get("sha256")
        or completion.get("final_policy_file_sha256")
        != holdout_registration.get("final_policy_file_sha256")
        or completion.get("final_policy_object_sha256")
        != holdout_registration.get("final_policy_object_sha256")
    ):
        raise NativeEvidenceError(
            f"completion holdout/final-policy identity mismatch in {run_dir}"
        )
    _verify_ancillary_inventory(run_dir, completion)
    _verify_native_lb_audits(run_dir, effective, manifest)
    completed_blocks = checkpoint.get("completed_blocks")
    if not isinstance(completed_blocks, dict):
        raise NativeEvidenceError(f"malformed checkpoint in {run_dir}")
    block_count = len(completed_blocks)
    if (
        manifest.get("status") != "completed"
        or int(manifest.get("completed_blocks", -1)) != block_count
        or int(manifest.get("total_blocks", -1)) != block_count
        or int(completion.get("raw_shards", -1)) != block_count
    ):
        raise NativeEvidenceError(f"completion block/status mismatch in {run_dir}")
    failures = manifest.get("failures", [])
    if not isinstance(failures, list) or any(
        not isinstance(failure, dict)
        or failure.get("kind") != "uncheckpointed_raw_preserved_on_resume"
        for failure in failures
    ):
        raise NativeEvidenceError(
            f"completed run contains a scientific failure: {run_dir}"
        )
    return manifest, effective


def _verify_native_lb_audits(
    run_dir: Path,
    effective: Mapping[str, Any],
    manifest: Mapping[str, Any],
) -> None:
    dataset_hashes = manifest.get("dataset_hashes")
    split_ids = manifest.get("split_ids")
    if not isinstance(dataset_hashes, dict) or not isinstance(split_ids, dict):
        raise NativeEvidenceError(f"run manifest has no audit identities: {run_dir}")
    requested = int(effective.get("lb_audit_queries", 0))
    for specification in effective.get("experiments", []):
        if not isinstance(specification, dict) or specification.get("id") is None:
            raise NativeEvidenceError(f"malformed experiment config in {run_dir}")
        experiment = str(specification["id"])
        dataset = specification.get("dataset")
        if not isinstance(dataset, dict):
            raise NativeEvidenceError(f"malformed dataset config for {experiment}")
        expected_queries = min(requested, int(dataset.get("n_validation", 0)))
        path = run_dir / "audits" / f"{_safe_experiment_name(experiment)}.native-lb.json"
        if not path.is_file():
            raise NativeEvidenceError(f"completed run is missing native LB audit: {path}")
        audit = _read_object(path)
        if (
            audit.get("experiment_id") != experiment
            or audit.get("dataset_hash") != dataset_hashes.get(experiment)
            or audit.get("split_id") != split_ids.get(experiment)
            or int(audit.get("query_count", -1)) != expected_queries
            or audit.get("passed") is not True
            or audit.get("failures") != []
        ):
            raise NativeEvidenceError(
                f"native LB audit identity/query-count/pass mismatch: {path}"
            )


def _verify_ancillary_inventory(
    run_dir: Path, completion: Mapping[str, Any]
) -> None:
    inventory = completion.get("ancillary_files")
    if not isinstance(inventory, list):
        raise NativeEvidenceError(
            f"completed native run has no ancillary_files inventory: {run_dir}"
        )
    expected_paths = {"effective_config.json", "build_manifest.json"}
    for directory_name in ("splits", "validation", "audits", "failures"):
        directory = run_dir / directory_name
        if directory.is_dir():
            expected_paths.update(
                path.relative_to(run_dir).as_posix()
                for path in directory.rglob("*")
                if path.is_file()
            )
    observed_paths: set[str] = set()
    root = run_dir.resolve()
    for entry in inventory:
        if not isinstance(entry, dict):
            raise NativeEvidenceError("ancillary inventory entry must be an object")
        relative = entry.get("path")
        if not isinstance(relative, str) or not relative or "\\" in relative:
            raise NativeEvidenceError("ancillary path must be a relative POSIX path")
        parsed = PurePosixPath(relative)
        if (
            parsed.is_absolute()
            or parsed.as_posix() != relative
            or any(part in {"", ".", ".."} for part in parsed.parts)
        ):
            raise NativeEvidenceError("ancillary path escapes or is not canonical")
        if relative in observed_paths:
            raise NativeEvidenceError(f"duplicate ancillary path: {relative}")
        observed_paths.add(relative)
        path = run_dir.joinpath(*parsed.parts)
        try:
            path.resolve(strict=True).relative_to(root)
        except (FileNotFoundError, ValueError) as error:
            raise NativeEvidenceError(
                f"ancillary path escapes the run directory or is missing: {relative}"
            ) from error
        if path.is_symlink() or not path.is_file():
            raise NativeEvidenceError(
                f"ancillary path is not a regular in-run file: {relative}"
            )
        digest = entry.get("sha256")
        if (
            not isinstance(digest, str)
            or len(digest) != hashlib.sha256().digest_size * 2
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise NativeEvidenceError(f"invalid ancillary SHA-256: {relative}")
        size = entry.get("bytes")
        if isinstance(size, bool) or not isinstance(size, int) or size < 0:
            raise NativeEvidenceError(f"invalid ancillary byte count: {relative}")
        if path.stat().st_size != size:
            raise NativeEvidenceError(f"ancillary byte-count mismatch: {relative}")
        if file_sha256(path) != digest:
            raise NativeEvidenceError(f"ancillary checksum mismatch: {relative}")
    if observed_paths != expected_paths:
        missing = sorted(expected_paths - observed_paths)
        unexpected = sorted(observed_paths - expected_paths)
        raise NativeEvidenceError(
            "ancillary inventory coverage mismatch: "
            f"missing={missing}, unexpected={unexpected}"
        )


def _resolve_checkpoint_raw_path(run_dir: Path, value: Any) -> tuple[str, Path]:
    """Resolve one checkpoint shard without permitting path indirection."""

    if not isinstance(value, str) or not value or "\\" in value:
        raise NativeEvidenceError("checkpoint raw path must be a relative POSIX path")
    parsed = PurePosixPath(value)
    if (
        parsed.is_absolute()
        or parsed.as_posix() != value
        or any(part in {"", ".", ".."} for part in parsed.parts)
        or not parsed.parts
        or parsed.parts[0] != "raw"
    ):
        raise NativeEvidenceError(
            f"checkpoint raw path escapes or is not canonical: {value}"
        )
    path = run_dir.joinpath(*parsed.parts)
    try:
        path.resolve(strict=True).relative_to(run_dir.resolve())
    except (FileNotFoundError, ValueError) as error:
        raise NativeEvidenceError(
            f"checkpoint raw path escapes the run directory or is missing: {value}"
        ) from error
    if path.is_symlink() or not path.is_file():
        raise NativeEvidenceError(
            f"checkpoint raw path is not a regular in-run file: {value}"
        )
    return value, path


def _integer_interval(value: Any, *, label: str) -> tuple[int, int]:
    if (
        not isinstance(value, list)
        or len(value) != 2
        or any(isinstance(item, bool) or not isinstance(item, int) for item in value)
    ):
        raise NativeEvidenceError(f"{label} must be a two-integer half-open interval")
    start, stop = (int(value[0]), int(value[1]))
    if start < 0 or stop < start:
        raise NativeEvidenceError(f"{label} is not a valid half-open interval")
    return start, stop


def _verify_validation_holdout(
    effective: Mapping[str, Any], *, run_dir: Path, run_manifest: Mapping[str, Any]
) -> dict[str, Any] | None:
    """Bind validation evidence to the pre-registered query selection."""

    declared = {
        str(value).strip().lower()
        for value in (effective.get("phase"), effective.get("evidence_role"))
        if value is not None and str(value).strip()
    }
    if declared and declared != {"validation"}:
        return None
    policy = effective.get("holdout_policy")
    if not isinstance(policy, dict):
        raise NativeEvidenceError(
            f"validation run has no holdout_policy: {run_dir}"
        )
    manifest_value = policy.get("manifest")
    if not isinstance(manifest_value, str) or not manifest_value:
        raise NativeEvidenceError(
            f"validation holdout_policy has no manifest path: {run_dir}"
        )
    manifest_path = Path(manifest_value)
    if not manifest_path.is_absolute():
        manifest_path = Path(__file__).resolve().parents[2] / manifest_path
    if not manifest_path.is_file():
        raise NativeEvidenceError(
            f"pre-registered holdout manifest is missing: {manifest_path}"
        )
    holdout = _read_object(manifest_path)
    current_manifest_sha = file_sha256(manifest_path)
    saved_registration = run_manifest.get("holdout_registration")
    if (
        not isinstance(saved_registration, dict)
        or saved_registration.get("path") != manifest_value
        or saved_registration.get("sha256") != current_manifest_sha
        or saved_registration.get("phase") != "validation"
    ):
        raise NativeEvidenceError(
            "current holdout manifest differs from the completion-bound registration"
        )
    if holdout.get("study_id") != "issue-3-native-recheck":
        raise NativeEvidenceError("holdout manifest study identity mismatch")
    selection = holdout.get("selection_integrity")
    if not isinstance(selection, dict):
        raise NativeEvidenceError("holdout manifest has no selection_integrity")
    registered_config_hash = selection.get(
        "validation_source_config_object_sha256"
    )
    observed_config_hash = effective.get("source_config_object_hash")
    if (
        not isinstance(registered_config_hash, str)
        or observed_config_hash != registered_config_hash
    ):
        raise NativeEvidenceError(
            "validation source config does not match the pre-registered hash"
        )
    registered_config_file_hash = selection.get(
        "validation_source_config_file_sha256"
    )
    observed_config_file_hash = effective.get("source_config_file_sha256")
    if (
        not isinstance(registered_config_file_hash, str)
        or observed_config_file_hash != registered_config_file_hash
    ):
        raise NativeEvidenceError(
            "validation source config file does not match the pre-registered hash"
        )
    configured_final = policy.get("pre_registered_final_policy")
    registered_final = holdout.get("pre_registered_final_policy")
    if not isinstance(configured_final, dict) or not isinstance(registered_final, dict):
        raise NativeEvidenceError("final policy was not pre-registered")
    final_identity = {
        "path": registered_final.get("path"),
        "file_sha256": registered_final.get("file_sha256"),
        "object_sha256": registered_final.get("object_sha256"),
    }
    if configured_final != final_identity:
        raise NativeEvidenceError(
            "validation config and holdout disagree on final-policy identity"
        )
    try:
        validate_issue_fixed_decision_values(
            registered_final.get("issue_fixed_decision_values")
        )
        validate_issue_fixed_decision_values(
            saved_registration.get("issue_fixed_decision_values")
        )
    except FinalPolicyError as error:
        raise NativeEvidenceError(str(error)) from error
    if (
        selection.get("final_policy_path") != final_identity["path"]
        or selection.get("final_policy_file_sha256")
        != final_identity["file_sha256"]
        or selection.get("final_policy_object_sha256")
        != final_identity["object_sha256"]
        or saved_registration.get("final_policy_path") != final_identity["path"]
        or saved_registration.get("final_policy_file_sha256")
        != final_identity["file_sha256"]
        or saved_registration.get("final_policy_object_sha256")
        != final_identity["object_sha256"]
    ):
        raise NativeEvidenceError("completion-bound final-policy registration differs")
    final_policy_path = Path(str(final_identity["path"]))
    if not final_policy_path.is_absolute():
        final_policy_path = Path(__file__).resolve().parents[2] / final_policy_path
    current_final_policy = _read_object(final_policy_path)
    try:
        validate_final_policy(current_final_policy)
    except FinalPolicyError as error:
        raise NativeEvidenceError(str(error)) from error
    if (
        file_sha256(final_policy_path) != final_identity["file_sha256"]
        or object_sha256(current_final_policy) != final_identity["object_sha256"]
    ):
        raise NativeEvidenceError("current final policy differs from pre-registration")

    sift = holdout.get("sift")
    if not isinstance(sift, dict):
        raise NativeEvidenceError("holdout manifest has no SIFT registration")
    history = sift.get("historically_used_query_ids")
    validation = sift.get("native_validation_selection")
    final = sift.get("pre_registered_fresh_final_holdout")
    if not all(isinstance(value, dict) for value in (history, validation, final)):
        raise NativeEvidenceError("holdout manifest has malformed SIFT sections")
    assert isinstance(history, dict)
    assert isinstance(validation, dict)
    assert isinstance(final, dict)
    historical_interval = _integer_interval(
        history.get("interval"), label="historical SIFT interval"
    )
    beta_interval = _integer_interval(
        validation.get("beta_calibration_interval"),
        label="registered SIFT beta interval",
    )
    measurement_interval = _integer_interval(
        validation.get("measurement_interval"),
        label="registered SIFT measurement interval",
    )
    final_interval = _integer_interval(
        final.get("interval"), label="registered SIFT final interval"
    )
    policy_final = _integer_interval(
        policy.get("sift_final_holdout_is_pre_registered_but_not_loaded"),
        label="configured SIFT final interval",
    )
    threshold = policy.get("validation_must_not_load_sift_query_ids_at_or_above")
    if isinstance(threshold, bool) or not isinstance(threshold, int):
        raise NativeEvidenceError("validation SIFT query threshold is missing")
    if int(threshold) != historical_interval[1] or policy_final != final_interval:
        raise NativeEvidenceError("validation holdout policy disagrees with registration")
    if (
        final.get("loaded_by_native_validation") is not False
        or final.get("selected_before_native_validation_timing") is not True
        or final.get("disjoint_from_historically_used_union") is not True
    ):
        raise NativeEvidenceError("fresh SIFT holdout registration is not fail-closed")

    sift_experiments = 0
    gist_experiments = 0
    for specification in effective.get("experiments", []):
        if not isinstance(specification, dict):
            raise NativeEvidenceError("malformed experiment in validation config")
        dataset = specification.get("dataset")
        if not isinstance(dataset, dict) or dataset.get("type") != "texmex":
            continue
        name = str(dataset.get("name", "")).lower()
        if name == "sift":
            sift_experiments += 1
            try:
                validation_start = int(dataset["validation_query_offset"])
                validation_stop = validation_start + int(dataset["n_validation"])
                test_start = int(dataset["test_query_offset"])
                test_stop = test_start + int(dataset["n_test"])
            except (KeyError, TypeError, ValueError) as error:
                raise NativeEvidenceError(
                    "SIFT validation experiment has no explicit query intervals"
                ) from error
            if (validation_start, validation_stop) != beta_interval or (
                test_start,
                test_stop,
            ) != measurement_interval:
                raise NativeEvidenceError(
                    "SIFT validation query intervals differ from pre-registration"
                )
            if max(validation_stop, test_stop) > int(threshold):
                raise NativeEvidenceError("validation loaded a reserved SIFT query ID")
            if not (
                validation_stop <= final_interval[0]
                and test_stop <= final_interval[0]
            ):
                raise NativeEvidenceError("validation overlaps the fresh SIFT holdout")
        elif name == "gist":
            gist_experiments += 1
    if sift_experiments == 0:
        raise NativeEvidenceError("validation evidence has no registered SIFT experiment")
    gist = holdout.get("gist")
    if gist_experiments and (
        policy.get("gist_measurement_is_reused_non_independent") is not True
        or not isinstance(gist, dict)
        or gist.get("independence_label") != "reused_non_independent"
        or gist.get("fresh_holdout_claim_permitted") is not False
    ):
        raise NativeEvidenceError("GIST reuse is not labelled non-independent")
    return {
        "passed": True,
        "manifest_path": str(manifest_path),
        "manifest_sha256": current_manifest_sha,
        "registered_source_config_object_hash": registered_config_hash,
        "registered_source_config_file_sha256": registered_config_file_hash,
        "registered_final_policy_path": final_identity["path"],
        "registered_final_policy_file_sha256": final_identity["file_sha256"],
        "registered_final_policy_object_sha256": final_identity["object_sha256"],
        "issue_fixed_decision_values": issue_fixed_decision_values(),
        "sift_validation_interval": list(beta_interval),
        "sift_measurement_interval": list(measurement_interval),
        "sift_final_interval": list(final_interval),
        "sift_experiments_checked": sift_experiments,
        "gist_experiments_checked": gist_experiments,
    }


def _safe_experiment_name(value: str) -> str:
    rendered = "".join(
        character if character.isalnum() or character in "-_" else "-"
        for character in value
    ).strip("-")
    if not rendered:
        raise NativeEvidenceError("experiment identifier has no safe characters")
    return rendered


def _load_split_query_maps(
    run_dir: Path,
    manifest: Mapping[str, Any],
    effective: Mapping[str, Any],
) -> dict[str, dict[int, str]]:
    manifest_split_ids = manifest.get("split_ids")
    manifest_dataset_hashes = manifest.get("dataset_hashes")
    if not isinstance(manifest_split_ids, dict) or not isinstance(
        manifest_dataset_hashes, dict
    ):
        raise NativeEvidenceError(f"run manifest has no split identities: {run_dir}")
    expected: dict[str, dict[int, str]] = {}
    for specification in effective.get("experiments", []):
        if not isinstance(specification, dict) or specification.get("id") is None:
            raise NativeEvidenceError(f"malformed experiment config in {run_dir}")
        experiment = str(specification["id"])
        split_path = run_dir / "splits" / f"{_safe_experiment_name(experiment)}.json"
        if not split_path.is_file():
            raise NativeEvidenceError(f"completed run is missing split manifest: {split_path}")
        split = _read_object(split_path)
        declared_split_id = split.get("split_id")
        unsigned_split = dict(split)
        unsigned_split.pop("split_id", None)
        if declared_split_id != object_sha256(unsigned_split):
            raise NativeEvidenceError(f"split manifest self-hash mismatch: {split_path}")
        if split.get("experiment_id") != experiment:
            raise NativeEvidenceError(f"split experiment identity mismatch: {split_path}")
        if manifest_split_ids.get(experiment) != declared_split_id:
            raise NativeEvidenceError(f"run/split identity mismatch: {split_path}")
        if manifest_dataset_hashes.get(experiment) != split.get("dataset_hash"):
            raise NativeEvidenceError(f"run/dataset identity mismatch: {split_path}")
        id_ranges = split.get("id_ranges")
        if not isinstance(id_ranges, dict) or not isinstance(id_ranges.get("test"), list):
            raise NativeEvidenceError(f"split has no explicit test query IDs: {split_path}")
        test_ids = [str(value) for value in id_ranges["test"]]
        dataset = specification.get("dataset")
        if not isinstance(dataset, dict) or int(dataset.get("n_test", -1)) != len(
            test_ids
        ):
            raise NativeEvidenceError(f"split/config test query count mismatch: {split_path}")
        if len(set(test_ids)) != len(test_ids):
            raise NativeEvidenceError(f"split contains duplicate test query IDs: {split_path}")
        expected[experiment] = dict(enumerate(test_ids))
    return expected


def _verify_run_query_coverage(
    rows: Sequence[Mapping[str, Any]],
    expected: Mapping[str, Mapping[int, str]],
    effective: Mapping[str, Any],
    *,
    run_dir: Path,
) -> None:
    specifications = {
        str(row["id"]): row
        for row in effective.get("experiments", [])
        if isinstance(row, dict) and row.get("id") is not None
    }
    for experiment, position_to_id in expected.items():
        experiment_rows = [
            row for row in rows if _experiment_id(row) == experiment
        ]
        if not experiment_rows:
            raise NativeEvidenceError(
                f"completed run has no rows for {experiment}: {run_dir}"
            )
        for row in experiment_rows:
            query_id, query_position = _query_identity(row)
            if position_to_id.get(query_position) != query_id:
                raise NativeEvidenceError(
                    f"raw query is outside its saved split for {experiment}: {run_dir}"
                )
        repetitions = int(
            specifications[experiment].get(
                "repetitions", effective.get("repetitions", 0)
            )
        )
        if repetitions <= 0:
            raise NativeEvidenceError(f"invalid repetition count for {experiment}")
        specification = specifications[experiment]
        expected_methods = {
            "native_F": "F",
            "faiss_A": "A",
            "faiss_A_reference": "A-reference",
            "native_N": "N",
        }
        beta_factors = [
            float(value) for value in specification.get("beta_factors", [0.0])
        ]
        expected_p_factors: dict[str, float] = {}
        for factor in beta_factors:
            suffix = (
                "beta0"
                if factor == 0.0
                else "factor_"
                + format(factor, ".8g").replace("-", "m").replace("+", "p").replace(".", "p")
            )
            name = f"native_P_{suffix}"
            expected_methods[name] = "P"
            expected_p_factors[name] = factor
        if specification.get("kernel_ablations") is True:
            expected_methods.update(
                {
                    "ablation_F_rescan_all_exact": "ablation",
                    "ablation_F_heap_all_exact": "ablation",
                    "ablation_P_beta0_rescan_adaptive": "ablation",
                }
            )
        expected_methods["old_O"] = "O"
        observed_methods = {_method_name(row) for row in experiment_rows}
        if observed_methods != set(expected_methods):
            raise NativeEvidenceError(
                f"completed native method matrix differs from effective config for "
                f"{experiment}: missing={sorted(set(expected_methods) - observed_methods)}, "
                f"unexpected={sorted(observed_methods - set(expected_methods))}"
            )
        for row in experiment_rows:
            method = _method_name(row)
            if method_role(row) != expected_methods[method]:
                raise NativeEvidenceError(
                    f"method role changes within {experiment}: {method}"
                )
        expected_trials = {
            (position, query_id, repetition)
            for position, query_id in position_to_id.items()
            for repetition in range(repetitions)
        }
        for method in expected_methods:
            if method == "old_O":
                continue
            observed_trials: list[tuple[int, str, int]] = []
            for row in experiment_rows:
                if _method_name(row) != method:
                    continue
                try:
                    repetition = int(row["repetition"])
                except (KeyError, TypeError, ValueError) as error:
                    raise NativeEvidenceError(
                        f"invalid repetition for {experiment}/{method}"
                    ) from error
                query_id, query_position = _query_identity(row)
                observed_trials.append((query_position, query_id, repetition))
            if len(observed_trials) != len(set(observed_trials)):
                raise NativeEvidenceError(
                    f"duplicate raw query/repetition rows for {experiment}/{method}"
                )
            if set(observed_trials) != expected_trials:
                raise NativeEvidenceError(
                    f"incomplete raw query coverage for {experiment}/{method}: {run_dir}"
                )
            if method in expected_p_factors:
                factor = expected_p_factors[method]
                p_rows = [
                    row for row in experiment_rows if _method_name(row) == method
                ]
                if any(float(row.get("beta_factor", float("nan"))) != factor for row in p_rows):
                    raise NativeEvidenceError(
                        f"P beta factor differs from effective config for {experiment}/{method}"
                    )
                requested = {float(row.get("requested_beta_l2", float("nan"))) for row in p_rows}
                if len(requested) != 1 or any(
                    not math.isfinite(value) or value < 0 for value in requested
                ):
                    raise NativeEvidenceError(
                        f"P requested beta is inconsistent for {experiment}/{method}"
                    )
                if factor == 0.0 and requested != {0.0}:
                    raise NativeEvidenceError(
                        f"P beta0 is nonzero for {experiment}/{method}"
                    )
                if factor > 0.0 and requested == {0.0}:
                    raise NativeEvidenceError(
                        f"positive P beta factor produced beta0 for {experiment}/{method}"
                    )
        oracle_rows = [
            row for row in experiment_rows if _method_name(row) == "old_O"
        ]
        oracle_trials = [
            (*reversed(_query_identity(row)), int(row.get("repetition", -2)))
            for row in oracle_rows
        ]
        expected_oracle = {
            (position, query_id, -1) for position, query_id in position_to_id.items()
        }
        if (
            len(oracle_trials) != len(set(oracle_trials))
            or set(oracle_trials) != expected_oracle
        ):
            raise NativeEvidenceError(
                f"incomplete O oracle coverage for {experiment}: {run_dir}"
            )
        fraction_count = min(
            int(effective.get("independent_oracle_queries", 0)),
            len(position_to_id),
        )
        for row in oracle_rows:
            _, position = _query_identity(row)
            expected_checked = position < fraction_count
            if (
                row.get("fraction_oracle_checked") is not expected_checked
                or (
                    expected_checked
                    and row.get("fraction_oracle_order_match") is not True
                )
                or (
                    not expected_checked
                    and row.get("fraction_oracle_order_match") is not None
                )
                or row.get("requested_beta_l2") != 0.0
                or row.get("repetition") != -1
                or row.get("micro_latency_ns") is not None
                or row.get("composed_e2e_latency_ns") is not None
                or row.get("api_wall_latency_ns") is not None
            ):
                raise NativeEvidenceError(
                    f"O Fraction-oracle declaration mismatch for {experiment} "
                    f"query position {position}"
                )


def load_native_evidence(input_path: str | Path) -> LoadedNativeEvidence:
    root = Path(input_path)
    run_dirs = _discover_run_dirs(root) if root.is_dir() else []
    if not run_dirs:
        if root.is_file() and root.suffix == ".jsonl":
            rows = _read_jsonl(root)
        elif root.is_dir():
            files = sorted(root.rglob("*.jsonl"))
            rows = [row for path in files for row in _read_jsonl(path)]
        else:
            raise NativeEvidenceError(f"no native evidence found at {root}")
        if not rows:
            raise NativeEvidenceError(f"no native raw rows found at {root}")
        return LoadedNativeEvidence(tuple(rows), (), (), {})

    rows: list[dict[str, Any]] = []
    runs: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    metadata: dict[str, Mapping[str, Any]] = {}
    for run_dir in run_dirs:
        manifest = _read_object(run_dir / "run_manifest.json")
        if not (run_dir / "COMPLETED.json").is_file():
            excluded.append(
                {
                    "run_id": manifest.get("run_id"),
                    "status": manifest.get("status"),
                    "reason": "missing_COMPLETED_json",
                }
            )
            continue
        manifest, effective = _verify_completed_run(run_dir)
        holdout_verification = _verify_validation_holdout(
            effective, run_dir=run_dir, run_manifest=manifest
        )
        run_id = str(manifest["run_id"])
        config_hash = str(manifest["config_hash"])
        implementation_hash = str(manifest["implementation_tree_sha256"])
        split_query_maps = _load_split_query_maps(run_dir, manifest, effective)
        for specification in effective.get("experiments", []):
            if isinstance(specification, dict) and specification.get("id") is not None:
                experiment_id = str(specification["id"])
                candidate_metadata = dict(specification)
                previous_metadata = metadata.get(experiment_id)
                if (
                    previous_metadata is not None
                    and object_sha256(previous_metadata)
                    != object_sha256(candidate_metadata)
                ):
                    raise NativeEvidenceError(
                        "experiment metadata changes across completed sessions: "
                        f"{experiment_id}"
                    )
                metadata[experiment_id] = candidate_metadata
        checkpoint = _read_object(run_dir / "checkpoint.json")
        run_rows: list[dict[str, Any]] = []
        raw_files: list[dict[str, Any]] = []
        observed_raw_paths: set[str] = set()
        raw_directory = run_dir / "raw"
        expected_raw_paths = (
            {
                path.relative_to(run_dir).as_posix()
                for path in raw_directory.rglob("*")
                if path.is_file()
            }
            if raw_directory.is_dir()
            else set()
        )
        for block_key, entry in sorted(checkpoint["completed_blocks"].items()):
            if not isinstance(entry, dict):
                raise NativeEvidenceError(f"malformed checkpoint block {block_key}")
            relative_raw_path, raw_path = _resolve_checkpoint_raw_path(
                run_dir, entry.get("path")
            )
            if relative_raw_path in observed_raw_paths:
                raise NativeEvidenceError(
                    f"duplicate checkpoint raw path: {relative_raw_path}"
                )
            observed_raw_paths.add(relative_raw_path)
            if file_sha256(raw_path) != entry.get("sha256"):
                raise NativeEvidenceError(f"raw checksum mismatch: {raw_path}")
            block_rows = _read_jsonl(raw_path)
            if len(block_rows) != int(entry.get("rows", -1)):
                raise NativeEvidenceError(f"raw row count mismatch: {raw_path}")
            for row in block_rows:
                if row.get("run_id") != run_id or row.get("config_hash") != config_hash:
                    raise NativeEvidenceError(f"raw run/config identity mismatch: {raw_path}")
                if row.get("implementation_tree_sha256") != implementation_hash:
                    raise NativeEvidenceError(
                        f"raw implementation identity mismatch: {raw_path}"
                    )
                experiment = _experiment_id(row)
                if experiment not in split_query_maps:
                    raise NativeEvidenceError(
                        f"raw row names an unconfigured experiment: {raw_path}"
                    )
                if row.get("dataset_hash") != manifest["dataset_hashes"].get(
                    experiment
                ):
                    raise NativeEvidenceError(
                        f"raw dataset identity mismatch: {raw_path}"
                    )
                if row.get("split_id") != manifest["split_ids"].get(experiment):
                    raise NativeEvidenceError(f"raw split identity mismatch: {raw_path}")
                expected_phase = manifest.get("phase", effective.get("phase"))
                expected_role = manifest.get(
                    "evidence_role", effective.get("evidence_role")
                )
                if expected_phase is not None and row.get("phase") != expected_phase:
                    raise NativeEvidenceError(f"raw phase identity mismatch: {raw_path}")
                if (
                    expected_role is not None
                    and row.get("evidence_role") != expected_role
                ):
                    raise NativeEvidenceError(
                        f"raw evidence-role identity mismatch: {raw_path}"
                    )
            run_rows.extend(block_rows)
            raw_files.append(
                {
                    "path": relative_raw_path,
                    "sha256": str(entry["sha256"]),
                    "rows": len(block_rows),
                }
            )
        if observed_raw_paths != expected_raw_paths:
            missing = sorted(expected_raw_paths - observed_raw_paths)
            unexpected = sorted(observed_raw_paths - expected_raw_paths)
            raise NativeEvidenceError(
                "raw shard inventory coverage mismatch: "
                f"missing={missing}, unexpected={unexpected}"
            )
        _verify_run_query_coverage(
            run_rows,
            split_query_maps,
            effective,
            run_dir=run_dir,
        )
        rows.extend(run_rows)
        runs.append(
            {
                "run_id": run_id,
                "config_hash": config_hash,
                "source_config_object_hash": effective.get(
                    "source_config_object_hash"
                ),
                "source_config_file_sha256": effective.get(
                    "source_config_file_sha256"
                ),
                "implementation_tree_sha256": implementation_hash,
                "phase": manifest.get("phase", effective.get("phase")),
                "evidence_role": manifest.get(
                    "evidence_role", effective.get("evidence_role")
                ),
                "session_id": manifest.get("session_id"),
                "holdout_verification": holdout_verification,
                "raw_files": raw_files,
                "completion_sha256": file_sha256(run_dir / "COMPLETED.json"),
            }
        )
    if not rows:
        raise NativeEvidenceError("no completed native run contributed raw rows")
    return LoadedNativeEvidence(tuple(rows), tuple(runs), tuple(excluded), metadata)


def analyze_native_path(
    input_path: str | Path,
    *,
    method_roles: Mapping[str, str] | None = None,
    bootstrap_resamples: int = MIN_BOOTSTRAP_RESAMPLES,
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> dict[str, Any]:
    evidence = load_native_evidence(input_path)
    summary = analyze_native_rows(
        evidence.rows,
        method_roles=method_roles,
        experiment_metadata=evidence.experiment_metadata,
        bootstrap_resamples=bootstrap_resamples,
        bootstrap_seed=bootstrap_seed,
    )
    summary["runs"] = list(evidence.runs)
    summary["excluded_incomplete_runs"] = list(evidence.excluded_incomplete_runs)
    non_validation_runs: list[dict[str, Any]] = []
    validation_runs = 0
    for run in evidence.runs:
        declared = [
            str(value).strip().lower()
            for value in (run.get("phase"), run.get("evidence_role"))
            if value is not None and str(value).strip()
        ]
        if declared and all(value == "validation" for value in declared):
            validation_runs += 1
        else:
            non_validation_runs.append(
                {
                    "run_id": run.get("run_id"),
                    "phase": run.get("phase"),
                    "evidence_role": run.get("evidence_role"),
                }
            )
    implementation_hashes = sorted(
        {
            str(run["implementation_tree_sha256"])
            for run in evidence.runs
            if run.get("implementation_tree_sha256") is not None
        }
    )
    source_config_hashes = sorted(
        {
            str(run["source_config_object_hash"])
            for run in evidence.runs
            if run.get("source_config_object_hash") is not None
        }
    )
    missing_source_config_hash_runs = [
        run.get("run_id")
        for run in evidence.runs
        if run.get("source_config_object_hash") is None
    ]
    source_config_file_hashes = sorted(
        {
            str(run["source_config_file_sha256"])
            for run in evidence.runs
            if run.get("source_config_file_sha256") is not None
        }
    )
    missing_source_config_file_hash_runs = [
        run.get("run_id")
        for run in evidence.runs
        if run.get("source_config_file_sha256") is None
    ]
    holdout_verifications = [
        run.get("holdout_verification")
        for run in evidence.runs
        if run.get("holdout_verification") is not None
    ]
    holdout_manifest_hashes = sorted(
        {
            str(value["manifest_sha256"])
            for value in holdout_verifications
            if isinstance(value, dict) and value.get("manifest_sha256") is not None
        }
    )
    holdout_passed = (
        len(holdout_verifications) == validation_runs
        and validation_runs > 0
        and len(holdout_manifest_hashes) == 1
        and all(
            isinstance(value, dict) and value.get("passed") is True
            for value in holdout_verifications
        )
    )
    source_passed = (
        bool(evidence.runs)
        and validation_runs == 1
        and not non_validation_runs
        and len(implementation_hashes) == 1
        and len(source_config_hashes) == 1
        and not missing_source_config_hash_runs
        and len(source_config_file_hashes) == 1
        and not missing_source_config_file_hash_runs
        and holdout_passed
    )
    summary["source_integrity"] = {
        "completed_manifest_runs": len(evidence.runs),
        "completed_validation_runs": validation_runs,
        "non_validation_runs": non_validation_runs,
        "implementation_tree_sha256": implementation_hashes,
        "source_config_object_hash": source_config_hashes,
        "missing_source_config_hash_runs": missing_source_config_hash_runs,
        "source_config_file_sha256": source_config_file_hashes,
        "missing_source_config_file_hash_runs": missing_source_config_file_hash_runs,
        "holdout_manifest_sha256": holdout_manifest_hashes,
        "holdout_verifications": holdout_verifications,
        "holdout_registration_passed": holdout_passed,
        "unmanifested_input": not evidence.runs,
        "fail_closed_checks_applied": bool(evidence.runs),
        "validation_gate_source_passed": source_passed,
    }
    return summary


__all__ = [
    "DEFAULT_BOOTSTRAP_SEED",
    "LoadedNativeEvidence",
    "METRIC_FIELDS",
    "MIN_BOOTSTRAP_RESAMPLES",
    "NATIVE_ANALYSIS_SCHEMA_VERSION",
    "NativeEvidenceError",
    "analyze_native_path",
    "analyze_native_rows",
    "evaluate_validation_gate",
    "load_native_evidence",
    "method_role",
    "paired_speedup",
]

#!/usr/bin/env python3
"""Generate the Japanese Issue #3 decision report from saved native evidence."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path, PurePosixPath
import statistics
from typing import Any, Mapping, Sequence

import numpy as np

from searchability.artifacts import (
    atomic_write_bytes,
    file_sha256,
    implementation_tree_sha256,
    object_sha256,
)
from searchability.native import native_build_info
from searchability.native_analysis import load_native_evidence


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected object: {path}")
    return value


def _query_medians(
    rows: Sequence[Mapping[str, Any]], experiment: str, method: str, field: str
) -> dict[tuple[str, str, str, int], float]:
    grouped: dict[tuple[str, str, str, int], list[float]] = {}
    for row in rows:
        if row.get("experiment_id") != experiment or row.get("method") != method:
            continue
        value = row.get(field)
        if value is None:
            continue
        key = (
            str(row.get("run_id", "unspecified")),
            str(row.get("session_id", "unspecified")),
            str(row["query_id"]),
            int(row.get("query_position", -1)),
        )
        grouped.setdefault(key, []).append(float(value))
    return {key: float(statistics.median(values)) for key, values in grouped.items()}


def _percentiles(values: Sequence[float]) -> tuple[float | None, float | None, float | None]:
    if not values:
        return None, None, None
    array = np.asarray(values, dtype=np.float64) / 1_000_000.0
    return tuple(float(np.percentile(array, value)) for value in (50, 95, 99))  # type: ignore[return-value]


def _fmt(value: float | None, digits: int = 3) -> str:
    return "—" if value is None else f"{value:.{digits}f}"


def _comparison_lookup(
    summary: Mapping[str, Any], metric: str = "api_wall"
) -> dict[tuple[str, str, str], Mapping[str, Any]]:
    result: dict[tuple[str, str, str], Mapping[str, Any]] = {}
    for row in summary.get("comparisons", []):
        if row.get("metric") != metric:
            continue
        key = (
            str(row["experiment_id"]),
            str(row["proposed_method"]),
            str(row["baseline_role"]),
        )
        if key in result:
            raise RuntimeError(f"duplicate {metric} comparison: {key}")
        result[key] = row
    return result


def _nested_query_medians(
    rows: Sequence[Mapping[str, Any]],
    experiment: str,
    method: str,
    container: str,
    field: str,
) -> dict[tuple[str, str, str, int], float]:
    grouped: dict[tuple[str, str, str, int], list[float]] = {}
    for row in rows:
        if row.get("experiment_id") != experiment or row.get("method") != method:
            continue
        nested = row.get(container)
        if not isinstance(nested, dict) or nested.get(field) is None:
            continue
        value = float(nested[field])
        if not math.isfinite(value) or value < 0:
            raise RuntimeError(
                f"invalid component timing: {experiment}/{method}/{container}.{field}"
            )
        key = (
            str(row.get("run_id", "unspecified")),
            str(row.get("session_id", "unspecified")),
            str(row["query_id"]),
            int(row.get("query_position", -1)),
        )
        grouped.setdefault(key, []).append(value)
    return {key: float(statistics.median(values)) for key, values in grouped.items()}


def _requested_beta_by_method(
    rows: Sequence[Mapping[str, Any]], experiment: str, methods: Sequence[str]
) -> dict[str, float]:
    result: dict[str, float] = {}
    for method in methods:
        values = {
            float(row["requested_beta_l2"])
            for row in rows
            if row.get("experiment_id") == experiment
            and row.get("method") == method
            and row.get("requested_beta_l2") is not None
        }
        if len(values) != 1 or not all(
            math.isfinite(value) and value >= 0 for value in values
        ):
            raise RuntimeError(
                f"missing/inconsistent requested beta: {experiment}/{method}"
            )
        result[method] = next(iter(values))
    return result


def _main_methods(
    summary: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]
) -> tuple[str, str, str, list[str], str]:
    roles = summary.get("methods_by_experiment", {}).get("sift-initial")
    if not isinstance(roles, dict):
        raise RuntimeError("main sift-initial method roles are missing")

    def unique(role: str) -> str:
        methods = roles.get(role)
        if not isinstance(methods, list) or len(methods) != 1:
            raise RuntimeError(f"main sift-initial {role} method is not unique")
        return str(methods[0])

    f_method, a_method, n_method = (unique(role) for role in ("F", "A", "N"))
    p_values = roles.get("P")
    if not isinstance(p_values, list) or not p_values:
        raise RuntimeError("main sift-initial P methods are missing")
    p_methods = [str(value) for value in p_values]
    beta_by_method = _requested_beta_by_method(rows, "sift-initial", p_methods)
    beta_zero = [method for method, beta in beta_by_method.items() if beta == 0.0]
    if len(beta_zero) != 1:
        raise RuntimeError("main sift-initial beta=0 method is not unique")
    return f_method, a_method, n_method, p_methods, beta_zero[0]


def _component_median(
    rows: Sequence[Mapping[str, Any]], experiment: str, method: str, field: str
) -> float | None:
    values: list[float] = []
    for row in rows:
        if row.get("experiment_id") != experiment or row.get("method") != method:
            continue
        receipt = row.get("receipt")
        if not isinstance(receipt, dict) or receipt.get(field) is None:
            continue
        values.append(float(receipt[field]))
    return None if not values else float(statistics.median(values))


def _n_over_p(
    rows: Sequence[Mapping[str, Any]], experiment: str, p_method: str
) -> float | None:
    n_methods = {
        str(row["method"])
        for row in rows
        if row.get("experiment_id") == experiment and row.get("method_role") == "N"
    }
    if len(n_methods) != 1:
        return None
    n_values = _query_medians(rows, experiment, next(iter(n_methods)), "api_wall_latency_ns")
    p_values = _query_medians(rows, experiment, p_method, "api_wall_latency_ns")
    keys = set(n_values).intersection(p_values)
    if not keys:
        return None
    logs = [math.log(n_values[key] / p_values[key]) for key in keys]
    return math.exp(math.fsum(logs) / len(logs))


def _completed_artifacts(
    input_path: Path,
    source_runs: Sequence[Mapping[str, Any]],
) -> tuple[
    dict[str, list[Mapping[str, Any]]],
    dict[str, list[Mapping[str, Any]]],
    dict[str, list[Mapping[str, Any]]],
    list[dict[str, Any]],
]:
    """Load ancillary evidence from exactly the runs accepted by the analyzer."""

    absolute_root = input_path.absolute()
    if (
        input_path.is_symlink()
        or not input_path.is_dir()
        or input_path.resolve(strict=True) != absolute_root
    ):
        raise RuntimeError("native evidence root is missing or traverses a symlink")
    root = absolute_root
    expected: dict[str, Mapping[str, Any]] = {}
    for run in source_runs:
        run_id = str(run.get("run_id", ""))
        completion_sha = run.get("completion_sha256")
        if (
            not run_id
            or run_id in expected
            or not isinstance(completion_sha, str)
            or len(completion_sha) != 64
        ):
            raise RuntimeError("analyzer-accepted run identities are incomplete/duplicate")
        expected[run_id] = run
    if not expected:
        raise RuntimeError("analyzer accepted no completed runs")

    builds: dict[str, list[Mapping[str, Any]]] = {}
    audits: dict[str, list[Mapping[str, Any]]] = {}
    cache_audits: dict[str, list[Mapping[str, Any]]] = {}
    bound_runs: list[dict[str, Any]] = []
    observed: set[str] = set()
    completion_paths = sorted(root.rglob("COMPLETED.json"))
    for completion_path in completion_paths:
        run_dir = completion_path.parent
        manifest_path = run_dir / "run_manifest.json"
        if (
            completion_path.is_symlink()
            or not completion_path.is_file()
            or manifest_path.is_symlink()
            or not manifest_path.is_file()
        ):
            raise RuntimeError(f"completion/manifest is missing, non-regular, or symlinked: {run_dir}")
        try:
            completion_path.resolve(strict=True).relative_to(root)
            manifest_path.resolve(strict=True).relative_to(root)
        except (FileNotFoundError, ValueError) as error:
            raise RuntimeError(f"completion/manifest escapes evidence root: {run_dir}") from error
        manifest = _read(manifest_path)
        run_id = str(manifest.get("run_id", ""))
        accepted = expected.get(run_id)
        if accepted is None:
            raise RuntimeError(f"orphan/unaccepted COMPLETED.json found: {run_dir}")
        if run_id in observed:
            raise RuntimeError(f"duplicate COMPLETED.json for accepted run: {run_id}")
        if (
            file_sha256(completion_path) != accepted.get("completion_sha256")
            or manifest.get("config_hash") != accepted.get("config_hash")
            or manifest.get("implementation_tree_sha256")
            != accepted.get("implementation_tree_sha256")
        ):
            raise RuntimeError(f"completion is not bound to analyzer-accepted run: {run_id}")
        observed.add(run_id)
        completion = _read(completion_path)
        entries = completion.get("ancillary_files")
        if not isinstance(entries, list) or not entries:
            raise RuntimeError(f"completion has no ancillary inventory: {run_dir}")
        inventory: dict[str, Mapping[str, Any]] = {}
        for entry in entries:
            if not isinstance(entry, dict) or not isinstance(entry.get("path"), str):
                raise RuntimeError(f"malformed ancillary inventory: {run_dir}")
            relative_text = entry["path"]
            relative = PurePosixPath(relative_text)
            if (
                not relative_text
                or "\\" in relative_text
                or relative.is_absolute()
                or relative.as_posix() != relative_text
                or any(part in {"", ".", ".."} for part in relative.parts)
                or relative_text in inventory
            ):
                raise RuntimeError(f"unsafe/duplicate ancillary path: {entry.get('path')}")
            path = run_dir.joinpath(*relative.parts)
            try:
                path.resolve(strict=True).relative_to(run_dir.resolve(strict=True))
            except (FileNotFoundError, ValueError) as error:
                raise RuntimeError(f"ancillary path escapes run: {relative_text}") from error
            if (
                path.is_symlink()
                or not path.is_file()
                or path.stat().st_size != int(entry.get("bytes", -1))
                or file_sha256(path) != entry.get("sha256")
            ):
                raise RuntimeError(f"ancillary identity mismatch: {path}")
            inventory[relative_text] = entry
        build_path = run_dir / "build_manifest.json"
        if "build_manifest.json" not in inventory:
            raise RuntimeError(f"build manifest is not completion-bound: {run_dir}")
        payload = _read(build_path)
        for experiment, value in payload.items():
            if isinstance(value, dict):
                builds.setdefault(str(experiment), []).append(value)
        for relative in sorted(inventory):
            if not relative.startswith("audits/") or not relative.endswith(".native-lb.json"):
                continue
            audit = _read(run_dir / relative)
            experiment = str(audit.get("experiment_id", ""))
            if not experiment or audit.get("passed") is not True:
                raise RuntimeError(f"failed or unidentified LB audit: {relative}")
            build = payload.get(experiment)
            if not isinstance(build, dict) or (
                audit.get("dataset_hash") != build.get("dataset_hash")
                or audit.get("split_id") != build.get("split_id")
            ):
                raise RuntimeError(f"LB audit/build identity mismatch: {relative}")
            audits.setdefault(experiment, []).append(audit)
            continue
        for relative in sorted(inventory):
            if not relative.startswith("audits/") or not relative.endswith(".base-cache.json"):
                continue
            audit = _read(run_dir / relative)
            experiment = str(audit.get("experiment_id", ""))
            build = payload.get(experiment)
            if not experiment or not isinstance(build, dict) or (
                audit.get("dataset_hash") != build.get("dataset_hash")
                or audit.get("split_id") != build.get("split_id")
            ):
                raise RuntimeError(f"Base-cache audit/build identity mismatch: {relative}")
            cache_audits.setdefault(experiment, []).append(audit)

        raw_inventory: list[dict[str, Any]] = []
        for raw in accepted.get("raw_files", []):
            if not isinstance(raw, dict) or not isinstance(raw.get("path"), str):
                raise RuntimeError(f"malformed accepted raw inventory: {run_id}")
            relative_text = str(raw["path"])
            parsed = PurePosixPath(relative_text)
            if (
                "\\" in relative_text
                or parsed.is_absolute()
                or parsed.as_posix() != relative_text
                or any(part in {"", ".", ".."} for part in parsed.parts)
            ):
                raise RuntimeError(f"unsafe accepted raw path: {relative_text}")
            path = run_dir.joinpath(*parsed.parts)
            try:
                path.resolve(strict=True).relative_to(run_dir.resolve(strict=True))
            except (FileNotFoundError, ValueError) as error:
                raise RuntimeError(f"accepted raw path escapes run: {relative_text}") from error
            if path.is_symlink() or not path.is_file() or file_sha256(path) != raw.get("sha256"):
                raise RuntimeError(f"accepted raw identity mismatch: {path}")
            raw_inventory.append({**raw, "bytes": path.stat().st_size})
        bound_runs.append(
            {
                "run_id": run_id,
                "run_dir": run_dir.relative_to(root).as_posix() or ".",
                "completion_sha256": accepted["completion_sha256"],
                "raw_files": raw_inventory,
                "raw_inventory_sha256": object_sha256(raw_inventory),
            }
        )
    if observed != set(expected):
        raise RuntimeError(
            "COMPLETED.json set differs from analyzer-accepted runs: "
            f"missing={sorted(set(expected) - observed)}"
        )
    if not builds or not audits or not cache_audits:
        raise RuntimeError("completed build/LB audit evidence is missing")
    return builds, audits, cache_audits, sorted(bound_runs, key=lambda row: row["run_id"])


def _positive_query_values(
    rows: Sequence[Mapping[str, Any]], experiment: str, method: str, field: str
) -> dict[tuple[str, str, str, int], float]:
    values = _query_medians(rows, experiment, method, field)
    if not values or any(not math.isfinite(value) or value <= 0 for value in values.values()):
        raise RuntimeError(f"missing/invalid measured ablation: {experiment}/{method}/{field}")
    return values


def _paired_ablation(
    rows: Sequence[Mapping[str, Any]],
    *,
    experiment: str,
    before: str,
    after: str,
    field: str,
) -> tuple[float, float, float, int]:
    before_values = _positive_query_values(rows, experiment, before, field)
    after_values = _positive_query_values(rows, experiment, after, field)
    if set(before_values) != set(after_values):
        raise RuntimeError(f"unpaired measured ablation: {before} -> {after}")
    ratio = math.exp(
        math.fsum(math.log(before_values[key] / after_values[key]) for key in before_values)
        / len(before_values)
    )
    return (
        statistics.median(before_values.values()) / 1e6,
        statistics.median(after_values.values()) / 1e6,
        ratio,
        len(before_values),
    )


def _quality_and_rechecks(
    rows: Sequence[Mapping[str, Any]], experiment: str, method: str
) -> tuple[float, float, float | None]:
    selected = [
        row
        for row in rows
        if row.get("experiment_id") == experiment and row.get("method") == method
    ]
    recalls = [
        float(row["full_visible_exact_recall"])
        for row in selected
        if row.get("full_visible_exact_recall") is not None
    ]
    if not recalls or any(not math.isfinite(value) or not 0 <= value <= 1 for value in recalls):
        raise RuntimeError(f"missing/invalid ablation quality: {experiment}/{method}")
    rechecks = [
        float(row["exact_boundary_rechecks"])
        for row in selected
        if row.get("exact_boundary_rechecks") is not None
    ]
    if any(not math.isfinite(value) or value < 0 for value in rechecks):
        raise RuntimeError(f"invalid exact-recheck ablation: {experiment}/{method}")
    return statistics.median(recalls), min(recalls), (
        statistics.median(rechecks) if rechecks else None
    )


def _base_cache_metrics(
    cache_audits: Mapping[str, Sequence[Mapping[str, Any]]], experiment: str
) -> tuple[float, float, float, int]:
    evidence = cache_audits.get(experiment)
    if not evidence:
        raise RuntimeError(f"missing Base-cache measured ablation: {experiment}")
    cached: list[float] = []
    legacy: list[float] = []
    for audit_index, audit in enumerate(evidence):
        rows = audit.get("rows")
        count = audit.get("query_count")
        if isinstance(count, bool) or not isinstance(count, int) or count <= 0 or not isinstance(rows, list):
            raise RuntimeError(f"malformed Base-cache measured ablation: {experiment}")
        pairs: dict[tuple[str, int], dict[str, Mapping[str, Any]]] = {}
        for row in rows:
            if not isinstance(row, dict) or row.get("mode") not in {"cached", "legacy_rebuild"}:
                raise RuntimeError(f"malformed Base-cache row: {experiment}")
            key = (str(row.get("query_id")), int(row.get("query_position", -1)))
            mode = str(row["mode"])
            if mode in pairs.setdefault(key, {}):
                raise RuntimeError(f"duplicate Base-cache row: {experiment}/{audit_index}/{key}")
            pairs[key][mode] = row
        if len(pairs) != count:
            raise RuntimeError(f"Base-cache query count differs from rows: {experiment}")
        for key, pair in pairs.items():
            if set(pair) != {"cached", "legacy_rebuild"}:
                raise RuntimeError(f"unpaired Base-cache row: {experiment}/{key}")
            if (
                pair["cached"].get("candidate_hash") != pair["legacy_rebuild"].get("candidate_hash")
                or pair["cached"].get("candidate_keys") != pair["legacy_rebuild"].get("candidate_keys")
            ):
                raise RuntimeError(f"Base-cache ablation changed frozen C: {experiment}/{key}")
            cached_wall = float(pair["cached"].get("wall_ns", float("nan")))
            legacy_wall = float(pair["legacy_rebuild"].get("wall_ns", float("nan")))
            if not all(math.isfinite(value) and value > 0 for value in (cached_wall, legacy_wall)):
                raise RuntimeError(f"invalid Base-cache latency: {experiment}/{key}")
            cached.append(cached_wall)
            legacy.append(legacy_wall)
    return (
        statistics.median(cached) / 1e6,
        statistics.median(legacy) / 1e6,
        math.exp(math.fsum(math.log(old / new) for old, new in zip(legacy, cached)) / len(cached)),
        len(cached),
    )


def _ablation_table(
    rows: Sequence[Mapping[str, Any]],
    cache_audits: Mapping[str, Sequence[Mapping[str, Any]]],
    profile: Mapping[str, Any],
) -> list[str]:
    experiment = "sift-initial"
    p_before, p_after, p_ratio, p_n = _paired_ablation(
        rows,
        experiment=experiment,
        before="ablation_P_beta0_rescan_adaptive",
        after="native_P_beta0",
        field="micro_latency_ns",
    )
    f_before, f_after, f_ratio, f_n = _paired_ablation(
        rows,
        experiment=experiment,
        before="ablation_F_heap_all_exact",
        after="native_F",
        field="micro_latency_ns",
    )
    cached, legacy, cache_ratio, cache_n = _base_cache_metrics(cache_audits, experiment)
    old = profile.get("unprofiled_run", profile.get("matched_unprofiled_run"))
    if not isinstance(old, dict):
        raise RuntimeError("old P unprofiled ablation evidence is missing")
    old_micro = float(old.get("old_pruning_beta0_micro_milliseconds", float("nan")))
    old_e2e = float(old.get("old_pruning_beta0_e2e_milliseconds", float("nan")))
    new_micro = statistics.median(
        _positive_query_values(rows, experiment, "native_P_beta0", "micro_latency_ns").values()
    ) / 1e6
    new_api = statistics.median(
        _positive_query_values(rows, experiment, "native_P_beta0", "api_wall_latency_ns").values()
    ) / 1e6
    if not all(math.isfinite(value) and value > 0 for value in (old_micro, old_e2e)):
        raise RuntimeError("old P unprofiled measured latency is missing/invalid")
    a_ref, a, a_ratio, a_n = _paired_ablation(
        rows,
        experiment=experiment,
        before="faiss_A_reference",
        after="faiss_A",
        field="api_wall_latency_ns",
    )
    ref_recall, ref_recall_min, ref_rechecks = _quality_and_rechecks(
        rows, experiment, "faiss_A_reference"
    )
    a_recall, a_recall_min, a_rechecks = _quality_and_rechecks(rows, experiment, "faiss_A")
    ref_recheck_text = "0（非認証 float 順序、exact recheck なし）" if ref_rechecks is None else _fmt(ref_rechecks, 1)
    if a_rechecks is None:
        raise RuntimeError("faiss_A exact-recheck evidence is missing")
    old_n = profile.get("condition", {}).get("development_queries")
    if isinstance(old_n, bool) or not isinstance(old_n, int) or old_n <= 0:
        raise RuntimeError("old P development query denominator is missing/invalid")
    new_n = len(
        _positive_query_values(rows, experiment, "native_P_beta0", "micro_latency_ns")
    )
    return [
        f"| P rescan → heap | micro p50 ms | {_fmt(p_before)} | {_fmt(p_after)} | {_fmt(p_ratio)} | {p_n} paired session-query |",
        f"| F all-exact → adaptive | micro p50 ms | {_fmt(f_before)} | {_fmt(f_after)} | {_fmt(f_ratio)} | {f_n} paired session-query |",
        f"| Base legacy rebuild → cached | audit wall p50 ms | {_fmt(legacy)} | {_fmt(cached)} | {_fmt(cache_ratio)} | {cache_n} paired audit queries; candidate hash/key identical |",
        f"| old P → packed/native P | micro p50 ms | {_fmt(old_micro)} | {_fmt(new_micro)} | {_fmt(old_micro / new_micro)} | **nonpaired** development n={old_n} vs validation session-query n={new_n}; old E2E/new API={_fmt(old_e2e)}/{_fmt(new_api)} ms |",
        f"| A-reference → A | API p50 ms | {_fmt(a_ref)} | {_fmt(a)} | {_fmt(a_ratio)} | {a_n} paired session-query; recall med/min {_fmt(ref_recall, 4)}/{_fmt(ref_recall_min, 4)} → {_fmt(a_recall, 4)}/{_fmt(a_recall_min, 4)}; exact rechecks {ref_recheck_text} → {_fmt(a_rechecks, 1)} |",
    ]


def _quality_table(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    chosen: dict[tuple[str, str, str, str, int], Mapping[str, Any]] = {}
    for row in rows:
        if row.get("method_role") not in {"F", "A", "A-reference", "P"}:
            continue
        key = (
            str(row.get("experiment_id")),
            str(row.get("method")),
            str(row.get("run_id")),
            str(row.get("session_id")),
            int(row.get("query_position", -1)),
        )
        prior = chosen.get(key)
        if prior is None or int(row.get("repetition", -1)) < int(
            prior.get("repetition", -1)
        ):
            chosen[key] = row
    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    for row in chosen.values():
        grouped.setdefault(
            (str(row["experiment_id"]), str(row["method"])), []
        ).append(row)
    output = []
    for (experiment, method), values in sorted(grouped.items()):
        matches = [
            bool(row.get("same_c_order_match", row.get("baseline_quality_match")))
            for row in values
        ]
        recalls = [float(row["full_visible_exact_recall"]) for row in values if row.get("full_visible_exact_recall") is not None]
        captures = [float(row["delta_neighbor_capture"]) for row in values if row.get("delta_neighbor_capture") is not None]
        observed = [float(row["observed_beta_upper_l2"]) for row in values if row.get("observed_beta_upper_l2") is not None]
        certified = [float(row["certified_beta_l2"]) for row in values if row.get("certified_beta_l2") is not None]
        requested = [float(row["requested_beta_l2"]) for row in values if row.get("requested_beta_l2") is not None]
        output.append(
            "| " + " | ".join(
                (
                    experiment,
                    method,
                    str(len(values)),
                    f"{sum(matches) / len(matches):.4f}" if matches else "—",
                    f"{statistics.median(recalls):.4f}/{min(recalls):.4f}" if recalls else "—",
                    f"{statistics.median(captures):.4f}/{min(captures):.4f} ({len(captures)})" if captures else "— (0)",
                    (
                        f"{max(observed):.6g}/{max(certified):.6g}/{max(requested):.6g}"
                        if observed and certified and requested
                        else "—"
                    ),
                )
            ) + " |"
        )
    return output


def _single_candidate_comparison(
    candidate: Mapping[str, Any], role: str
) -> Mapping[str, Any]:
    outcomes = candidate.get("comparison_outcomes")
    outcome = outcomes.get(role) if isinstance(outcomes, dict) else None
    comparisons = outcome.get("comparisons") if isinstance(outcome, dict) else None
    if not isinstance(comparisons, list) or len(comparisons) != 1:
        raise RuntimeError(
            "candidate comparison is missing/non-unique: "
            f"{candidate.get('experiment_id')}/{candidate.get('proposed_method')}/{role}"
        )
    comparison = comparisons[0]
    if not isinstance(comparison, dict):
        raise RuntimeError("candidate comparison is malformed")
    return comparison


def _delta_influence_tables(
    gate: Mapping[str, Any],
) -> tuple[list[str], list[str], dict[str, int]]:
    """Render every eligible SIFT subset and the non-independent GIST anchors."""

    sift_rows: list[str] = []
    gist_rows: list[str] = []
    sift_f_lower_passes = 0
    sift_a_lower_passes = 0
    sift_a_upper_below_one = 0
    candidates = gate.get("candidates")
    if not isinstance(candidates, list):
        raise RuntimeError("validation gate candidate inventory is missing")
    for candidate in candidates:
        if not isinstance(candidate, dict) or candidate.get("real_non_degenerate") is not True:
            continue
        influence_n = candidate.get("delta_influence_queries")
        if isinstance(influence_n, bool) or not isinstance(influence_n, int) or influence_n < 0:
            raise RuntimeError("candidate Delta-influence denominator is invalid")
        is_sift = candidate.get("sift_primary_for_fresh_holdout") is True
        is_gist = str(candidate.get("experiment_id", "")).startswith("gist-")
        if not ((is_sift and influence_n > 0) or is_gist):
            continue
        role_values: dict[str, Mapping[str, Any]] = {}
        for role in ("F", "A"):
            comparison = _single_candidate_comparison(candidate, role)
            subset = comparison.get("delta_influence_subset")
            if not isinstance(subset, dict):
                raise RuntimeError(
                    "Delta-influence subset is missing: "
                    f"{candidate.get('experiment_id')}/{candidate.get('proposed_method')}/{role}"
                )
            try:
                values = {
                    name: float(subset[name])
                    for name in (
                        "geometric_mean",
                        "bootstrap_95pct_lower",
                        "bootstrap_95pct_upper",
                    )
                }
                paired_queries = int(subset["paired_queries"])
            except (KeyError, TypeError, ValueError) as error:
                raise RuntimeError("Delta-influence subset is malformed") from error
            if (
                not all(math.isfinite(value) and value > 0 for value in values.values())
                or paired_queries != influence_n
            ):
                raise RuntimeError("Delta-influence subset denominator/value mismatch")
            role_values[role] = {**values, "paired_queries": paired_queries}
        beta = float(candidate.get("requested_beta_l2", float("nan")))
        if not math.isfinite(beta) or beta < 0:
            raise RuntimeError("candidate beta is missing/invalid")
        row = (
            f"| {candidate.get('experiment_id')} | {candidate.get('proposed_method')} | "
            f"{_fmt(beta, 6)} | {influence_n} | "
            f"{_fmt(role_values['F']['geometric_mean'])} "
            f"[{_fmt(role_values['F']['bootstrap_95pct_lower'])}, "
            f"{_fmt(role_values['F']['bootstrap_95pct_upper'])}] | "
            f"{_fmt(role_values['A']['geometric_mean'])} "
            f"[{_fmt(role_values['A']['bootstrap_95pct_lower'])}, "
            f"{_fmt(role_values['A']['bootstrap_95pct_upper'])}] |"
        )
        (sift_rows if is_sift else gist_rows).append(row)
        if is_sift:
            sift_f_lower_passes += int(
                float(role_values["F"]["bootstrap_95pct_lower"]) > 1.0
            )
            sift_a_lower_passes += int(
                float(role_values["A"]["bootstrap_95pct_lower"]) > 1.0
            )
            sift_a_upper_below_one += int(
                float(role_values["A"]["bootstrap_95pct_upper"]) < 1.0
            )
    if not sift_rows or not gist_rows:
        raise RuntimeError("SIFT/GIST Delta-influence table is incomplete")
    return sift_rows, gist_rows, {
        "sift": len(sift_rows),
        "gist": len(gist_rows),
        "sift_f_ci_lower_above_one": sift_f_lower_passes,
        "sift_a_ci_lower_above_one": sift_a_lower_passes,
        "sift_a_ci_upper_below_one": sift_a_upper_below_one,
    }


def _main_latency_scope_table(
    summary: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]
) -> tuple[list[str], dict[str, int]]:
    f_method, a_method, _, _, p_method = _main_methods(summary, rows)
    scopes = (
        ("micro", "micro_latency_ns"),
        ("composed_e2e", "composed_e2e_latency_ns"),
        ("api_wall", "api_wall_latency_ns"),
    )
    output: list[str] = []
    process_sessions: set[tuple[str, str]] = set()
    paired_queries: set[int] = set()
    paired_observations: set[int] = set()
    sessions_per_query: set[int] = set()
    for scope, field in scopes:
        lookup = _comparison_lookup(summary, scope)
        f_comparison = lookup.get(("sift-initial", p_method, "F"))
        a_comparison = lookup.get(("sift-initial", p_method, "A"))
        if not isinstance(f_comparison, dict) or not isinstance(a_comparison, dict):
            raise RuntimeError(f"main {scope} F/A comparison is missing")
        method_percentiles: dict[str, tuple[float | None, float | None, float | None]] = {}
        method_keys: dict[str, set[tuple[str, str, str, int]]] = {}
        for method in (f_method, a_method, p_method):
            values = _query_medians(rows, "sift-initial", method, field)
            if not values or any(not math.isfinite(value) or value <= 0 for value in values.values()):
                raise RuntimeError(f"main {scope} latency is missing/invalid: {method}")
            method_percentiles[method] = _percentiles(list(values.values()))
            method_keys[method] = set(values)
        if not (
            method_keys[f_method] == method_keys[a_method] == method_keys[p_method]
        ):
            raise RuntimeError(f"main {scope} latency rows are not paired")
        process_sessions.update((key[0], key[1]) for key in method_keys[p_method])
        for comparison in (f_comparison, a_comparison):
            try:
                q_n = int(comparison["paired_queries"])
                observation_n = int(comparison["paired_session_query_observations"])
                session_range = comparison["sessions_per_query"]
                session_min = int(session_range["min"])
                session_max = int(session_range["max"])
            except (KeyError, TypeError, ValueError) as error:
                raise RuntimeError(f"main {scope} denominator is malformed") from error
            if (
                q_n <= 0
                or observation_n != len(method_keys[p_method])
                or not (0 < session_min <= session_max)
            ):
                raise RuntimeError(f"main {scope} denominator is inconsistent")
            paired_queries.add(q_n)
            paired_observations.add(observation_n)
            sessions_per_query.update((session_min, session_max))

        def latency_text(method: str) -> str:
            return "/".join(_fmt(value) for value in method_percentiles[method])

        output.append(
            f"| {scope} | {latency_text(f_method)} | {latency_text(a_method)} | "
            f"{latency_text(p_method)} | "
            f"{_fmt(f_comparison.get('geometric_mean'))} "
            f"[{_fmt(f_comparison.get('bootstrap_95pct_lower'))}, "
            f"{_fmt(f_comparison.get('bootstrap_95pct_upper'))}] | "
            f"{_fmt(a_comparison.get('geometric_mean'))} "
            f"[{_fmt(a_comparison.get('bootstrap_95pct_lower'))}, "
            f"{_fmt(a_comparison.get('bootstrap_95pct_upper'))}] | "
            f"{int(f_comparison['paired_queries'])}/"
            f"{int(f_comparison['paired_session_query_observations'])} |"
        )
    if len(paired_queries) != 1 or len(paired_observations) != 1:
        raise RuntimeError("main latency denominators differ across scopes/roles")
    return output, {
        "paired_queries": next(iter(paired_queries)),
        "paired_session_query_observations": next(iter(paired_observations)),
        "process_sessions": len(process_sessions),
        "sessions_per_query_min": min(sessions_per_query),
        "sessions_per_query_max": max(sessions_per_query),
    }


def _main_component_table(
    summary: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]
) -> list[str]:
    f_method, a_method, n_method, p_methods, _ = _main_methods(summary, rows)
    beta_by_method = _requested_beta_by_method(rows, "sift-initial", p_methods)
    methods = [f_method, n_method, a_method, *sorted(p_methods, key=beta_by_method.get)]
    output: list[str] = []
    for method in methods:
        micro = _query_medians(rows, "sift-initial", method, "micro_latency_ns")
        base_prepare = _query_medians(rows, "sift-initial", method, "base_prepare_wall_ns")
        method_prepare = _query_medians(
            rows, "sift-initial", method, "native_query_prepare_wall_ns"
        )
        if not micro or set(micro) != set(base_prepare) or set(micro) != set(method_prepare):
            raise RuntimeError(f"main component denominator mismatch: {method}")

        def nested(field: str, *, required: bool = True) -> dict[tuple[str, str, str, int], float]:
            values = _nested_query_medians(
                rows, "sift-initial", method, "component_timings_ns", field
            )
            if required and set(values) != set(micro):
                raise RuntimeError(f"main component is missing: {method}/{field}")
            return values

        if method == a_method:
            kernel_or_delta = nested("delta_faiss_search_ns")
            exact_or_merge = nested("shortlist_exact_merge_ns")
            lb, order, group_scan, raw_scan, receipt = ({}, {}, {}, {}, {})
            accounted = {
                key: kernel_or_delta[key] + exact_or_merge[key] for key in micro
            }
        else:
            kernel_or_delta = nested("kernel_total_ns")
            lb = nested("lb_calculation_ns")
            order = nested("group_ordering_ns")
            group_scan = nested("group_scan_ns")
            raw_scan = nested("raw_scan_ns")
            exact_or_merge = nested("adaptive_exact_ns")
            receipt = nested("receipt_ns")
            accounted = {
                key: kernel_or_delta[key] + exact_or_merge[key] + receipt[key]
                for key in micro
            }
        residual = {key: micro[key] - accounted[key] for key in micro}
        if any(not math.isfinite(value) or value < -1.0 for value in residual.values()):
            raise RuntimeError(f"main component timers overlap/exceed micro: {method}")

        def p50_ms(values: Mapping[Any, float]) -> float | None:
            return None if not values else statistics.median(values.values()) / 1e6

        role = next(
            (
                role_name
                for role_name, role_method in (
                    ("F", f_method),
                    ("N", n_method),
                    ("A", a_method),
                )
                if method == role_method
            ),
            "P",
        )
        beta_text = _fmt(beta_by_method[method], 6) if method in beta_by_method else "—"
        output.append(
            "| " + " | ".join(
                (
                    role,
                    method,
                    beta_text,
                    str(len(micro)),
                    _fmt(p50_ms(base_prepare)),
                    _fmt(p50_ms(method_prepare)),
                    _fmt(p50_ms(micro)),
                    _fmt(p50_ms(kernel_or_delta)),
                    _fmt(p50_ms(lb)),
                    _fmt(p50_ms(order)),
                    _fmt(p50_ms(group_scan)),
                    _fmt(p50_ms(raw_scan)),
                    _fmt(p50_ms(exact_or_merge)),
                    _fmt(p50_ms(receipt)),
                    _fmt(p50_ms(residual)),
                )
            ) + " |"
        )
    return output


def _main_positive_beta_table(
    summary: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]
) -> tuple[list[str], str]:
    _, _, _, p_methods, beta_zero_method = _main_methods(summary, rows)
    beta_by_method = _requested_beta_by_method(rows, "sift-initial", p_methods)
    ordered = sorted(p_methods, key=beta_by_method.get)
    if len(ordered) < 2 or beta_by_method[ordered[-1]] <= 0:
        raise RuntimeError("main positive-beta operating points are missing")
    metrics: dict[str, dict[str, float | int]] = {}
    output: list[str] = []
    for method in ordered:
        latency = _query_medians(rows, "sift-initial", method, "api_wall_latency_ns")
        skips = _nested_query_medians(
            rows, "sift-initial", method, "receipt", "groups_skipped"
        )
        scanned = _nested_query_medians(
            rows, "sift-initial", method, "receipt", "vectors_scanned"
        )
        if not latency or set(latency) != set(skips) or set(latency) != set(scanned):
            raise RuntimeError(f"positive-beta denominator mismatch: {method}")
        p50, p95, p99 = _percentiles(list(latency.values()))
        assert p50 is not None and p95 is not None and p99 is not None
        metrics[method] = {
            "n": len(latency),
            "p50": p50,
            "p95": p95,
            "p99": p99,
            "skips": statistics.median(skips.values()),
            "scanned": statistics.median(scanned.values()),
        }
    baseline = metrics[beta_zero_method]
    for method in ordered:
        value = metrics[method]

        def percent_change(name: str) -> float:
            denominator = float(baseline[name])
            if denominator <= 0:
                raise RuntimeError(f"positive-beta baseline is non-positive: {name}")
            return 100.0 * (float(value[name]) / denominator - 1.0)

        output.append(
            f"| {method} | {_fmt(beta_by_method[method], 6)} | {value['n']} | "
            f"{_fmt(float(value['p50']))}/{_fmt(float(value['p95']))}/"
            f"{_fmt(float(value['p99']))} | "
            f"{percent_change('p50'):+.3f}/{percent_change('p95'):+.3f}/"
            f"{percent_change('p99'):+.3f}% | {_fmt(float(value['skips']), 1)} | "
            f"{_fmt(float(value['scanned']), 1)} ({percent_change('scanned'):+.3f}%) |"
        )
    strongest = ordered[-1]
    strongest_values = metrics[strongest]
    api_lookup = _comparison_lookup(summary, "api_wall")
    a_comparison = api_lookup.get(("sift-initial", strongest, "A"))
    if not isinstance(a_comparison, dict):
        raise RuntimeError("positive-beta A/P comparison is missing")
    changed_pruning = (
        float(strongest_values["skips"]) > float(baseline["skips"])
        or float(strongest_values["scanned"]) < float(baseline["scanned"])
    )
    changed_latency = any(
        float(strongest_values[name]) < float(baseline[name])
        for name in ("p50", "p95", "p99")
    )
    positive_raw = [
        row
        for row in rows
        if row.get("method_role") == "P"
        and row.get("requested_beta_l2") is not None
        and float(row["requested_beta_l2"]) > 0
    ]
    if not positive_raw:
        raise RuntimeError("positive-beta P raw population is missing")
    gap_rows: list[tuple[float, float, float]] = []
    main_gap_rows: list[tuple[float, float, float]] = []
    for row in positive_raw:
        try:
            observed = float(row["observed_beta_upper_l2"])
            certified = float(row["certified_beta_l2"])
            requested = float(row["requested_beta_l2"])
        except (KeyError, TypeError, ValueError) as error:
            raise RuntimeError("positive-beta gap chain is missing/malformed") from error
        if (
            not all(
                math.isfinite(value) and value >= 0
                for value in (observed, certified, requested)
            )
            or not observed <= certified <= requested
        ):
            raise RuntimeError("positive-beta observed/certified/requested chain is invalid")
        gap = (observed, certified, requested)
        gap_rows.append(gap)
        if row.get("experiment_id") == "sift-initial":
            main_gap_rows.append(gap)
    if not main_gap_rows:
        raise RuntimeError("main positive-beta gap population is missing")
    nonzero_observed = sum(observed > 0 for observed, _, _ in gap_rows)
    max_observed = max(observed for observed, _, _ in main_gap_rows)
    max_certified = max(certified for _, certified, _ in main_gap_rows)
    max_requested = max(requested for _, _, requested in main_gap_rows)
    gap_tradeoff = (
        "observed gap は全て 0 であり、この実測 latency/pruning 差は"
        "観測された品質 gap との trade-off で得たものではない"
        if nonzero_observed == 0
        else "nonzero observed gap があり、latency/pruning 差を品質 gap と切り離しては解釈できない"
    )
    contribution = (
        "正の beta が pruning と少なくとも一つの latency percentile に作用した"
        if changed_pruning and changed_latency
        else "正の beta の pruning/latency 改善はこの比較では一貫して観測されなかった"
    )
    interpretation = (
        f"beta=0 から最大の保存済み正値 beta={_fmt(beta_by_method[strongest], 6)} "
        f"（session-query n={strongest_values['n']}）への比較では、{contribution}。"
        f"ただし同 operating point の A/P={_fmt(a_comparison.get('geometric_mean'))} "
        f"[CI {_fmt(a_comparison.get('bootstrap_95pct_lower'))}, "
        f"{_fmt(a_comparison.get('bootstrap_95pct_upper'))}] であり、"
        "この記述的寄与だけで gate 結論を変更しない。"
        f"全 positive-beta P raw の nonzero observed gap は "
        f"{nonzero_observed}/{len(gap_rows)} rows。主条件の positive-beta "
        f"raw rows（repetitionを含む）n={len(main_gap_rows)} における "
        "max observed/certified/requested="
        f"{_fmt(max_observed, 6)}/{_fmt(max_certified, 6)}/"
        f"{_fmt(max_requested, 6)} L2。{gap_tradeoff}"
        "（他条件での gap 不発生は主張しない）。"
    )
    return output, interpretation


def _geometry_table(
    audits: Mapping[str, Sequence[Mapping[str, Any]]]
) -> tuple[list[str], dict[str, Any]]:
    output: list[str] = []
    total_audits = total_queries = total_decisions = total_scan = total_skip = 0
    main_queries = main_decisions = 0
    scopes: set[str] = set()
    for experiment, evidence_rows in sorted(audits.items()):
        rows: list[Mapping[str, Any]] = []
        experiment_queries = 0
        for audit in evidence_rows:
            audit_rows = audit.get("rows")
            if not isinstance(audit_rows, list):
                raise RuntimeError(f"malformed LB audit rows: {experiment}")
            try:
                query_count = int(audit["query_count"])
                groups_checked = int(audit["groups_checked"])
                scanned = int(audit["scanned_groups_checked"])
                skipped = int(audit["skipped_groups_checked"])
            except (KeyError, TypeError, ValueError) as error:
                raise RuntimeError(f"malformed LB audit counts: {experiment}") from error
            query_keys = {
                (str(row.get("query_id")), int(row.get("query_position", -1)))
                for row in audit_rows
                if isinstance(row, dict)
            }
            action_scan = sum(
                isinstance(row, dict) and row.get("action") == "scan"
                for row in audit_rows
            )
            action_skip = sum(
                isinstance(row, dict) and row.get("action") == "skip"
                for row in audit_rows
            )
            if (
                query_count <= 0
                or groups_checked != len(audit_rows)
                or (bool(audit_rows) and len(query_keys) != query_count)
                or scanned != action_scan
                or skipped != action_skip
                or scanned + skipped != groups_checked
            ):
                raise RuntimeError(f"LB audit count/action mismatch: {experiment}")
            scope = audit.get("scope")
            if not isinstance(scope, str) or not scope:
                raise RuntimeError(f"LB audit scope is missing: {experiment}")
            scopes.add(scope)
            total_audits += 1
            total_queries += query_count
            total_decisions += groups_checked
            total_scan += scanned
            total_skip += skipped
            experiment_queries += query_count
            rows.extend(row for row in audit_rows if isinstance(row, dict))
        if experiment == "sift-initial":
            main_queries = experiment_queries
            main_decisions = len(rows)
        if not rows:
            output.append(
                f"| {experiment} | {experiment_queries} | none | 0 | — | — | — |"
            )
            continue
        for action in ("scan", "skip"):
            selected = [row for row in rows if row.get("action") == action]
            if not selected:
                continue
            slack = [
                max(0.0, float(row["exact_min_l2_lower"]) - float(row["lb_lower"]))
                for row in selected
            ]
            radius = [float(row["radius_upper"]) for row in selected]
            exact_min = [float(row["exact_min_l2_lower"]) for row in selected]
            output.append(
                "| " + " | ".join(
                    (
                        experiment,
                        str(experiment_queries),
                        action,
                        str(len(selected)),
                        f"{statistics.median(radius):.6g}",
                        f"{statistics.median(exact_min):.6g}",
                        f"{statistics.median(slack):.6g}/{float(np.percentile(slack, 95)):.6g}",
                    )
                ) + " |"
            )
    if not output or main_queries <= 0:
        raise RuntimeError("LB geometry audit coverage is incomplete")
    return output, {
        "audit_files": total_audits,
        "audited_query_sets": total_queries,
        "decisions": total_decisions,
        "scan_decisions": total_scan,
        "skip_decisions": total_skip,
        "main_queries": main_queries,
        "main_decisions": main_decisions,
        "scopes": sorted(scopes),
    }


def _break_even(
    rows: Sequence[Mapping[str, Any]],
    builds: Mapping[str, Sequence[Mapping[str, Any]]],
    experiment: str,
    f_method: str,
    p_method: str,
) -> dict[str, Any]:
    f_values = _query_medians(rows, experiment, f_method, "api_wall_latency_ns")
    p_values = _query_medians(rows, experiment, p_method, "api_wall_latency_ns")
    keys = sorted(set(f_values).intersection(p_values))
    saving_ns = (
        None
        if not keys
        else float(statistics.median(f_values[key] - p_values[key] for key in keys))
    )
    build_rows = list(builds.get(experiment, ()))
    build_costs = []
    packed_build_costs = []
    memory_costs = []
    combined_memory_costs = []
    for build in build_rows:
        groups = build.get("groups")
        if not isinstance(groups, dict):
            continue
        build_costs.append(
            sum(
                int(groups.get(field, 0))
                for field in (
                    "center_training_ns",
                    "assignment_ns",
                    "packing_and_radius_ns",
                )
            )
        )
        memory_costs.append(
            int(groups.get("member_bytes", 0)) + int(groups.get("metadata_bytes", 0))
        )
        packed = build.get("native_packed_view")
        memory = build.get("memory")
        if not isinstance(packed, dict) or not isinstance(memory, dict):
            continue
        packed_build_costs.append(int(packed.get("build_ns", 0)))
        combined_memory_costs.append(
            int(groups.get("member_bytes", 0))
            + int(groups.get("metadata_bytes", 0))
            + int(memory.get("packed_python_owned_bytes", 0))
            + int(memory.get("packed_native_owned_bytes", 0))
        )
    build_ns = None if not build_costs else float(statistics.median(build_costs))
    packed_build_ns = (
        None if not packed_build_costs else float(statistics.median(packed_build_costs))
    )
    memory_bytes = None if not memory_costs else float(statistics.median(memory_costs))
    combined_memory_bytes = (
        None
        if not combined_memory_costs
        else float(statistics.median(combined_memory_costs))
    )
    finite = build_ns is not None and saving_ns is not None and saving_ns > 0.0
    combined_finite = finite and packed_build_ns is not None
    return {
        "incremental_group_build_ns": build_ns,
        "packed_view_build_ns": packed_build_ns,
        "median_paired_query_saving_ns": saving_ns,
        "q_break_even": (build_ns / saving_ns) if finite else None,
        "q_break_even_group_plus_packed": (
            (build_ns + packed_build_ns) / saving_ns if combined_finite else None
        ),
        "finite": finite,
        "combined_finite": combined_finite,
        "paired_session_queries": len(keys),
        "group_memory_bytes": memory_bytes,
        "group_plus_packed_memory_bytes": combined_memory_bytes,
    }


def _main_build_table(
    builds: Mapping[str, Sequence[Mapping[str, Any]]],
    cache_audits: Mapping[str, Sequence[Mapping[str, Any]]],
) -> list[str]:
    build_rows = list(builds.get("sift-initial", ()))
    if not build_rows:
        raise RuntimeError("main build evidence is missing")

    def median_nested(container: str, field: str) -> float:
        values: list[float] = []
        for build in build_rows:
            nested = build.get(container)
            if not isinstance(nested, dict) or nested.get(field) is None:
                raise RuntimeError(f"main build metric is missing: {container}.{field}")
            value = float(nested[field])
            if not math.isfinite(value) or value < 0:
                raise RuntimeError(f"main build metric is invalid: {container}.{field}")
            values.append(value)
        return float(statistics.median(values))

    center = median_nested("groups", "center_training_ns")
    assignment = median_nested("groups", "assignment_ns")
    packing = median_nested("groups", "packing_and_radius_ns")
    packed_cold = median_nested("native_packed_view", "build_ns")
    packed_warm = median_nested("native_packed_view", "warm_cache_lookup_ns")
    group_member = median_nested("memory", "group_member_bytes")
    group_metadata = median_nested("memory", "group_metadata_bytes_estimate")
    packed_python = median_nested("memory", "packed_python_owned_bytes")
    packed_native = median_nested("memory", "packed_native_owned_bytes")
    rss_change = median_nested("memory", "rss_change_bytes")
    cached, legacy, cache_ratio, cache_n = _base_cache_metrics(
        cache_audits, "sift-initial"
    )
    cache_stats_rows = []
    for audit in cache_audits.get("sift-initial", ()):
        stats = audit.get("cache_stats")
        if not isinstance(stats, dict):
            raise RuntimeError("main Base-cache stats are missing")
        cache_stats_rows.append(stats)
    if not cache_stats_rows:
        raise RuntimeError("main Base-cache stats are missing")

    def median_stat(field: str) -> float:
        values = [float(stats[field]) for stats in cache_stats_rows if field in stats]
        if len(values) != len(cache_stats_rows) or any(
            not math.isfinite(value) or value < 0 for value in values
        ):
            raise RuntimeError(f"main Base-cache stat is invalid: {field}")
        return float(statistics.median(values))

    cache_build = median_stat("cache_build_ns")
    cache_lookup = median_stat("cache_lookup_ns")
    hits = median_stat("hits")
    misses = median_stat("misses")
    build_n = len(build_rows)
    return [
        f"| group centers | cold build | {_fmt(center / 1e6)} ms | build n={build_n}; Base-only training |",
        f"| group assignment | cold build | {_fmt(assignment / 1e6)} ms | build n={build_n} |",
        f"| group packing/radius | cold build | {_fmt(packing / 1e6)} ms | build n={build_n} |",
        f"| group total | cold build | {_fmt((center + assignment + packing) / 1e6)} ms | build n={build_n} |",
        f"| native packed view | cold build | {_fmt(packed_cold / 1e6)} ms | build n={build_n} |",
        f"| native packed view | warm cache identity lookup | {_fmt(packed_warm / 1e6, 6)} ms | build n={build_n} |",
        f"| Base visible map | cold cache build | {_fmt(cache_build / 1e6)} ms | audit files n={len(cache_stats_rows)}; misses median={_fmt(misses, 0)} |",
        f"| Base visible map | cached / legacy p50 | {_fmt(cached)}/{_fmt(legacy)} ms ({_fmt(cache_ratio)}x) | paired audit queries n={cache_n}; hits median={_fmt(hits, 0)}; total lookup median={_fmt(cache_lookup, 0)} ns |",
        f"| groups | retained memory | {_fmt(group_member + group_metadata, 0)} bytes | member + metadata; build n={build_n} |",
        f"| packed view | retained memory | {_fmt(packed_python + packed_native, 0)} bytes | Python + native owned; build n={build_n} |",
        f"| groups + packed view | retained memory scenario | {_fmt(group_member + group_metadata + packed_python + packed_native, 0)} bytes | additive sensitivity scenario; build n={build_n} |",
        f"| process | RSS change during build | {_fmt(rss_change, 0)} bytes | includes shared Base/Delta/truth/benchmark buffers; not P-only attribution |",
    ]


def _fallback_counts(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    fnp = [row for row in rows if row.get("method_role") in {"F", "N", "P"}]
    p_rows = [row for row in fnp if row.get("method_role") == "P"]
    if not fnp or not p_rows:
        raise RuntimeError("native F/N/P fallback population is missing")
    invalid = [
        row
        for row in fnp
        if row.get("python_fallback_used") not in {True, False}
    ]
    if invalid:
        raise RuntimeError("native fallback flags are missing/malformed")
    return {
        "fnp_rows": len(fnp),
        "fnp_fallbacks": sum(row.get("python_fallback_used") is True for row in fnp),
        "p_rows": len(p_rows),
        "p_fallbacks": sum(row.get("python_fallback_used") is True for row in p_rows),
    }


def _comparison_denominator_text(
    f_comparison: Mapping[str, Any], a_comparison: Mapping[str, Any]
) -> str:
    values: list[str] = []
    for role, comparison in (("F", f_comparison), ("A", a_comparison)):
        try:
            queries = int(comparison["paired_queries"])
            observations = int(comparison["paired_session_query_observations"])
        except (KeyError, TypeError, ValueError) as error:
            raise RuntimeError(f"{role} comparison denominator is missing") from error
        if queries <= 0 or observations < queries:
            raise RuntimeError(f"{role} comparison denominator is invalid")
        values.append(f"{role} {queries}/{observations}")
    return "; ".join(values)


def _rq_result_answers(
    *,
    gate: Mapping[str, Any],
    summary: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    builds: Mapping[str, Sequence[Mapping[str, Any]]],
    audits: Mapping[str, Sequence[Mapping[str, Any]]],
) -> tuple[str, str, str]:
    real_candidates = [
        value
        for value in gate.get("candidates", [])
        if isinstance(value, dict) and value.get("real_non_degenerate") is True
    ]
    gate_candidates = [
        value
        for value in real_candidates
        if value.get("sift_primary_for_fresh_holdout") is True
        and int(value.get("delta_influence_queries", 0)) > 0
    ]
    gist_candidates = [
        value
        for value in real_candidates
        if str(value.get("experiment_id", "")).startswith("gist-")
    ]
    main_candidates = [
        value
        for value in gate_candidates
        if value.get("experiment_id") == "sift-initial"
    ]
    secondary_candidates = [
        value
        for value in gate_candidates
        if value.get("experiment_id") != "sift-initial"
    ]

    def role_result(
        candidates: Sequence[Mapping[str, Any]], role: str
    ) -> tuple[int, list[tuple[float, Mapping[str, Any], Mapping[str, Any]]], int, int]:
        passed = 0
        comparisons: list[tuple[float, Mapping[str, Any], Mapping[str, Any]]] = []
        mismatch_rows = 0
        mismatch_queries = 0
        for candidate in candidates:
            outcomes = candidate.get("comparison_outcomes")
            outcome = outcomes.get(role) if isinstance(outcomes, dict) else None
            if not isinstance(outcome, dict):
                continue
            passed += int(outcome.get("passed") is True)
            values = outcome.get("comparisons")
            if not isinstance(values, list):
                continue
            for comparison in values:
                if not isinstance(comparison, dict):
                    continue
                value = comparison.get("geometric_mean")
                if value is None:
                    continue
                comparisons.append((float(value), candidate, comparison))
                mismatch_rows += int(
                    comparison.get("baseline_quality_mismatch_rows_retained", 0)
                )
                mismatch_queries += int(
                    comparison.get("baseline_quality_mismatch_queries_retained", 0)
                )
        return passed, comparisons, mismatch_rows, mismatch_queries

    def main_range(role: str) -> str:
        _, comparisons, _, _ = role_result(main_candidates, role)
        if not comparisons:
            return f"事前指定 main `sift-initial` の {role}/P comparison なし"
        values = [item[0] for item in comparisons]
        if role == "F":
            supported = sum(
                float(item[2].get("bootstrap_95pct_lower", float("-inf"))) > 1.0
                for item in comparisons
            )
            ci_text = f"95% CI lower > 1: {supported}/{len(comparisons)}"
        else:
            supported = sum(
                float(item[2].get("bootstrap_95pct_upper", float("inf"))) < 1.0
                for item in comparisons
            )
            ci_text = f"95% CI upper < 1: {supported}/{len(comparisons)}"
        return (
            f"事前指定 main `sift-initial` の {role}/P geomean 範囲は "
            f"{_fmt(min(values))}–{_fmt(max(values))}（{ci_text}）"
        )

    def secondary_maximum(role: str) -> str:
        _, comparisons, _, _ = role_result(secondary_candidates, role)
        if not comparisons:
            return "secondary sweep の比較なし"
        value, candidate, comparison = max(comparisons, key=lambda item: item[0])
        return (
            f"secondary sweep 上の事後的最大 {role}/P={_fmt(value)} "
            f"[CI {_fmt(comparison.get('bootstrap_95pct_lower'))}, "
            f"{_fmt(comparison.get('bootstrap_95pct_upper'))}] "
            f"(`{candidate.get('experiment_id')}` / "
            f"`{candidate.get('proposed_method')}`)は記述のみで、"
            "事前指定 main family の確認的要約とは区別する（事前登録済みの"
            "gate candidate 集合には含まれる）"
        )

    f_passed, _, _, _ = role_result(gate_candidates, "F")
    a_passed, _, mismatch_rows, mismatch_queries = role_result(gate_candidates, "A")
    gist_f_passed, _, _, _ = role_result(gist_candidates, "F")
    gist_a_passed, _, _, _ = role_result(gist_candidates, "A")
    if gate.get("gate_status") == "PASSED":
        validation_conclusion = "validation gate は `PASSED` し fresh final を許可した。"
    else:
        selected_candidate = json.dumps(
            gate.get("selected_candidate"), ensure_ascii=False, sort_keys=True
        )
        validation_conclusion = (
            "validation gate は `NOT_PASSED` で fresh final を許可しなかった。"
            f"`selected_candidate` は `{selected_candidate}`。"
        )
    rq1 = (
        f"{main_range('F')}。gate-eligible SIFT 候補 {len(gate_candidates)} 件中、"
        f"F/P criterion 通過は {f_passed} 件。real/non-degenerate の"
        f"記述対象は計 {len(real_candidates)} 件（GIST anchor {len(gist_candidates)} 件を含み、"
        f"その F/P 通過は {gist_f_passed}/{len(gist_candidates)}）。"
        f"{secondary_maximum('F')}。{validation_conclusion}"
    )
    rq2 = (
        f"{main_range('A')}。gate-eligible SIFT 候補 {len(gate_candidates)} 件中、"
        f"A/P timing+quality criterion 通過は {a_passed} 件。GIST anchor の"
        f"A/P 通過は {gist_a_passed}/{len(gist_candidates)}。{secondary_maximum('A')}。"
        f"gate-eligible SIFT の A mismatch は {mismatch_rows} rows / "
        f"{mismatch_queries} queries（timing から除外せず保持）。"
    )

    main_roles = summary.get("methods_by_experiment", {}).get("sift-initial", {})
    main_p_methods = [str(value) for value in main_roles.get("P", [])]
    n_over_p_values = [
        value
        for method in main_p_methods
        for value in (_n_over_p(rows, "sift-initial", method),)
        if value is not None and math.isfinite(value)
    ]
    skipped_group_medians = [
        value
        for method in main_p_methods
        for value in (_component_median(rows, "sift-initial", method, "groups_skipped"),)
        if value is not None and math.isfinite(value)
    ]
    group_counts = {
        int(receipt.get("groups_skipped", 0)) + int(receipt.get("groups_scanned", 0))
        for row in rows
        if row.get("experiment_id") == "sift-initial"
        and row.get("method") in main_p_methods
        and isinstance((receipt := row.get("receipt")), dict)
        and int(receipt.get("groups_skipped", 0))
        + int(receipt.get("groups_scanned", 0))
        > 0
    }
    geometry_parts: list[str] = []
    for action in ("scan", "skip"):
        gaps = [
            max(0.0, float(item["exact_min_l2_lower"]) - float(item["lb_lower"]))
            for audit in audits.get("sift-initial", [])
            for item in audit.get("rows", [])
            if isinstance(item, dict)
            and item.get("action") == action
            and item.get("exact_min_l2_lower") is not None
            and item.get("lb_lower") is not None
        ]
        if gaps:
            geometry_parts.append(
                f"{action} gap median/p95={_fmt(statistics.median(gaps))}/"
                f"{_fmt(float(np.percentile(gaps, 95)))} L2"
            )
    build_costs = [
        float(groups.get("center_training_ns", 0))
        + float(groups.get("assignment_ns", 0))
        + float(groups.get("packing_and_radius_ns", 0))
        for build in builds.get("sift-initial", [])
        if isinstance(build.get("groups"), dict)
        for groups in (build["groups"],)
    ]
    main_f_methods = [str(value) for value in main_roles.get("F", [])]
    break_even_results = (
        [
            _break_even(
                rows, builds, "sift-initial", main_f_methods[0], p_method
            )
            for p_method in main_p_methods
        ]
        if main_f_methods
        else []
    )
    finite_break_even_values = [
        float(value["q_break_even"])
        for value in break_even_results
        if value["finite"] and value["q_break_even"] is not None
    ]
    combined_break_even_values = [
        float(value["q_break_even_group_plus_packed"])
        for value in break_even_results
        if value["combined_finite"]
        and value["q_break_even_group_plus_packed"] is not None
    ]
    n_over_p_text = (
        "—"
        if not n_over_p_values
        else f"{_fmt(min(n_over_p_values))}–{_fmt(max(n_over_p_values))}"
    )
    group_total_text = (
        str(next(iter(group_counts))) if len(group_counts) == 1 else "condition別"
    )
    skipped_text = (
        "—"
        if not skipped_group_medians
        else f"{_fmt(min(skipped_group_medians), 1)}–{_fmt(max(skipped_group_medians), 1)}"
    )
    break_even_text = (
        "有限値なし"
        if not finite_break_even_values
        else f"{_fmt(min(finite_break_even_values), 1)}–{_fmt(max(finite_break_even_values), 1)} queries"
    )
    combined_break_even_text = (
        "有限値なし"
        if not combined_break_even_values
        else f"{_fmt(min(combined_break_even_values), 1)}–{_fmt(max(combined_break_even_values), 1)} queries"
    )
    main_query_counts = {
        len(_query_medians(rows, "sift-initial", method, "api_wall_latency_ns"))
        for method in main_p_methods
    }
    main_n_text = (
        str(next(iter(main_query_counts))) if len(main_query_counts) == 1 else "condition別"
    )
    geometry_text = "、".join(geometry_parts) if geometry_parts else "geometry gap evidence なし"
    rq3 = (
        f"事前指定 main `sift-initial` 内（各 operating pointの "
        f"session-query n={main_n_text}）で N/P geomean 範囲={n_over_p_text}、"
        f"P の operating-point別 median skipped groups={skipped_text}/{group_total_text}。"
        f"{geometry_text}。"
        f"incremental group build median="
        f"{_fmt(None if not build_costs else statistics.median(build_costs) / 1e6)} ms、"
        f"packed view を F/P 共通と扱う group-only の F 比有限 break-even は "
        f"{len(finite_break_even_values)}/{len(break_even_results)} operating points"
        f"（{break_even_text}）、packed build も P に課す感度分析は "
        f"{len(combined_break_even_values)}/{len(break_even_results)} points"
        f"（{combined_break_even_text}）。"
        "他 dataset/secondary axis は層別表の記述値とし、異なる次元・座標尺度の"
        "L2 gap を pooled aggregate しない。これは保存 evidence の記述的対応であり、"
        "単独では勝敗原因を因果確定しない。"
    )
    return rq1, rq2, rq3


def _final_session_lines(final_summary: Mapping[str, Any]) -> list[str]:
    metrics = final_summary.get("primary_session_metrics")
    fresh = final_summary.get("fresh_primary")
    if not isinstance(metrics, list) or not isinstance(fresh, dict):
        raise RuntimeError("completed final is missing primary per-seed metrics")
    try:
        by_seed = {int(row["session_seed"]): row for row in metrics if isinstance(row, dict)}
    except (KeyError, TypeError, ValueError) as error:
        raise RuntimeError("completed final has malformed primary per-seed metrics") from error
    if len(metrics) != 3 or set(by_seed) != {0, 1, 2}:
        raise RuntimeError("completed final primary per-seed metrics are not exactly 0/1/2")
    proposed = str(fresh.get("proposed_method", ""))
    output: list[str] = []
    for seed in (0, 1, 2):
        row = by_seed[seed]
        methods = row.get("methods")
        if (
            row.get("session_id") != f"session-seed-{seed}:process-0000"
            or int(row.get("query_count", -1)) != 1000
            or not isinstance(methods, dict)
            or set(methods) != {"F", "A", "P"}
            or methods.get("P") != proposed
            or not all(isinstance(methods.get(role), str) and methods.get(role) for role in ("F", "A"))
        ):
            raise RuntimeError("completed final primary per-seed identity is inconsistent")
        values: dict[str, float] = {}
        try:
            for role in ("F", "A", "P"):
                role_metrics = row[role]
                values[f"{role}50"] = float(role_metrics["api_wall_p50_ns"])
                values[f"{role}95"] = float(role_metrics["api_wall_p95_ns"])
            values["F/P"] = float(row["F_over_P_geomean"])
            values["A/P"] = float(row["A_over_P_geomean"])
        except (KeyError, TypeError, ValueError) as error:
            raise RuntimeError("completed final primary per-seed metrics are malformed") from error
        if any(not math.isfinite(value) or value <= 0 for value in values.values()):
            raise RuntimeError("completed final primary per-seed metrics are non-positive/non-finite")
        output.append(
            f"| {seed} | {_fmt(values['F50'] / 1e6)}/{_fmt(values['F95'] / 1e6)} | "
            f"{_fmt(values['A50'] / 1e6)}/{_fmt(values['A95'] / 1e6)} | "
            f"{_fmt(values['P50'] / 1e6)}/{_fmt(values['P95'] / 1e6)} | "
            f"{_fmt(values['F/P'])} | {_fmt(values['A/P'])} |"
        )
    return output


def _final_evidence(
    *,
    decision_path: Path,
    summary_path: Path,
    gate_path: Path,
    lock_path: Path,
    config_path: Path,
    pre_hnsw_authorization_path: Path,
    hnsw_path: Path,
    validation_summary_path: Path,
    validation_gate_path: Path,
) -> tuple[dict[str, Any], str]:
    """Verify optional large-final state and render its compact report section."""

    decision = _read(decision_path)
    validation_gate = _read(validation_gate_path)
    if decision.get("study_id") != "issue-3-native-recheck":
        raise RuntimeError("final decision study identity mismatch")
    if (
        decision.get("gate_sha256") != file_sha256(validation_gate_path)
        or decision.get("validation_summary_sha256")
        != file_sha256(validation_summary_path)
    ):
        raise RuntimeError("final decision is not bound to this validation evidence")
    status = str(decision.get("final_status", ""))
    if status == "NOT_RUN_GATE_NOT_PASSED":
        if (
            decision.get("validation_gate_status") != "NOT_PASSED"
            or validation_gate.get("gate_status") != "NOT_PASSED"
            or decision.get("performance_gate_status")
            != "NOT_RUN_GATE_NOT_PASSED"
            or decision.get("performance_verdict")
            != "NOT_SUPPORTED_IN_TESTED_REGIME"
            or decision.get("verdict") != "NOT_SUPPORTED_IN_TESTED_REGIME"
            or decision.get("engineering_decision") != "NO_GO"
            or decision.get("lock_created") is not False
            or decision.get("large_final_started") is not False
            or decision.get("fresh_sift_holdout_loaded") is not False
            or lock_path.exists()
            or config_path.exists()
            or summary_path.exists()
            or gate_path.exists()
            or pre_hnsw_authorization_path.exists()
            or hnsw_path.exists()
        ):
            raise RuntimeError("negative final decision violates the no-lock/no-load policy")
        selected_candidate = json.dumps(
            decision.get("selected_candidate"), ensure_ascii=False, sort_keys=True
        )
        return decision, (
            "## Fresh final\n\n"
            "Validation は `NOT_PASSED` で終了した。ここでの研究 verdict は"
            "事前規定の validation stopping rule によるものであり、"
            "fresh-final holdout estimate ではない。candidate lock と final run config は"
            "作成せず、"
            "SIFT fresh query 1200..2199 と final-only HNSW reference は読み込んでいない。"
            "工学的導入判定は `NO_GO` である。\n\n"
            "| final_decision field | value |\n"
            "|---|---|\n"
            f"| final_status | `{decision.get('final_status')}` |\n"
            f"| validation_gate_status | `{decision.get('validation_gate_status')}` |\n"
            f"| performance_gate_status | `{decision.get('performance_gate_status')}` |\n"
            f"| performance_verdict | `{decision.get('performance_verdict')}` |\n"
            f"| engineering_decision | `{decision.get('engineering_decision')}` |\n"
            f"| selected_candidate | `{selected_candidate}` |\n"
            f"| lock_created | `{str(decision.get('lock_created')).lower()}` |\n"
            f"| large_final_started | `{str(decision.get('large_final_started')).lower()}` |\n"
            f"| fresh_sift_holdout_loaded | `{str(decision.get('fresh_sift_holdout_loaded')).lower()}` |"
        )
    if status not in {
        "AUTHORIZED_NOT_YET_RUN",
        "COMPLETED_PASSED",
        "COMPLETED_NOT_PASSED",
    }:
        raise RuntimeError(f"unknown final decision status: {status}")
    if (
        decision.get("validation_gate_status") != "PASSED"
        or validation_gate.get("gate_status") != "PASSED"
        or decision.get("final_lock_sha256") != file_sha256(lock_path)
        or decision.get("final_config_file_sha256") != file_sha256(config_path)
    ):
        raise RuntimeError("authorized final lock/config identity mismatch")
    if status == "AUTHORIZED_NOT_YET_RUN":
        if (
            decision.get("verdict") != "INCONCLUSIVE"
            or decision.get("performance_verdict") != "INCONCLUSIVE"
            or decision.get("performance_gate_status") != "NOT_RUN"
            or decision.get("engineering_decision") != "NO_GO"
            or decision.get("lock_created") is not True
            or decision.get("large_final_started") is not False
            or decision.get("fresh_sift_holdout_loaded") is not False
            or summary_path.exists()
            or gate_path.exists()
            or pre_hnsw_authorization_path.exists()
            or hnsw_path.exists()
        ):
            raise RuntimeError("pending final decision violates the pre-final policy")
        return decision, (
            "## Fresh final\n\n"
            "Validation は `PASSED` し設定を lock 済みだが、3 process session の fresh "
            "final は未完了である。この時点の verdict は `INCONCLUSIVE`、工学的導入判定は"
            "未確定のため `NO_GO` とする。"
        )

    final_summary = _read(summary_path)
    final_gate = _read(gate_path)
    expected_gate_state = {
        "COMPLETED_PASSED": (
            "PASSED",
            True,
            "SUPPORTED_IN_TESTED_REGIME",
        ),
        "COMPLETED_NOT_PASSED": (
            "NOT_PASSED",
            False,
            "NOT_SUPPORTED_IN_TESTED_REGIME",
        ),
    }[status]
    expected_hnsw_status = (
        "completed"
        if status == "COMPLETED_PASSED"
        else "not_run_performance_gate_not_passed"
    )
    if status == "COMPLETED_PASSED":
        hnsw_identity_ok = (
            isinstance(decision.get("hnsw_completion_sha256"), str)
            and final_gate.get("hnsw_completion_sha256")
            == decision.get("hnsw_completion_sha256")
            and pre_hnsw_authorization_path.is_file()
            and decision.get("pre_hnsw_authorization_sha256")
            == file_sha256(pre_hnsw_authorization_path)
            and final_gate.get("pre_hnsw_authorization_sha256")
            == decision.get("pre_hnsw_authorization_sha256")
            and hnsw_path.is_dir()
        )
    else:
        hnsw_identity_ok = (
            "hnsw_completion_sha256" not in decision
            and "hnsw_completion_sha256" not in final_gate
            and "pre_hnsw_authorization_sha256" not in decision
            and "pre_hnsw_authorization_sha256" not in final_gate
            and not pre_hnsw_authorization_path.exists()
            and not hnsw_path.exists()
        )
    if (
        decision.get("final_summary_sha256") != file_sha256(summary_path)
        or decision.get("final_gate_sha256") != file_sha256(gate_path)
        or final_summary.get("study_id") != "issue-3-native-recheck"
        or final_summary.get("evidence_role") != "final"
        or final_gate.get("study_id") != "issue-3-native-recheck"
        or final_summary.get("final_gate") != final_gate
        or final_gate.get("verdict") != decision.get("verdict")
        or final_gate.get("performance_verdict")
        != decision.get("performance_verdict")
        or final_gate.get("performance_gate_status")
        != decision.get("performance_gate_status")
        or final_gate.get("engineering_decision")
        != decision.get("engineering_decision")
        or (
            final_gate.get("gate_status"),
            final_gate.get("passed"),
            final_gate.get("verdict"),
        )
        != expected_gate_state
        or final_gate.get("performance_gate_status") != expected_gate_state[0]
        or final_gate.get("performance_verdict") != expected_gate_state[2]
        or final_gate.get("engineering_decision") not in {"GO", "NO_GO"}
        or decision.get("lock_created") is not True
        or decision.get("large_final_started") is not True
        or decision.get("fresh_sift_holdout_loaded") is not True
        or final_summary.get("final_lock_sha256")
        != decision.get("final_lock_sha256")
        or final_summary.get("final_config_file_sha256")
        != decision.get("final_config_file_sha256")
        or final_summary.get("final_config_object_sha256")
        != decision.get("final_config_object_sha256")
        or final_summary.get("validation_gate_sha256")
        != decision.get("gate_sha256")
        or final_summary.get("validation_summary_sha256")
        != decision.get("validation_summary_sha256")
        or final_gate.get("final_lock_sha256")
        != decision.get("final_lock_sha256")
        or final_gate.get("final_config_file_sha256")
        != decision.get("final_config_file_sha256")
        or final_gate.get("hnsw_reference_status") != expected_hnsw_status
        or decision.get("hnsw_reference_status") != expected_hnsw_status
        or not hnsw_identity_ok
        or final_gate.get("gist_used_in_gate") is not False
        or final_gate.get("hnsw_references_used_in_gate") is not False
        or final_gate.get("fresh_query_interval") != [1200, 2200]
        or final_gate.get("process_session_seeds") != [0, 1, 2]
    ):
        raise RuntimeError("completed final summary/gate identity mismatch")
    fresh = final_summary.get("fresh_primary")
    sessions = final_summary.get("sessions")
    gist = final_summary.get("gist_interpretation")
    hnsw = final_summary.get("hnsw_references")
    if (
        not isinstance(fresh, dict)
        or fresh.get("query_interval") != [1200, 2200]
        or fresh.get("independence_label") != "fresh_pre_registered_holdout"
        or not isinstance(sessions, list)
        or sorted(int(row["session_seed"]) for row in sessions) != [0, 1, 2]
        or not isinstance(gist, dict)
        or gist.get("independence_label") != "reused_non_independent"
        or gist.get("gate_eligible") is not False
        or not isinstance(hnsw, dict)
        or hnsw.get("status") != expected_hnsw_status
        or hnsw.get("role") != "approximate_reference_only_not_in_final_gate"
        or (
            status == "COMPLETED_PASSED"
            and hnsw.get("completion_sha256")
            != decision.get("hnsw_completion_sha256")
        )
        or (
            status == "COMPLETED_PASSED"
            and hnsw.get("pre_hnsw_authorization_sha256")
            != decision.get("pre_hnsw_authorization_sha256")
        )
        or (
            status == "COMPLETED_NOT_PASSED"
            and (
                "completion_sha256" in hnsw
                or "pre_hnsw_authorization_sha256" in hnsw
            )
        )
    ):
        raise RuntimeError("completed final scope/independence identity mismatch")
    session_lines = _final_session_lines(final_summary)
    outcomes = final_gate.get("comparison_outcomes")
    if not isinstance(outcomes, dict) or set(outcomes) != {"F", "A"}:
        raise RuntimeError("completed final has no F/A outcomes")
    influence = final_gate.get("fresh_delta_influence")
    if not isinstance(influence, dict):
        raise RuntimeError("completed final has no Delta-influence evidence")
    for role in ("F", "A"):
        outcome = outcomes[role]
        try:
            ci_lower = float(outcome["bootstrap_95pct_lower"])
            ci_threshold = float(
                outcome["bootstrap_95pct_lower_must_be_strictly_above"]
            )
            speedup = float(outcome["geometric_mean_speedup"])
            minimum_speedup = float(outcome["minimum_required_speedup"])
            p95_ratio = float(outcome["P_over_baseline_p95_ratio"])
            maximum_p95_ratio = float(outcome["maximum_allowed_p95_ratio"])
        except (KeyError, TypeError, ValueError) as error:
            raise RuntimeError("completed final has malformed performance metrics") from error
        if not all(
            math.isfinite(value)
            for value in (
                ci_lower,
                ci_threshold,
                speedup,
                minimum_speedup,
                p95_ratio,
                maximum_p95_ratio,
            )
        ):
            raise RuntimeError("completed final has non-finite performance metrics")
        ci_passed = ci_lower > ci_threshold
        quality_passed = role == "F" or outcome.get("A_quality_match") is True
        expected_performance = (
            outcome.get("complete_query_session_pairs") is True
            and ci_passed
            and quality_passed
        )
        expected_engineering = (
            expected_performance
            and speedup >= minimum_speedup
            and p95_ratio <= maximum_p95_ratio
        )
        if (
            ci_threshold != 1.0
            or outcome.get("bootstrap_CI_supports_speedup") is not ci_passed
            or outcome.get("performance_gate_passed") is not expected_performance
            or outcome.get("engineering_GO_passed") is not expected_engineering
        ):
            raise RuntimeError("completed final performance metric decision is inconsistent")
    performance_passed = all(
        row.get("performance_gate_passed") is True for row in outcomes.values()
    )
    engineering_go = (
        performance_passed
        and all(row.get("engineering_GO_passed") is True for row in outcomes.values())
        and influence.get("evidence_complete") is True
        and final_gate.get("build_cost_evidence_complete") is True
    )
    expected_engineering_reasons: list[str] = []
    for role in ("F", "A"):
        outcome = outcomes[role]
        if outcome.get("performance_gate_passed") is not True:
            expected_engineering_reasons.append(f"performance_gate_failed_vs_{role}")
        elif outcome.get("engineering_GO_passed") is not True:
            expected_engineering_reasons.append(
                f"speedup_or_p95_requirement_failed_vs_{role}"
            )
    if influence.get("evidence_complete") is not True:
        expected_engineering_reasons.append("fresh_delta_influence_subset_missing")
    if final_gate.get("build_cost_evidence_complete") is not True:
        expected_engineering_reasons.append("build_cost_evidence_missing")
    if (
        performance_passed is not bool(final_gate["passed"])
        or ("GO" if engineering_go else "NO_GO")
        != final_gate["engineering_decision"]
        or final_gate.get("engineering_reasons") != expected_engineering_reasons
        or int(influence.get("labelled_queries", -1)) != 1000
        or int(influence.get("required_minimum", -1)) < 1
        or int(influence.get("influenced_queries", -1))
        < int(influence.get("required_minimum", -1))
    ):
        raise RuntimeError("completed performance/engineering decision is inconsistent")
    comparison_lines = []
    for role in ("F", "A"):
        row = outcomes[role]
        comparison_lines.append(
            "| " + " | ".join(
                (
                    role,
                    _fmt(row.get("geometric_mean_speedup")),
                    _fmt(row.get("P_query_folded_p95_ns") / 1e6),
                    _fmt(row.get("baseline_query_folded_p95_ns") / 1e6),
                    _fmt(row.get("P_over_baseline_p95_ratio")),
                    "PASS" if row.get("performance_gate_passed") else "FAIL",
                    "GO" if row.get("engineering_GO_passed") else "NO_GO",
                )
            ) + " |"
        )
    engineering_decision = str(final_gate["engineering_decision"])
    engineering_reasons = final_gate["engineering_reasons"]
    reasons_text = (
        "なし" if not engineering_reasons else ", ".join(map(str, engineering_reasons))
    )
    hnsw_text = (
        "Base+Delta HNSW と全体 HNSW（efSearch 128/512）も実行したが、非認証参考値であり gate 非対象である。"
        if expected_hnsw_status == "completed"
        else "performance gate が通らなかったため、final-only HNSW reference は実行していない。"
    )
    return decision, f"""## Fresh final

SIFT query 1200..2199（事前登録 fresh 1,000件）を primary とし、process/group-build/method-order seed 0/1/2 の3 sessionを query 単位で fold した。GIST 0..999 は `reused_non_independent` の補助 anchor であり gate には使っていない。{hnsw_text}

科学的 performance verdict は `{final_gate['performance_verdict']}`、工学的導入判定は **`{engineering_decision}`**（理由: {reasons_text}）である。`SUPPORTED_IN_TESTED_REGIME` と `NO_GO` は両立し得る。前者は F/A 両比較の bootstrap CI 下端が 1 を上回ること、後者はさらに geomean 1.10、p95 比 1.05、Delta-influence subset、build-cost evidence を要求する。

| baseline | baseline/P geomean | P p95 ms | baseline p95 ms | P/baseline p95 | performance | engineering |
|---|---:|---:|---:|---:|---|---|
{chr(10).join(comparison_lines)}

session 間 fold 前の primary F/A/P 実測変動（各 seed 内は query/repetition median、speedup は query-paired geomean）:

| seed | F p50/p95 ms | A p50/p95 ms | P p50/p95 ms | F/P geomean | A/P geomean |
|---:|---:|---:|---:|---:|---:|
{chr(10).join(session_lines)}

Fresh-final performance gate: **`{final_gate['performance_gate_status']}`**、verdict: **`{decision['verdict']}`**、engineering: **`{engineering_decision}`**。
"""


def _verify_final_decision_correctness(
    decision: Mapping[str, Any], correctness_path: Path
) -> None:
    expected = decision.get("correctness_sha256")
    if not isinstance(expected, str) or expected != file_sha256(correctness_path):
        raise RuntimeError("final decision is not bound to current correctness evidence")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--summary", required=True, type=Path)
    parser.add_argument("--gate", required=True, type=Path)
    parser.add_argument("--correctness", required=True, type=Path)
    parser.add_argument("--profile", required=True, type=Path)
    parser.add_argument("--representative-raw", required=True, type=Path)
    parser.add_argument("--final-decision", required=True, type=Path)
    parser.add_argument("--final-summary", required=True, type=Path)
    parser.add_argument("--final-gate", required=True, type=Path)
    parser.add_argument("--final-lock", required=True, type=Path)
    parser.add_argument("--final-config", required=True, type=Path)
    parser.add_argument("--pre-hnsw-authorization", required=True, type=Path)
    parser.add_argument("--final-hnsw", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    evidence = load_native_evidence(args.input)
    summary = _read(args.summary)
    gate = _read(args.gate)
    correctness = _read(args.correctness)
    profile = _read(args.profile)
    representative = _read(args.representative_raw)
    final_decision, final_section = _final_evidence(
        decision_path=args.final_decision,
        summary_path=args.final_summary,
        gate_path=args.final_gate,
        lock_path=args.final_lock,
        config_path=args.final_config,
        pre_hnsw_authorization_path=args.pre_hnsw_authorization,
        hnsw_path=args.final_hnsw,
        validation_summary_path=args.summary,
        validation_gate_path=args.gate,
    )
    _verify_final_decision_correctness(final_decision, args.correctness)
    rows = list(evidence.rows)
    comparisons = _comparison_lookup(summary)
    methods_by_experiment = summary["methods_by_experiment"]
    for payload, label in (
        (summary, "summary"),
        (gate, "gate"),
        (correctness, "correctness"),
        (representative, "representative raw"),
    ):
        if payload.get("study_id") != "issue-3-native-recheck":
            raise RuntimeError(f"{label} study identity mismatch")
    if summary.get("validation_gate") != gate:
        raise RuntimeError("gate differs from the decision embedded in summary")
    if representative.get("source_summary_sha256") != file_sha256(args.summary):
        raise RuntimeError("representative raw is not bound to this summary")
    if representative.get("source_gate_sha256") != file_sha256(args.gate):
        raise RuntimeError("representative raw is not bound to this gate")
    if representative.get("source_runs") != list(evidence.runs):
        raise RuntimeError("representative raw run identities differ from input evidence")
    if summary.get("runs") != list(evidence.runs):
        raise RuntimeError("summary run identities differ from input evidence")
    implementation_hashes = summary.get("source_integrity", {}).get(
        "implementation_tree_sha256", []
    )
    if implementation_hashes != [correctness.get("implementation_tree_sha256")]:
        raise RuntimeError("correctness and validation implementation identities differ")
    binary_hashes = summary.get("integrity_checks", {}).get(
        "native_backend_calls", {}
    ).get("native_binary_sha256", [])
    if binary_hashes != [correctness.get("native_shared_object_sha256")]:
        raise RuntimeError("correctness and validation native binary identities differ")
    repository_root = Path(__file__).resolve().parents[1]
    current_implementation_hash = implementation_tree_sha256(repository_root)
    current_native_hash = str(native_build_info().get("shared_object_sha256", ""))
    if correctness.get("implementation_tree_sha256") != current_implementation_hash:
        raise RuntimeError("current implementation differs from correctness evidence")
    if correctness.get("native_shared_object_sha256") != current_native_hash:
        raise RuntimeError("current native binary differs from correctness evidence")
    if implementation_hashes != [current_implementation_hash] or binary_hashes != [current_native_hash]:
        raise RuntimeError("current runtime differs from validation source identity")
    if final_decision.get("final_status") != "NOT_RUN_GATE_NOT_PASSED":
        final_lock = _read(args.final_lock)
        if (
            final_lock.get("implementation_tree_sha256") != current_implementation_hash
            or final_lock.get("native_shared_object_sha256") != current_native_hash
            or final_lock.get("implementation_tree_sha256")
            != correctness.get("implementation_tree_sha256")
            or final_lock.get("native_shared_object_sha256")
            != correctness.get("native_shared_object_sha256")
        ):
            raise RuntimeError("final lock differs from current/correctness runtime identity")
    if profile.get("issue") != 3 or not profile.get("profiled_implementation_tree_sha256"):
        raise RuntimeError("old-path profile identity is incomplete")
    required_experiments = {
        "sift-initial", "gist-initial", "sift-delta-0", "sift-delta-1000",
        "sift-delta-100000", "sift-groups-64", "sift-groups-512",
        "sift-k-1", "sift-k-100", "synthetic-clustered",
        "synthetic-high-dimensional-isotropic", "synthetic-delta-near-queries",
        "synthetic-outlier-radius",
    }
    missing_experiments = required_experiments - set(methods_by_experiment)
    if missing_experiments:
        raise RuntimeError(
            f"report refuses an incomplete validation matrix: {sorted(missing_experiments)}"
        )
    builds, audits, cache_audits, bound_runs = _completed_artifacts(
        args.input, evidence.runs
    )
    ablation_rows = _ablation_table(rows, cache_audits, profile)
    quality_rows = _quality_table(rows)
    geometry_rows, geometry_coverage = _geometry_table(audits)
    delta_sift_rows, delta_gist_rows, delta_counts = _delta_influence_tables(gate)
    main_latency_rows, main_latency_coverage = _main_latency_scope_table(summary, rows)
    main_component_rows = _main_component_table(summary, rows)
    positive_beta_rows, positive_beta_interpretation = _main_positive_beta_table(
        summary, rows
    )
    main_build_rows = _main_build_table(builds, cache_audits)
    fallback_counts = _fallback_counts(rows)
    rq1_answer, rq2_answer, rq3_answer = _rq_result_answers(
        gate=gate,
        summary=summary,
        rows=rows,
        builds=builds,
        audits=audits,
    )

    table: list[str] = []
    break_even_rows: list[str] = []
    for experiment, roles in sorted(methods_by_experiment.items()):
        f_method = roles.get("F", [None])[0]
        a_method = roles.get("A", [None])[0]
        p_methods = roles.get("P", [])
        for p_method in p_methods:
            p_rows = [
                row for row in rows
                if row.get("experiment_id") == experiment and row.get("method") == p_method
            ]
            beta = float(p_rows[0]["requested_beta_l2"]) if p_rows else None
            p50, p95, p99 = _percentiles(
                list(_query_medians(rows, experiment, p_method, "api_wall_latency_ns").values())
            )
            f_cmp = comparisons.get((experiment, p_method, "F"), {})
            a_cmp = comparisons.get((experiment, p_method, "A"), {})
            skipped = _component_median(rows, experiment, p_method, "groups_skipped")
            scanned = _component_median(rows, experiment, p_method, "vectors_scanned")
            table.append(
                "| " + " | ".join(
                    (
                        experiment,
                        p_method,
                        _fmt(beta, 6),
                        f"{_fmt(p50)}/{_fmt(p95)}/{_fmt(p99)}",
                        f"{_fmt(f_cmp.get('geometric_mean'))} [{_fmt(f_cmp.get('bootstrap_95pct_lower'))}, {_fmt(f_cmp.get('bootstrap_95pct_upper'))}]",
                        f"{_fmt(a_cmp.get('geometric_mean'))} [{_fmt(a_cmp.get('bootstrap_95pct_lower'))}, {_fmt(a_cmp.get('bootstrap_95pct_upper'))}]",
                        _fmt(_n_over_p(rows, experiment, p_method)),
                        f"{_fmt(skipped, 1)} / {_fmt(scanned, 1)}",
                        _comparison_denominator_text(f_cmp, a_cmp),
                    )
                ) + " |"
            )
            if f_method is not None:
                amortization = _break_even(
                    rows, builds, experiment, str(f_method), p_method
                )
                q_text = (
                    _fmt(amortization["q_break_even"], 1)
                    if amortization["finite"]
                    else "有限解なし"
                )
                break_even_rows.append(
                    "| " + " | ".join(
                        (
                            experiment,
                            p_method,
                            _fmt(
                                None
                                if amortization["incremental_group_build_ns"] is None
                                else amortization["incremental_group_build_ns"] / 1e6,
                                3,
                            ),
                            _fmt(
                                None
                                if amortization["packed_view_build_ns"] is None
                                else amortization["packed_view_build_ns"] / 1e6,
                                3,
                            ),
                            _fmt(
                                None
                                if amortization["median_paired_query_saving_ns"] is None
                                else amortization["median_paired_query_saving_ns"] / 1e6,
                                6,
                            ),
                            q_text,
                            (
                                _fmt(amortization["q_break_even_group_plus_packed"], 1)
                                if amortization["combined_finite"]
                                else "有限解なし"
                            ),
                            str(amortization["paired_session_queries"]),
                            _fmt(amortization["group_memory_bytes"], 0),
                            _fmt(amortization["group_plus_packed_memory_bytes"], 0),
                        )
                    ) + " |"
                )

    gate_status = str(gate["gate_status"])
    requirements = gate.get("requirements", {})
    integrity_complete = all(
        bool(requirements.get(name, {}).get("passed"))
        for name in ("native_backend_calls", "correctness")
    ) and bool(
        requirements.get("validation_evidence_source", {}).get(
            "validation_gate_source_passed"
        )
    )
    if not integrity_complete:
        raise RuntimeError("validation integrity requirements are incomplete")
    verdict = str(final_decision["verdict"])
    engineering_decision = str(final_decision["engineering_decision"])
    final_status = str(final_decision["final_status"])
    if final_status == "NOT_RUN_GATE_NOT_PASSED":
        final_statement = (
            "validation gate は通過しなかったため、規定どおり大規模 final holdout は実行していない。"
        )
    elif final_status == "AUTHORIZED_NOT_YET_RUN":
        final_statement = (
            "validation gate は通過したが、lock 済み fresh holdout 評価はまだ完了していない。"
        )
    else:
        final_statement = (
            f"lock 済み fresh holdout 評価は `{final_status}` として完了した。"
        )
    profile_run = profile.get("unprofiled_run", profile.get("matched_unprofiled_run", {}))
    profile_note = json.dumps(profile_run, ensure_ascii=False, sort_keys=True)
    binary = correctness.get("native_build", {})
    run_lines = [
        f"| `{run['run_id']}` | `{run['config_hash']}` | `{run['implementation_tree_sha256']}` |"
        for run in evidence.runs
    ]
    raw_inventory_lines = [
        "| " + " | ".join(
            (
                f"`{run['run_id']}`",
                f"`{run['run_dir']}/raw`",
                str(len(run["raw_files"])),
                str(sum(int(item["rows"]) for item in run["raw_files"])),
                str(sum(int(item["bytes"]) for item in run["raw_files"])),
                f"`{run['raw_inventory_sha256']}`",
                f"`{run['completion_sha256']}`",
            )
        ) + " |"
        for run in bound_runs
    ]
    gate_reasons = gate.get("reasons")
    if not isinstance(gate_reasons, list) or not all(
        isinstance(value, str) and value for value in gate_reasons
    ):
        raise RuntimeError("validation gate reasons are missing/malformed")
    gate_reason_text = ", ".join(f"`{value}`" for value in gate_reasons) or "なし"
    main_test_query_counts = {
        int(point["expected_test_queries"])
        for point in summary.get("operating_points", [])
        if isinstance(point, dict)
        and point.get("experiment_id") == "sift-initial"
        and point.get("expected_test_queries") is not None
    }
    if len(main_test_query_counts) != 1 or next(iter(main_test_query_counts)) <= 0:
        raise RuntimeError("main timing query denominator is missing/inconsistent")
    main_test_query_count = next(iter(main_test_query_counts))
    deselected = correctness.get("pytest_deselected")
    if deselected is None:
        deselected_text = "bound JSON/JUnit に未保存（数値は主張しない）"
    elif isinstance(deselected, bool) or not isinstance(deselected, int) or deselected < 0:
        raise RuntimeError("pytest deselected count is malformed")
    else:
        deselected_text = str(deselected)
    content = f"""# Native kernel 再検証報告（Issue #3）

研究全体の事前規定 stopping-rule verdict: **`{verdict}`**、工学的導入判定: **`{engineering_decision}`**（validation gate: **{gate_status}**、final status: **`{final_status}`**）。これは fresh-final holdout estimate ではない。{final_statement}

Validation gate の保存理由: {gate_reason_text}。

本報告は [GitHub Issue #3](https://github.com/wasanemon/Searchability-Obligations/issues/3) の手順に従い、旧 Python 実装の遅さと center-radius pruning 自体の限界を分離して再検証した結果である。ordinary L2 と Faiss の squared-L2 を区別し、全 native 保証経路は [`docs/native_numerics.md`](../docs/native_numerics.md) の interval/error-bound と厳密境界順序を用いた。

## 結論（RQ1 / RQ2 / RQ3）

- **RQ1（P 対 F、同一保証）**: {rq1_answer} P と F は同じ frozen `C`・同じ保証で、`F/P` paired API wall が直接比較である。
- **RQ2（P 対 A、practical comparator）**: {rq2_answer} A は optimized Faiss Delta Flat + small exact rerank だが、rerank は返却 shortlist 内だけで全 Delta に対する保証ではない。
- **RQ3（勝敗理由）**: {rq3_answer} skip/read、N/P、ablation、Base cache、build・memory・限定的 break-even を併記して解釈する。

## 正しさ gate

- fixed-seed native cases: {correctness.get('fixed_seed_cases')}。
- pytest JUnit: executed {correctness.get('pytest_testcases')} passed、failures {correctness.get('pytest_failures')}、errors {correctness.get('pytest_errors')}、skipped {correctness.get('pytest_skipped')}、deselected {deselected_text}。
- native backend: `{binary.get('backend')}`、shared object SHA-256 `{correctness.get('native_shared_object_sha256')}`。
- compile flags: `{binary.get('compile_flags')}`。fast-math 無効、FE_TONEAREST 強制、strict `LB > tau-beta`、曖昧境界だけ exact Fraction で再順位付けした。
- raw native 行の backend/call evidence と beta chain の aggregate failure はそれぞれ {summary['integrity_checks']['native_backend_calls']['failure_count']} / {summary['integrity_checks']['correctness']['failure_count']}。
- Python fallback は F/N/P raw 行 {fallback_counts['fnp_fallbacks']}/{fallback_counts['fnp_rows']}、そのうち P は {fallback_counts['p_fallbacks']}/{fallback_counts['p_rows']}。

## Validation 結果

latency は API wall。各 session/query 内で repetition median、session 間で比の幾何平均、query を独立単位として固定 seed の paired bootstrap（{summary['bootstrap_policy']['resamples']} resamples）を用いた。speedup は baseline/P なので 1 より大きいほど P が速い。A の品質不一致行も timing から除外していない。`n q/session-q` は paired query 数 / paired process-session-query observation 数である。

| condition | P method | beta (L2) | P p50/p95/p99 ms | F/P geomean [95% CI] | A/P geomean [95% CI] | N/P geomean | median skipped groups / vectors scanned | n q/session-q (F; A) |
|---|---|---:|---:|---:|---:|---:|---:|---|
{chr(10).join(table)}

主条件の beta=0 について、3 timing scope を同じ分母で並べる。この validation は process session {main_latency_coverage['process_sessions']} 回、各 query の session 数 {main_latency_coverage['sessions_per_query_min']}–{main_latency_coverage['sessions_per_query_max']}、paired query/session-query は {main_latency_coverage['paired_queries']}/{main_latency_coverage['paired_session_query_observations']}である。したがって CI は query sampling の不確実性は表すが、process/session 間変動は推定しない。

| scope | F p50/p95/p99 ms | A p50/p95/p99 ms | P(beta=0) p50/p95/p99 ms | F/P geomean [95% CI] | A/P geomean [95% CI] | n q/session-q |
|---|---:|---:|---:|---:|---:|---:|
{chr(10).join(main_latency_rows)}

## Delta-influence subset

Delta neighbor が exact visible top-k に影響する query だけを同じ query-paired bootstrap で再集計した。下表は real/non-degenerate かつ SIFT primary で subset n>0 の gate-eligible 候補を全件（n={delta_counts['sift']}）表示する。subset CI 下端 > 1 は F/P で {delta_counts['sift_f_ci_lower_above_one']}/{delta_counts['sift']}、A/P で {delta_counts['sift_a_ci_lower_above_one']}/{delta_counts['sift']}（A/P の CI 上端 < 1 は {delta_counts['sift_a_ci_upper_below_one']}/{delta_counts['sift']}）。

| condition | P method | beta (L2) | influenced queries n | F/P subset geomean [95% CI] | A/P subset geomean [95% CI] |
|---|---|---:|---:|---:|---:|
{chr(10).join(delta_sift_rows)}

GIST は再利用済みの非独立 anchor で gate 非対象のため、SIFT と分けて全 {delta_counts['gist']} operating points を記述する。

| GIST anchor | P method | beta (L2) | influenced queries n | F/P subset geomean [95% CI] | A/P subset geomean [95% CI] |
|---|---|---:|---:|---:|---:|
{chr(10).join(delta_gist_rows)}

A mismatch は `baseline_quality_mismatch_*_retained` と raw `baseline_validation_failure` に残した。

## 正の beta の記述的寄与

| P method | requested beta (L2) | session-query n | API p50/p95/p99 ms | beta=0 からの p50/p95/p99 変化 | median skipped groups | median vectors scanned (変化) |
|---|---:|---:|---:|---:|---:|---:|
{chr(10).join(positive_beta_rows)}

{positive_beta_interpretation}


## 主条件の timing component（p50 ms）

各値は各 session-query 内の repetition median を取った後の p50。`kernel/Δ-search` は native F/N/P では kernel total、A では Delta Flat search、`adaptive/merge` は native adaptive exact または A shortlist exact merge。`other micro` は micro からそれらの重複しない timer を引いた残差である。Base prepare と native query prepare は micro 外だが composed/API の層を明示するため併記する。

| role | method | beta | n session-query | Base prep | native prep | micro | kernel/Δ-search | LB | order | group scan | raw scan | adaptive/merge | receipt | other micro |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(main_component_rows)}

`faiss_A_reference` は Faiss の squared-L2 順序を直接使う**非認証**の参考経路であり、RQ2 の A や gate には採用していない。その行は raw と代表 raw export に別 role `A-reference` で保存した。

{final_section}

## 品質・保証指標

match は同一 `C` の O との ordered key 一致率、recall は全 visible exact top-k に対する median/min、Delta capture は exact top-k に Delta neighbor がある queryだけの median/min（括弧内は該当 query-session 数）である。`n` は最初の repetition を代表とした query-session 数。P の gap は `max observed / max certified / max requested beta`（ordinary L2）で、必ずこの順の非減少 chain を満たすことを integrity gate が確認した。

| condition | method | n query-session | same-C match rate | full exact recall median/min | Delta capture median/min (n) | P gap obs/cert/req max |
|---|---|---:|---:|---:|---:|---:|
{chr(10).join(quality_rows)}

## Geometry audit

全 {geometry_coverage['audit_files']} audit files は、condition ごとの固定 validation/calibration subset（件数は表示）を使い、合計 condition-query sets {geometry_coverage['audited_query_sets']} / {geometry_coverage['decisions']} group decisions（scan {geometry_coverage['scan_decisions']}、skip {geometry_coverage['skip_decisions']}）を timing 外で照合した。主 `sift-initial` は audit {geometry_coverage['main_queries']} queries / {geometry_coverage['main_decisions']} decisions であり、timing の別 test partition {main_test_query_count} queries 全件を audit したものではない。audit scope は `{', '.join(geometry_coverage['scopes'])}`。保存した厳密 group 最小距離と native `LB` を照合し、`exact_min_lower - LB` を下界の緩さとして scan/skip 別に記述する。全 audit は completion の ancillary SHA-256 inventory と build の dataset/split ID に一致し、`passed=true` のものだけを集計した。

| condition | audited queries | action | decisions | radius median | true group-min L2 median | (true min-LB) median/p95 |
|---|---:|---|---:|---:|---:|---:|
{chr(10).join(geometry_rows)}

## Build / memory と限定的 break-even

主条件の cold/warm construction、cache audit、保持 memory は以下。RSS 差はプロセス全体の high-water/allocator 効果を含み、P 固有 memory とはみなさない。全 query timing は保存済みの warm immutable packed view で、各 condition の timing 前 warmup query 後に実行した。

| object | phase/metric | value | denominator / scope |
|---|---|---:|---|
{chr(10).join(main_build_rows)}

`Q_break_even = B_extra / (t_F - t_P)` とし、分母は query-paired API wall 差の中央値で近似した。主列の `B_group` は center training + assignment + group packing/radius だけで、native packed view を F/P 共通の warm prerequisite と扱う。`感度 Q(group+packed)` は packed cold build も全て P に課した場合であり、両者の間に実システムの分担があり得る。更新頻度、永続化 I/O、共通 layout の利用範囲を含まない感度分析であり、`t_F - t_P <= 0` なら有限解なしとする。

| condition | P method | B_group ms | packed cold ms | median (t_F-t_P) ms/query | Q(group-only) | 感度 Q(group+packed) | n paired session-query | group bytes | group+packed bytes |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(break_even_rows)}

## 段階 ablation と原因分解

以下は保存された実測値である。paired 行の `before/after ratio` は query-paired geometric mean であり、p50 列同士の単純比とは限らない。1 より大きいほど after が速い。旧 P 比較だけは query 集合が異なるため、表示 p50 の比を記述するだけで paired 推論には使わない。

| stage | metric | before | after | before/after ratio | scope / quality |
|---|---|---:|---:|---:|---|
{chr(10).join(ablation_rows)}

1. **旧 layout → packed native**: 旧 Python P-only development profile は保存済みであり、cProfile 自体の膨張値を性能推定には使わない。対応する unprofiled evidence の抜粋は `{profile_note}`。
2. **rescan → incremental heap**: `ablation_P_beta0_rescan_adaptive` と主 `native_P_beta0` は同じ P・beta=0・adaptive ranking を使い、各group decisionでの kth threshold 更新方式だけを変えた。F の rescan/heap は最終順位計算寄与の補助値に限る。
3. **all-boundary exact → adaptive exact**: `ablation_F_heap_all_exact` と主 F は heap を共通にし、境界 exact 範囲だけを変えた。
4. **no pruning → pruning**: N/P の paired比で group pruning の純寄与を分離した。
5. **Base visible-map rebuild → immutable cache**: validation queryだけの非計時 auditで両者の candidate hash/key が完全一致することを確認し、wall time と hit/miss/rebuild counterを各 runの `audits/*.base-cache.json` に保存した。

## 測定範囲と限界

- CPU 1 thread、immutable snapshot、Base HNSW `M=32, efConstruction=200, efSearch=128`、既定 `k=10, C=64, groups=128`。center は Base のみで学習し、beta は measurement query ではなく validation query の kth L2 medianから固定した。
- SIFT は Base 100k / Delta 10kを主条件、Delta 0/1k/100k、groups 64/128/512、k 1/10/100を検証した。GIST は memory 制約どおり Base 50k / Delta 5k / d=960。4 synthetic familyも実行した。
- validation は SIFT query 0..399 の再利用領域だけを読み、事前登録 fresh final holdout 1200..2199 は gate 通過前に読み込んでいない。GIST 全 query は過去使用済みなので独立 holdoutとは呼ばない。
- API wall は Base candidate preparationからReceipt完成までを全 method の独立randomized passで測り、その後に別順序のmicro passを実行した。composed E2E は共通 Base preparation + method固有 preparation + micro、micro は frozen C/native query preparation後の核である。build、oracle、audit、profileは query timing外。
- Formal validation は 1 process session のみで、process/session 間分散は validation CI に含まれない。Issue #3 で許された optional post-validation tuning は 0/2 rounds で、measurement partition を使った center/group/beta の再調整は行っていない。
- shared hostでexclusive CPU reservationはない。静的SIFT/GISTは更新時系列、text embedding、production DBMSを代表しない。

## Evidence identity

| run | config SHA-256 | implementation-tree SHA-256 |
|---|---|---|
{chr(10).join(run_lines)}

current implementation-tree / native shared-object SHA-256 はそれぞれ `{current_implementation_hash}` / `{current_native_hash}` で、correctness・validation、および final が lock 済みなら lock と一致することを report 生成時にも再検証した。

大容量 raw は report 入力 root `{args.input}` の各 run の `raw/` にあり、git 管理対象外である。下表の inventory digest は `(relative path, SHA-256, rows, bytes)` の canonical JSON に対する SHA-256、completion は ancillary/build/audit を束縛する marker の SHA-256 である。各 shard の相対 path・SHA-256・row 数の完全 inventory は hash-bound analysis JSON の `runs[].raw_files` にある。

| run | raw location (input-root relative) | shards | rows | bytes | raw inventory SHA-256 | completion SHA-256 |
|---|---|---:|---:|---:|---|---|
{chr(10).join(raw_inventory_lines)}

保存 raw がある環境では `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 make native-report NATIVE_VALIDATION_INPUT="{args.input}"` で checksum 検証から report を再生成できる。実験自体の再実行は同じ 1-thread 環境で `make native-validate`。raw JSONL は非圧縮であり、別の圧縮 artifact は主張しない。

追跡対象の summary/gate、deterministic representative sample、correctness、final decision、policy/holdout registration を合わせれば、研究 verdict、run/completion identity、全 raw shard checksum inventory、代表 row は第三者検査できる。summary/gate/representative の三点だけで確認できるのは validation outcome までである。ただし gitignore 済み raw/run directory がなければ全 query 行の再集計、ancillary audit の再読込、latency 分布の独立再計算はできず、summary から raw evidence を復元することもできない。

- analysis SHA-256: `{file_sha256(args.summary)}`
- gate SHA-256: `{file_sha256(args.gate)}`
- correctness SHA-256: `{file_sha256(args.correctness)}`
- old profile summary SHA-256: `{file_sha256(args.profile)}`
- deterministic representative raw SHA-256: `{file_sha256(args.representative_raw)}`
- final decision SHA-256: `{file_sha256(args.final_decision)}`
- final summary/gate SHA-256: `{final_decision.get('final_summary_sha256', 'not run')}` / `{final_decision.get('final_gate_sha256', 'not run')}`
- incomplete run は analysis から除外され、一覧は summary の `excluded_incomplete_runs` に残る。

この結果は指定条件の検索方式比較であり、production DBMS 全体の優位性、ACID commit throughput、一般の embedding 分布への外挿を主張しない。
"""
    atomic_write_bytes(args.output, content.encode("utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

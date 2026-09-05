"""Fail-closed fresh-final authorization and analysis for Issue #3.

The validation gate is the only component allowed to choose an operating
point.  This module turns that choice into an immutable recipe without reading
the pre-registered SIFT holdout, and later evaluates three process sessions at
the query level.  GIST and the fixed anchors are descriptive only.
"""

from __future__ import annotations

from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import statistics
from typing import Any, Mapping, Sequence

import numpy as np

from .artifacts import file_sha256, implementation_tree_sha256, object_sha256
from .final_policy import (
    FinalPolicyError,
    validate_final_policy,
    validate_issue_fixed_decision_values,
)
from .native_analysis import (
    DEFAULT_BOOTSTRAP_SEED,
    MIN_BOOTSTRAP_RESAMPLES,
    analyze_native_rows,
)


FINAL_SCHEMA_VERSION = 1
STUDY_ID = "issue-3-native-recheck"
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


class NativeFinalError(ValueError):
    """Raised when final authorization or evidence identity is not exact."""


def read_object(path: str | Path) -> dict[str, Any]:
    location = Path(path)
    try:
        value = json.loads(location.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise NativeFinalError(f"cannot read JSON object: {location}") from error
    if not isinstance(value, dict):
        raise NativeFinalError(f"expected JSON object: {location}")
    return value


def _timestamp(value: str | None) -> str:
    return value or datetime.now(timezone.utc).isoformat()


def _id_set_sha256(start: int, stop: int) -> str:
    payload = "".join(f"{identifier}\n" for identifier in range(start, stop)).encode()
    return hashlib.sha256(payload).hexdigest()


def _float_name(value: float) -> str:
    return format(value, ".8g").replace("-", "m").replace("+", "p").replace(".", "p")


def method_name_for_factor(factor: float) -> str:
    return (
        "native_P_beta0"
        if factor == 0.0
        else f"native_P_factor_{_float_name(factor)}"
    )


def _materialized_experiments(
    validation_config: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    defaults = validation_config.get("defaults")
    experiments = validation_config.get("experiments")
    if not isinstance(defaults, dict) or not isinstance(experiments, list):
        raise NativeFinalError("validation config has malformed defaults/experiments")
    result: dict[str, dict[str, Any]] = {}
    for value in experiments:
        if not isinstance(value, dict) or not str(value.get("id", "")):
            raise NativeFinalError("validation config has malformed experiment")
        row = json.loads(json.dumps({**defaults, **value}, allow_nan=False))
        identifier = str(row["id"])
        if identifier in result:
            raise NativeFinalError("validation experiment IDs are not unique")
        if "beta_factors" not in row:
            row["beta_factors"] = [
                0.0,
                *[float(item) for item in row.get("positive_beta_factors", [])],
            ]
        result[identifier] = row
    return result


def _selected_factor(
    selected: Mapping[str, Any], specification: Mapping[str, Any]
) -> float:
    proposed = str(selected.get("proposed_method", ""))
    candidates = [
        float(value) for value in specification.get("beta_factors", [0.0])
    ]
    matching = [factor for factor in candidates if method_name_for_factor(factor) == proposed]
    if len(matching) != 1:
        raise NativeFinalError(
            "selected P method does not identify exactly one configured beta factor"
        )
    requested = selected.get("requested_beta_l2")
    if requested is None or not math.isfinite(float(requested)) or float(requested) < 0:
        raise NativeFinalError("selected candidate has no finite non-negative beta")
    if matching[0] == 0.0 and float(requested) != 0.0:
        raise NativeFinalError("beta-zero method has nonzero requested beta")
    return matching[0]


def _validate_source_identity(
    *,
    gate: Mapping[str, Any],
    summary: Mapping[str, Any],
    validation_config: Mapping[str, Any],
    correctness: Mapping[str, Any],
    holdout: Mapping[str, Any],
    policy: Mapping[str, Any],
    validation_config_path: Path,
    holdout_path: Path,
    policy_path: Path,
) -> str:
    if gate.get("study_id") != STUDY_ID or summary.get("study_id") != STUDY_ID:
        raise NativeFinalError("validation evidence has the wrong study_id")
    if summary.get("validation_gate") != gate:
        raise NativeFinalError("gate file differs from the gate embedded in summary")
    try:
        validate_final_policy(policy)
    except FinalPolicyError as error:
        raise NativeFinalError(str(error)) from error
    source = summary.get("source_integrity")
    if not isinstance(source, dict) or source.get("validation_gate_source_passed") is not True:
        raise NativeFinalError("validation source-integrity gate did not pass")
    if source.get("non_validation_runs") not in ([], ()):
        raise NativeFinalError("validation summary contains a non-validation run")
    integrity_checks = summary.get("integrity_checks")
    if not isinstance(integrity_checks, Mapping):
        raise NativeFinalError("validation integrity checks are missing or malformed")
    validation_correctness = integrity_checks.get("correctness")
    if not isinstance(validation_correctness, Mapping):
        raise NativeFinalError(
            "validation correctness integrity record is missing or malformed"
        )
    rows_checked = validation_correctness.get("rows_checked")
    failure_count = validation_correctness.get("failure_count")
    failure_examples = validation_correctness.get("failure_examples")
    if (
        isinstance(rows_checked, bool)
        or not isinstance(rows_checked, int)
        or rows_checked <= 0
        or isinstance(failure_count, bool)
        or not isinstance(failure_count, int)
        or not isinstance(failure_examples, list)
    ):
        raise NativeFinalError(
            "validation correctness integrity record is missing or malformed"
        )
    if (
        validation_correctness.get("passed") is not True
        or failure_count != 0
        or failure_examples
    ):
        raise NativeFinalError("validation correctness gate did not pass")
    implementation_hashes = source.get("implementation_tree_sha256")
    if not isinstance(implementation_hashes, list) or len(implementation_hashes) != 1:
        raise NativeFinalError("validation has no single implementation identity")
    implementation_hash = str(implementation_hashes[0])
    source_hashes = source.get("source_config_object_hash")
    expected_source_hash = object_sha256(validation_config)
    if source_hashes != [expected_source_hash]:
        raise NativeFinalError("validation source config object identity mismatch")
    if source.get("holdout_manifest_sha256") != [file_sha256(holdout_path)]:
        raise NativeFinalError("validation summary binds a different holdout manifest")
    selection_integrity = holdout.get("selection_integrity")
    if not isinstance(selection_integrity, dict):
        raise NativeFinalError("holdout manifest has no selection integrity")
    if selection_integrity.get("validation_source_config_object_sha256") != expected_source_hash:
        raise NativeFinalError("holdout manifest binds a different validation config")
    if selection_integrity.get("validation_source_config_file_sha256") != file_sha256(
        validation_config_path
    ):
        raise NativeFinalError("holdout manifest validation-config file hash mismatch")
    registered_policy = holdout.get("pre_registered_final_policy")
    configured_policy = validation_config.get("holdout_policy", {}).get(
        "pre_registered_final_policy"
    )
    if not isinstance(registered_policy, dict) or not isinstance(
        configured_policy, dict
    ):
        raise NativeFinalError("final policy was not registered before validation")
    policy_identity = {
        "path": registered_policy.get("path"),
        "file_sha256": registered_policy.get("file_sha256"),
        "object_sha256": registered_policy.get("object_sha256"),
    }
    registered_path = Path(str(policy_identity["path"]))
    if not registered_path.is_absolute():
        registered_path = REPOSITORY_ROOT / registered_path
    try:
        validate_issue_fixed_decision_values(
            registered_policy.get("issue_fixed_decision_values")
        )
    except FinalPolicyError as error:
        raise NativeFinalError(str(error)) from error
    if (
        configured_policy != policy_identity
        or registered_path.resolve() != policy_path.resolve()
        or policy_identity["file_sha256"] != file_sha256(policy_path)
        or policy_identity["object_sha256"] != object_sha256(policy)
        or selection_integrity.get("final_policy_path") != policy_identity["path"]
        or selection_integrity.get("final_policy_file_sha256")
        != policy_identity["file_sha256"]
        or selection_integrity.get("final_policy_object_sha256")
        != policy_identity["object_sha256"]
    ):
        raise NativeFinalError("final policy differs from pre-validation registration")
    sift = holdout.get("sift")
    gist = holdout.get("gist")
    if not isinstance(sift, dict) or not isinstance(gist, dict):
        raise NativeFinalError("holdout manifest is missing SIFT/GIST declarations")
    registered = sift.get("pre_registered_fresh_final_holdout")
    if not isinstance(registered, dict):
        raise NativeFinalError("SIFT fresh holdout was not pre-registered")
    interval = policy.get("primary", {}).get("query_interval")
    if interval != [1200, 2200] or registered.get("interval") != interval:
        raise NativeFinalError("SIFT final interval differs from pre-registration")
    if registered.get("set_sha256") != _id_set_sha256(1200, 2200):
        raise NativeFinalError("SIFT final query-ID digest mismatch")
    if (
        registered.get("disjoint_from_historically_used_union") is not True
        or registered.get("loaded_by_native_validation") is not False
        or registered.get("selected_before_native_validation_timing") is not True
    ):
        raise NativeFinalError("SIFT holdout freshness declaration is incomplete")
    if (
        gist.get("independence_label") != "reused_non_independent"
        or gist.get("fresh_holdout_claim_permitted") is not False
    ):
        raise NativeFinalError("GIST reuse declaration is incomplete")
    if correctness.get("status") != "passed":
        raise NativeFinalError("native correctness prerequisite is not passed")
    if int(correctness.get("fixed_seed_cases", 0)) < 10_000:
        raise NativeFinalError("native correctness prerequisite has fewer than 10000 cases")
    if correctness.get("implementation_tree_sha256") != implementation_hash:
        raise NativeFinalError("correctness/validation implementation identities differ")
    backend = integrity_checks.get("native_backend_calls")
    if not isinstance(backend, Mapping) or backend.get("passed") is not True:
        raise NativeFinalError("validation native backend gate did not pass")
    binary_hashes = backend.get("native_binary_sha256")
    if binary_hashes != [correctness.get("native_shared_object_sha256")]:
        raise NativeFinalError("correctness/validation native binary identities differ")
    current_hash = implementation_tree_sha256(REPOSITORY_ROOT)
    if current_hash != implementation_hash:
        raise NativeFinalError(
            "implementation changed after validation; fresh final authorization refused"
        )
    return implementation_hash


def _final_experiment(
    source: Mapping[str, Any],
    *,
    roles: Sequence[str],
    beta_factors: Sequence[float],
    gate_eligible: bool,
    interval: Sequence[int],
    independence_label: str,
) -> dict[str, Any]:
    row = json.loads(json.dumps(source, allow_nan=False))
    dataset = dict(row["dataset"])
    start, stop = map(int, interval)
    dataset["test_query_offset"] = start
    dataset["n_test"] = stop - start
    if str(dataset.get("name")) == "gist":
        # All 1,000 GIST rows were used before registration.  With beta=0 no
        # calibration rows are needed, so the entire reused interval can be
        # timed without overlapping a validation partition.
        dataset["validation_query_offset"] = 0
        dataset["n_validation"] = 0
    else:
        dataset["validation_query_offset"] = 0
        dataset["n_validation"] = 200
    row["dataset"] = dataset
    row["beta_factors"] = sorted(set(float(value) for value in beta_factors))
    row["positive_beta_factors"] = [value for value in row["beta_factors"] if value > 0]
    row["kernel_ablations"] = False
    row["final_roles"] = sorted(set(str(value) for value in roles))
    row["final_gate_eligible"] = bool(gate_eligible)
    row["final_independence_label"] = str(independence_label)
    row["final_query_interval"] = [start, stop]
    return row


def prepare_final_authorization(
    *,
    gate: Mapping[str, Any],
    summary: Mapping[str, Any],
    validation_config: Mapping[str, Any],
    correctness: Mapping[str, Any],
    holdout: Mapping[str, Any],
    policy: Mapping[str, Any],
    gate_path: Path,
    summary_path: Path,
    validation_config_path: Path,
    correctness_path: Path,
    holdout_path: Path,
    policy_path: Path,
    decided_at_utc: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any] | None, dict[str, Any] | None]:
    """Return decision, immutable lock recipe, and runner config.

    A completed negative validation is a successful terminal result and never
    constructs a final config.  Every malformed or inconsistent identity
    raises instead of being converted into a negative performance result.
    """

    decided = _timestamp(decided_at_utc)
    common = {
        "schema_version": FINAL_SCHEMA_VERSION,
        "study_id": STUDY_ID,
        "decided_at_utc": decided,
        "gate_sha256": file_sha256(gate_path),
        "validation_summary_sha256": file_sha256(summary_path),
        "validation_config_path": str(validation_config_path),
        "validation_config_file_sha256": file_sha256(validation_config_path),
        "validation_config_object_sha256": object_sha256(validation_config),
        "correctness_path": str(correctness_path),
        "correctness_sha256": file_sha256(correctness_path),
        "holdout_manifest_path": str(holdout_path),
        "holdout_manifest_sha256": file_sha256(holdout_path),
        "final_policy_path": str(policy_path),
        "final_policy_sha256": file_sha256(policy_path),
        "final_policy_object_sha256": object_sha256(policy),
    }
    status = gate.get("gate_status")
    if status not in {"PASSED", "NOT_PASSED"}:
        raise NativeFinalError("unknown validation gate status")
    if summary.get("validation_gate") != gate:
        raise NativeFinalError("gate file differs from the gate embedded in summary")
    implementation_hash = _validate_source_identity(
        gate=gate,
        summary=summary,
        validation_config=validation_config,
        correctness=correctness,
        holdout=holdout,
        policy=policy,
        validation_config_path=validation_config_path,
        holdout_path=holdout_path,
        policy_path=policy_path,
    )
    registered_policy = holdout["pre_registered_final_policy"]
    assert isinstance(registered_policy, dict)
    policy_identity = {
        "path": registered_policy["path"],
        "file_sha256": registered_policy["file_sha256"],
        "object_sha256": registered_policy["object_sha256"],
    }
    if status == "NOT_PASSED":
        return (
            {
                **common,
                "validation_gate_status": status,
                "final_status": "NOT_RUN_GATE_NOT_PASSED",
                "performance_gate_status": "NOT_RUN_GATE_NOT_PASSED",
                "performance_verdict": "NOT_SUPPORTED_IN_TESTED_REGIME",
                "verdict": "NOT_SUPPORTED_IN_TESTED_REGIME",
                "engineering_decision": "NO_GO",
                "selected_candidate": None,
                "lock_created": False,
                "large_final_started": False,
                "fresh_sift_holdout_loaded": False,
                "reason": "completed validation did not authorize the large final",
            },
            None,
            None,
        )

    selected = gate.get("selected_candidate")
    if not isinstance(selected, dict) or selected.get("passed") is not True:
        raise NativeFinalError("PASSED validation has no passed selected candidate")
    experiments = _materialized_experiments(validation_config)
    selected_id = str(selected.get("experiment_id", ""))
    if selected_id not in experiments:
        raise NativeFinalError("selected validation experiment is not configured")
    selected_source = experiments[selected_id]
    selected_dataset = selected_source.get("dataset")
    if not isinstance(selected_dataset, dict) or (
        selected_dataset.get("type") != "texmex"
        or selected_dataset.get("name") != "sift"
        or int(selected_dataset.get("n_delta", -1)) < 1_000
    ):
        raise NativeFinalError("selected primary is not an eligible SIFT condition")
    factor = _selected_factor(selected, selected_source)
    proposed_method = method_name_for_factor(factor)

    anchors = policy.get("fixed_anchors")
    if not isinstance(anchors, list) or len(anchors) != 3:
        raise NativeFinalError("final policy must contain exactly three fixed anchors")
    selected_roles = ["locked_primary"]
    anchor_by_id: dict[str, list[Mapping[str, Any]]] = {}
    for anchor in anchors:
        if not isinstance(anchor, dict):
            raise NativeFinalError("malformed final anchor")
        anchor_id = str(anchor.get("validation_experiment_id", ""))
        if anchor_id not in experiments:
            raise NativeFinalError(f"unknown final anchor: {anchor_id}")
        anchor_by_id.setdefault(anchor_id, []).append(anchor)
        if anchor_id == selected_id:
            selected_roles.append(str(anchor["role"]))

    final_experiments: list[dict[str, Any]] = []
    selected_factors = [factor]
    if selected_id in anchor_by_id:
        # A fixed anchor is beta=0.  If a positive-beta primary selects the
        # same geometry, run both operating points without rebuilding it.
        selected_factors.append(0.0)
    primary = _final_experiment(
        selected_source,
        roles=selected_roles,
        beta_factors=selected_factors,
        gate_eligible=True,
        interval=policy["primary"]["query_interval"],
        independence_label=policy["primary"]["independence_label"],
    )
    primary["locked_primary_proposed_method"] = proposed_method
    primary["locked_primary_requested_beta_l2"] = float(
        selected["requested_beta_l2"]
    )
    final_experiments.append(primary)
    for anchor in anchors:
        anchor_id = str(anchor["validation_experiment_id"])
        if anchor_id == selected_id:
            continue
        final_experiments.append(
            _final_experiment(
                experiments[anchor_id],
                roles=[str(anchor["role"])],
                beta_factors=[0.0],
                gate_eligible=False,
                interval=anchor["query_interval"],
                independence_label=str(anchor["independence_label"]),
            )
        )

    seeds = [int(value) for value in policy.get("process_session_seeds", [])]
    if seeds != [0, 1, 2]:
        raise NativeFinalError("final process/group seeds must be exactly 0,1,2")
    final_gate = policy.get("final_gate")
    if not isinstance(final_gate, dict):
        raise NativeFinalError("final policy has no final_gate")
    hnsw_policy = policy.get("hnsw_references")
    if (
        not isinstance(hnsw_policy, dict)
        or hnsw_policy.get("run_only_after_fresh_final_performance_pass") is not True
        or "run_only_after_validation_pass" in hnsw_policy
    ):
        raise NativeFinalError(
            "HNSW references must be locked to run only after the fresh performance pass"
        )
    locked_hnsw_policy = {
        **dict(hnsw_policy),
        "output_root": str(policy["hnsw_output_root"]),
        "authorization_artifact": str(policy["pre_hnsw_authorization"]),
    }
    lock = {
        **common,
        "validation_gate_status": "PASSED",
        "status": "LOCKED_FOR_FRESH_FINAL",
        "implementation_tree_sha256": implementation_hash,
        "native_shared_object_sha256": correctness["native_shared_object_sha256"],
        "selected_candidate": selected,
        "selected_operating_point": {
            "validation_experiment_id": selected_id,
            "proposed_method": proposed_method,
            "beta_factor": factor,
            "requested_beta_l2": float(selected["requested_beta_l2"]),
            "materialized_validation_experiment_sha256": object_sha256(
                selected_source
            ),
        },
        "selection_policy": "mechanical validation gate; no post-gate retuning",
        "final_experiment_roles": {
            str(row["id"]): list(row["final_roles"]) for row in final_experiments
        },
        "primary_sift_query_interval": [1200, 2200],
        "primary_sift_query_ids_sha256": _id_set_sha256(1200, 2200),
        "gist_query_interval": [0, 1000],
        "gist_independence_label": "reused_non_independent",
        "gist_used_in_final_gate": False,
        "process_session_seeds": seeds,
        "session_seed_role": "process session, method-order seed, and group-build seed",
        "required_final_gate": dict(final_gate),
        "hnsw_references": locked_hnsw_policy,
        "pre_registered_final_policy": policy_identity,
        "locked_execution": dict(policy["locked_execution"]),
    }
    final_config = {
        "schema_version": 2,
        "study_id": STUDY_ID,
        "run_name": "native-recheck-fresh-final",
        "phase": "final",
        "evidence_role": "final",
        "output_root": str(policy["output_root"]),
        "threads": 1,
        "process_sessions": 3,
        "process_session_seeds": seeds,
        "warmup_queries": int(policy["warmup_queries"]),
        "repetitions": int(policy["repetitions_per_session"]),
        "query_block_size": int(policy["query_block_size"]),
        "method_order_seed": 0,
        "bootstrap_resamples": int(policy["bootstrap_resamples"]),
        "bootstrap_seed": int(policy["bootstrap_seed"]),
        # The source-frozen 10,000-case/Fraction gate and validation oracle are
        # already lock prerequisites. Fresh final timing retains old O and
        # full-visible exact truth, but does not add a post-selection oracle.
        "independent_oracle_queries": 0,
        "independent_oracle_max_population": 120000,
        "lb_audit_queries": 0,
        "base_cache_ablation_queries": 0,
        "required_correctness_evidence": str(policy["correctness_evidence"]),
        "selection_policy": "locked validation operating point plus fixed anchors",
        "final_primary_experiment_id": selected_id,
        "final_primary_method": proposed_method,
        "final_primary_beta_factor": factor,
        "final_primary_requested_beta_l2": float(selected["requested_beta_l2"]),
        "holdout_policy": {
            "manifest": str(policy["holdout_manifest"]),
            "pre_registered_final_policy": policy_identity,
            "sift_primary_interval": [1200, 2200],
            "sift_primary_independence": "fresh_pre_registered_holdout",
            "gist_interval": [0, 1000],
            "gist_independence": "reused_non_independent",
            "gist_gate_eligible": False,
        },
        "final_gate": dict(final_gate),
        "hnsw_references": locked_hnsw_policy,
        "locked_execution": dict(policy["locked_execution"]),
        "experiments": final_experiments,
    }
    decision = {
        **common,
        "validation_gate_status": "PASSED",
        "final_status": "AUTHORIZED_NOT_YET_RUN",
        "performance_gate_status": "NOT_RUN",
        "performance_verdict": "INCONCLUSIVE",
        "verdict": "INCONCLUSIVE",
        "engineering_decision": "NO_GO",
        "selected_candidate": selected,
        "lock_created": True,
        "large_final_started": False,
        "fresh_sift_holdout_loaded": False,
    }
    return decision, lock, final_config


def session_effective_config(
    final_config: Mapping[str, Any],
    *,
    session_seed: int,
    final_config_sha256: str,
    final_lock_sha256: str,
) -> dict[str, Any]:
    """Apply the registered seed as process, order, and group-build seed."""

    seeds = [int(value) for value in final_config.get("process_session_seeds", [])]
    if session_seed not in seeds or seeds != [0, 1, 2]:
        raise NativeFinalError("unregistered final session seed")
    effective = json.loads(json.dumps(final_config, allow_nan=False))
    effective["session_id"] = f"session-seed-{session_seed}"
    effective["session_seed"] = session_seed
    effective["method_order_seed"] = session_seed
    effective["source_config_object_hash"] = object_sha256(final_config)
    effective["locked_final_config_file_sha256"] = final_config_sha256
    effective["final_lock_sha256"] = final_lock_sha256
    for experiment in effective["experiments"]:
        experiment["group_seed"] = session_seed
    return effective


def summarize_build_costs(
    manifest: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    """Extract compact, numeric construction/memory evidence from a run build."""

    def measured_int(value: Any, label: str) -> int:
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise NativeFinalError(
                f"native final build manifest has invalid non-negative {label}"
            )
        return value

    result: dict[str, dict[str, Any]] = {}
    for experiment_id, value in manifest.items():
        if not isinstance(value, dict):
            raise NativeFinalError("malformed native final build manifest")
        base = value.get("base")
        groups = value.get("groups")
        packed = value.get("native_packed_view")
        memory = value.get("memory")
        if not all(isinstance(item, dict) for item in (base, groups, packed, memory)):
            raise NativeFinalError("native final build manifest lacks component costs")
        group_build_ns = sum(
            measured_int(groups.get(field), f"groups.{field}")
            for field in ("center_training_ns", "assignment_ns", "packing_and_radius_ns")
        )
        base_vector_bytes = measured_int(
            memory.get("numpy_base_bytes"), "memory.numpy_base_bytes"
        )
        group_member_bytes = measured_int(
            groups.get("member_bytes"), "groups.member_bytes"
        )
        group_metadata_bytes = measured_int(
            groups.get("metadata_bytes"), "groups.metadata_bytes"
        )
        packed_python_bytes = measured_int(
            packed.get("python_owned_bytes"), "native_packed_view.python_owned_bytes"
        )
        packed_native_bytes = measured_int(
            packed.get("native_owned_bytes"), "native_packed_view.native_owned_bytes"
        )
        result[str(experiment_id)] = {
            "base_build_wall_ns": measured_int(
                base.get("build_wall_ns"), "base.build_wall_ns"
            ),
            # The runner does not serialize the Faiss index merely to size it;
            # this is the hash-bound input-vector footprint recorded at build.
            "base_input_vector_bytes": base_vector_bytes,
            "group_build_wall_ns": group_build_ns,
            "group_member_bytes": group_member_bytes,
            "group_metadata_bytes": group_metadata_bytes,
            "group_total_bytes": group_member_bytes + group_metadata_bytes,
            "native_packed_build_ns": measured_int(
                packed.get("build_ns"), "native_packed_view.build_ns"
            ),
            "native_packed_python_owned_bytes": packed_python_bytes,
            "native_packed_native_owned_bytes": packed_native_bytes,
            "native_packed_bytes": packed_python_bytes + packed_native_bytes,
            "rss_change_bytes": memory.get("rss_change_bytes"),
            "peak_rss_bytes_after": memory.get("peak_rss_bytes_after"),
        }
    return result


def _method_query_latencies(
    rows: Sequence[Mapping[str, Any]], *, experiment: str, method: str
) -> dict[tuple[str, int], float]:
    per_session: dict[tuple[str, str, int], list[float]] = {}
    for row in rows:
        if row.get("experiment_id") != experiment or row.get("method") != method:
            continue
        value = float(row["api_wall_latency_ns"])
        if not math.isfinite(value) or value <= 0:
            raise NativeFinalError("final API latency is not finite and positive")
        key = (
            str(row.get("session_id")),
            str(row.get("query_id")),
            int(row.get("query_position", -1)),
        )
        per_session.setdefault(key, []).append(value)
    medians = {
        key: float(statistics.median(values)) for key, values in per_session.items()
    }
    by_query: dict[tuple[str, int], list[float]] = {}
    for (_, query_id, position), value in medians.items():
        by_query.setdefault((query_id, position), []).append(value)
    result: dict[tuple[str, int], float] = {}
    for query, values in by_query.items():
        if len(values) != 3:
            raise NativeFinalError("each final query must have exactly three sessions")
        result[query] = math.exp(math.fsum(math.log(value) for value in values) / 3)
    return result


def _primary_session_metrics(
    rows: Sequence[Mapping[str, Any]],
    *,
    experiment: str,
    proposed: str,
    expected_queries: int,
) -> list[dict[str, Any]]:
    """Preserve seed-to-seed primary latency variation before session folding."""

    primary_rows = [row for row in rows if row.get("experiment_id") == experiment]
    methods_by_role = {
        role: {
            str(row.get("method"))
            for row in primary_rows
            if row.get("method_role") == role
        }
        for role in ("F", "A")
    }
    if any(len(methods) != 1 for methods in methods_by_role.values()):
        raise NativeFinalError("fresh primary does not have unique F/A methods")
    methods = {
        "F": next(iter(methods_by_role["F"])),
        "A": next(iter(methods_by_role["A"])),
        "P": proposed,
    }
    output: list[dict[str, Any]] = []
    for seed in (0, 1, 2):
        session_id = f"session-seed-{seed}:process-0000"
        measured: dict[str, dict[tuple[str, int], float]] = {}
        for role, method in methods.items():
            grouped: dict[tuple[str, int], list[float]] = {}
            for row in primary_rows:
                if row.get("session_id") != session_id or row.get("method") != method:
                    continue
                value = float(row["api_wall_latency_ns"])
                if not math.isfinite(value) or value <= 0:
                    raise NativeFinalError("final per-session API latency is not positive")
                key = (str(row["query_id"]), int(row.get("query_position", -1)))
                grouped.setdefault(key, []).append(value)
            measured[role] = {
                key: float(statistics.median(values)) for key, values in grouped.items()
            }
            if len(measured[role]) != expected_queries:
                raise NativeFinalError(
                    f"fresh primary session {seed}/{role} query coverage is incomplete"
                )
        if not (set(measured["F"]) == set(measured["A"]) == set(measured["P"])):
            raise NativeFinalError(f"fresh primary session {seed} F/A/P rows are unpaired")
        p_values = measured["P"]
        record: dict[str, Any] = {
            "session_seed": seed,
            "session_id": session_id,
            "query_count": expected_queries,
            "methods": methods,
        }
        for role in ("F", "A", "P"):
            values = list(measured[role].values())
            record[role] = {
                "api_wall_p50_ns": float(np.quantile(values, 0.50)),
                "api_wall_p95_ns": float(np.quantile(values, 0.95)),
            }
        for role in ("F", "A"):
            record[f"{role}_over_P_geomean"] = math.exp(
                math.fsum(
                    math.log(measured[role][key] / p_values[key]) for key in p_values
                )
                / len(p_values)
            )
        output.append(record)
    return output


def evaluate_final_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    final_config: Mapping[str, Any],
    lock: Mapping[str, Any],
    session_records: Sequence[Mapping[str, Any]],
    hnsw_reference: Mapping[str, Any],
    require_post_gate_reference: bool = True,
    bootstrap_resamples: int | None = None,
    bootstrap_seed: int | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Evaluate the fresh primary and return (summary, final gate)."""

    required_seeds = [0, 1, 2]
    observed_seeds = sorted(int(record["session_seed"]) for record in session_records)
    if observed_seeds != required_seeds or len(session_records) != 3:
        raise NativeFinalError("final evidence needs exactly sessions/seeds 0,1,2")
    if len({str(record.get("run_id")) for record in session_records}) != 3:
        raise NativeFinalError("final process sessions must have distinct run IDs")
    primary = str(final_config["final_primary_experiment_id"])
    proposed = str(final_config["final_primary_method"])
    expected_session_ids = {
        f"session-seed-{value}:process-0000" for value in required_seeds
    }
    observed_measurement_sessions = {str(row.get("session_id")) for row in rows}
    if observed_measurement_sessions != expected_session_ids:
        raise NativeFinalError(
            "final raw evidence must contain exactly one process-0000 segment per seed"
        )
    resamples = int(
        bootstrap_resamples
        if bootstrap_resamples is not None
        else final_config.get("bootstrap_resamples", MIN_BOOTSTRAP_RESAMPLES)
    )
    seed = int(
        bootstrap_seed
        if bootstrap_seed is not None
        else final_config.get("bootstrap_seed", DEFAULT_BOOTSTRAP_SEED)
    )
    analysis = analyze_native_rows(
        rows, bootstrap_resamples=resamples, bootstrap_seed=seed
    )
    if analysis["integrity_checks"]["native_backend_calls"]["passed"] is not True:
        raise NativeFinalError("final native backend/call integrity failed")
    if analysis["integrity_checks"]["correctness"]["passed"] is not True:
        raise NativeFinalError("final certified-search correctness checks failed")
    primary_rows = [row for row in rows if row.get("experiment_id") == primary]
    proposed_rows = [row for row in primary_rows if row.get("method") == proposed]
    expected_queries = int(final_config["final_gate"]["required_primary_queries"])
    expected_ids = {str(value) for value in range(1200, 2200)}
    observed_ids = {str(row["query_id"]) for row in proposed_rows}
    if observed_ids != expected_ids or len(observed_ids) != expected_queries:
        raise NativeFinalError("fresh SIFT primary query coverage is not exactly 1200..2199")
    selected_beta = Fraction.from_float(
        float(lock["selected_operating_point"]["requested_beta_l2"])
    )
    observed_betas = {
        Fraction.from_float(float(row["requested_beta_l2"])) for row in proposed_rows
    }
    if observed_betas != {selected_beta}:
        raise NativeFinalError("fresh final P beta differs from the validation lock")
    if {str(row.get("session_id")) for row in proposed_rows} != expected_session_ids:
        raise NativeFinalError("fresh primary does not contain all three process sessions")

    comparisons = [
        row
        for row in analysis["comparisons"]
        if row["experiment_id"] == primary
        and row["proposed_method"] == proposed
        and row["metric"] == "api_wall"
        and row["baseline_role"] in {"F", "A"}
    ]
    by_role = {str(row["baseline_role"]): row for row in comparisons}
    if set(by_role) != {"F", "A"}:
        raise NativeFinalError("fresh final lacks exactly one F/P and A/P comparison")
    gate_policy = final_config["final_gate"]
    minimum_speedup = float(
        gate_policy["minimum_geometric_mean_speedup_vs_each_baseline"]
    )
    maximum_p95_ratio = float(
        gate_policy["maximum_p95_latency_ratio_vs_each_baseline"]
    )
    p_values = _method_query_latencies(rows, experiment=primary, method=proposed)
    comparison_outcomes: dict[str, Any] = {}
    for role in ("F", "A"):
        comparison = by_role[role]
        baseline_method = str(comparison["baseline_method"])
        baseline_values = _method_query_latencies(
            rows, experiment=primary, method=baseline_method
        )
        if set(baseline_values) != set(p_values):
            raise NativeFinalError(f"unpaired primary query latencies versus {role}")
        p95_p = float(np.quantile(list(p_values.values()), 0.95))
        p95_baseline = float(np.quantile(list(baseline_values.values()), 0.95))
        p95_ratio = p95_p / p95_baseline
        speedup = float(comparison["geometric_mean"])
        ci_lower = comparison.get("bootstrap_95pct_lower")
        ci_supports_speedup = ci_lower is not None and float(ci_lower) > 1.0
        complete_pairs = (
            int(comparison["paired_queries"]) == expected_queries
            and comparison["sessions_per_query"] == {"min": 3, "max": 3}
            and int(comparison["unpaired_baseline_session_queries"]) == 0
            and int(comparison["unpaired_proposed_session_queries"]) == 0
        )
        quality_match = (
            True
            if role == "F"
            else int(comparison["baseline_quality_mismatch_rows_retained"]) == 0
        )
        performance_passed = (
            complete_pairs
            and ci_supports_speedup
            and (
                quality_match
                or not bool(gate_policy.get("require_A_quality_match", True))
            )
        )
        engineering_go = (
            performance_passed
            and speedup >= minimum_speedup
            and p95_ratio <= maximum_p95_ratio
        )
        comparison_outcomes[role] = {
            "passed": performance_passed,
            "performance_gate_passed": performance_passed,
            "engineering_GO_passed": engineering_go,
            "complete_query_session_pairs": complete_pairs,
            "geometric_mean_speedup": speedup,
            "minimum_required_speedup": minimum_speedup,
            "bootstrap_95pct_lower": ci_lower,
            "bootstrap_95pct_upper": comparison["bootstrap_95pct_upper"],
            "bootstrap_95pct_lower_must_be_strictly_above": 1.0,
            "bootstrap_CI_supports_speedup": ci_supports_speedup,
            "P_query_folded_p95_ns": p95_p,
            "baseline_query_folded_p95_ns": p95_baseline,
            "P_over_baseline_p95_ratio": p95_ratio,
            "maximum_allowed_p95_ratio": maximum_p95_ratio,
            "A_quality_match": quality_match if role == "A" else None,
            "baseline_quality_mismatch_rows_retained": comparison[
                "baseline_quality_mismatch_rows_retained"
            ],
            "paired_comparison": comparison,
        }
    performance_passed = all(
        value["performance_gate_passed"] for value in comparison_outcomes.values()
    )
    influence_by_query: dict[tuple[str, int], set[bool]] = {}
    for row in proposed_rows:
        value = row.get("delta_influence")
        if not isinstance(value, bool):
            raise NativeFinalError("fresh primary is missing a Delta-influence label")
        identity = (str(row["query_id"]), int(row.get("query_position", -1)))
        influence_by_query.setdefault(identity, set()).add(value)
    if any(len(values) != 1 for values in influence_by_query.values()):
        raise NativeFinalError("fresh primary Delta-influence labels change across sessions")
    influenced_queries = sum(next(iter(values)) for values in influence_by_query.values())
    influence_evidence_complete = (
        len(influence_by_query) == expected_queries and influenced_queries > 0
    )
    configured_experiments = final_config.get("experiments")
    if not isinstance(configured_experiments, list) or not configured_experiments:
        raise NativeFinalError("locked final config has no experiment set")
    expected_build_experiments = {
        str(experiment["id"])
        for experiment in configured_experiments
        if isinstance(experiment, dict) and "id" in experiment
    }
    if len(expected_build_experiments) != len(configured_experiments):
        raise NativeFinalError("locked final config has malformed/duplicate experiments")
    required_build_fields = (
        "base_build_wall_ns",
        "base_input_vector_bytes",
        "group_build_wall_ns",
        "group_member_bytes",
        "group_metadata_bytes",
        "group_total_bytes",
        "native_packed_build_ns",
        "native_packed_python_owned_bytes",
        "native_packed_native_owned_bytes",
        "native_packed_bytes",
    )
    build_cost_evidence_complete = all(
        isinstance(record.get("build_manifest_sha256"), str)
        and len(str(record["build_manifest_sha256"])) == 64
        and all(
            character in "0123456789abcdef"
            for character in str(record["build_manifest_sha256"])
        )
        and isinstance(record.get("build_costs"), dict)
        and set(record["build_costs"]) == expected_build_experiments
        and all(
            isinstance(record["build_costs"].get(experiment_id), dict)
            and all(
                not isinstance(
                    record["build_costs"][experiment_id].get(field), bool
                )
                and isinstance(
                    record["build_costs"][experiment_id].get(field), int
                )
                and int(record["build_costs"][experiment_id][field]) >= 0
                for field in required_build_fields
            )
            for experiment_id in expected_build_experiments
        )
        for record in session_records
    )
    engineering_reasons: list[str] = []
    for role, outcome in comparison_outcomes.items():
        if not outcome["performance_gate_passed"]:
            engineering_reasons.append(f"performance_gate_failed_vs_{role}")
        elif not outcome["engineering_GO_passed"]:
            engineering_reasons.append(f"speedup_or_p95_requirement_failed_vs_{role}")
    if not influence_evidence_complete:
        engineering_reasons.append("fresh_delta_influence_subset_missing")
    if not build_cost_evidence_complete:
        engineering_reasons.append("build_cost_evidence_missing")
    engineering_go = performance_passed and not engineering_reasons
    hnsw_status = str(hnsw_reference.get("status", ""))
    if require_post_gate_reference:
        if performance_passed and hnsw_status != "completed":
            raise NativeFinalError(
                "performance-passed final requires completed post-gate HNSW references"
            )
        if (
            not performance_passed
            and hnsw_status != "not_run_performance_gate_not_passed"
        ):
            raise NativeFinalError(
                "performance-negative final must record that HNSW was not run"
            )
    final_gate = {
        "schema_version": FINAL_SCHEMA_VERSION,
        "study_id": STUDY_ID,
        "gate_status": "PASSED" if performance_passed else "NOT_PASSED",
        "performance_gate_status": (
            "PASSED" if performance_passed else "NOT_PASSED"
        ),
        "performance_verdict": (
            "SUPPORTED_IN_TESTED_REGIME"
            if performance_passed
            else "NOT_SUPPORTED_IN_TESTED_REGIME"
        ),
        "verdict": (
            "SUPPORTED_IN_TESTED_REGIME"
            if performance_passed
            else "NOT_SUPPORTED_IN_TESTED_REGIME"
        ),
        "engineering_decision": "GO" if engineering_go else "NO_GO",
        "engineering_reasons": engineering_reasons,
        "primary_dataset": "sift",
        "primary_experiment_id": primary,
        "primary_proposed_method": proposed,
        "fresh_query_interval": [1200, 2200],
        "fresh_query_count": expected_queries,
        "process_session_seeds": required_seeds,
        "comparison_outcomes": comparison_outcomes,
        "fresh_delta_influence": {
            "labelled_queries": len(influence_by_query),
            "influenced_queries": influenced_queries,
            "required_minimum": 1,
            "evidence_complete": influence_evidence_complete,
        },
        "build_cost_evidence_complete": build_cost_evidence_complete,
        "build_cost_experiments": sorted(expected_build_experiments),
        "gist_used_in_gate": False,
        "hnsw_references_used_in_gate": False,
        "hnsw_reference_status": hnsw_status,
        "passed": performance_passed,
    }
    anchor_summaries = [
        operating_point
        for operating_point in analysis["operating_points"]
        if operating_point["experiment_id"] != primary
        or operating_point["proposed_method"] != proposed
    ]
    summary = {
        "schema_version": FINAL_SCHEMA_VERSION,
        "study_id": STUDY_ID,
        "evidence_role": "final",
        "fresh_primary": {
            "dataset": "sift",
            "experiment_id": primary,
            "query_interval": [1200, 2200],
            "query_ids_sha256": _id_set_sha256(1200, 2200),
            "independence_label": "fresh_pre_registered_holdout",
            "proposed_method": proposed,
            "requested_beta_l2": float(selected_beta),
        },
        "sessions": [dict(value) for value in session_records],
        "primary_session_metrics": _primary_session_metrics(
            rows,
            experiment=primary,
            proposed=proposed,
            expected_queries=expected_queries,
        ),
        "native_analysis": analysis,
        "anchor_operating_points": anchor_summaries,
        "gist_interpretation": {
            "independence_label": "reused_non_independent",
            "gate_eligible": False,
            "note": "all distributed GIST query IDs 0..999 were previously used",
        },
        "hnsw_references": dict(hnsw_reference),
        "final_gate": final_gate,
    }
    return summary, final_gate


__all__ = [
    "FINAL_SCHEMA_VERSION",
    "NativeFinalError",
    "evaluate_final_rows",
    "method_name_for_factor",
    "prepare_final_authorization",
    "read_object",
    "session_effective_config",
    "summarize_build_costs",
]

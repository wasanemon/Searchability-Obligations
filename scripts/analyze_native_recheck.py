#!/usr/bin/env python3
"""Analyze saved Issue #3 native rows and write a durable validation gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from searchability.artifacts import atomic_write_json, file_sha256
from searchability.native_analysis import (
    DEFAULT_BOOTSTRAP_SEED,
    METRIC_FIELDS,
    MIN_BOOTSTRAP_RESAMPLES,
    analyze_native_path,
    evaluate_validation_gate,
    load_native_evidence,
)


def _method_role(value: str) -> tuple[str, str]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("method role must be METHOD=ROLE")
    method, role = value.split("=", 1)
    if not method or role not in {"O", "F", "A", "N", "P", "P-old"}:
        raise argparse.ArgumentTypeError(
            "role must be one of O, F, A, N, P, P-old"
        )
    return method, role


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Checksum-verify completed native runs, aggregate F/P and A/P "
            "paired speedups, and optionally persist the validation gate."
        )
    )
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--gate-output",
        type=Path,
        help="Write PASSED or durable NOT_PASSED validation decision",
    )
    parser.add_argument(
        "--primary-metric", choices=tuple(METRIC_FIELDS), default="api_wall"
    )
    parser.add_argument(
        "--bootstrap-resamples", type=int, default=MIN_BOOTSTRAP_RESAMPLES
    )
    parser.add_argument("--bootstrap-seed", type=int, default=DEFAULT_BOOTSTRAP_SEED)
    parser.add_argument(
        "--method-role",
        action="append",
        type=_method_role,
        default=[],
        metavar="METHOD=ROLE",
        help="Map a runner-specific method name to an analysis role",
    )
    parser.add_argument(
        "--representative-raw-output",
        type=Path,
        help="Export one deterministic, fully preserved raw row per experiment/method",
    )
    args = parser.parse_args()
    if args.bootstrap_resamples < MIN_BOOTSTRAP_RESAMPLES:
        parser.error(
            f"--bootstrap-resamples must be at least {MIN_BOOTSTRAP_RESAMPLES}"
        )
    roles = dict(args.method_role)
    summary = analyze_native_path(
        args.input,
        method_roles=roles or None,
        bootstrap_resamples=args.bootstrap_resamples,
        bootstrap_seed=args.bootstrap_seed,
    )
    gate = evaluate_validation_gate(summary, primary_metric=args.primary_metric)
    summary["validation_gate"] = gate
    atomic_write_json(args.output, summary)
    if args.gate_output is not None:
        atomic_write_json(args.gate_output, gate)
    if args.representative_raw_output is not None:
        evidence = load_native_evidence(args.input)
        selected: dict[tuple[str, str], dict[str, object]] = {}

        def priority(row: dict[str, object]) -> tuple[object, ...]:
            exceptional = bool(row.get("contract_violation")) or bool(
                row.get("baseline_validation_failure")
            )
            return (
                0 if exceptional else 1,
                0 if row.get("delta_influence") is True else 1,
                int(row.get("query_position", -1)),
                int(row.get("repetition", -1)),
                str(row.get("session_id", "")),
                str(row.get("run_id", "")),
            )

        for row in evidence.rows:
            key = (str(row["experiment_id"]), str(row["method"]))
            current = selected.get(key)
            if current is None or priority(row) < priority(current):
                selected[key] = dict(row)
        representative = {
            "schema_version": 1,
            "study_id": "issue-3-native-recheck",
            "selection_policy": (
                "one row per (experiment_id,method); contract/A-quality failures first, "
                "then Delta-influence, query_position, repetition, session_id, run_id"
            ),
            "source_input": str(args.input),
            "source_summary_sha256": file_sha256(args.output),
            "source_gate_sha256": (
                None if args.gate_output is None else file_sha256(args.gate_output)
            ),
            "source_raw_rows": len(evidence.rows),
            "source_runs": list(evidence.runs),
            "representative_rows": [selected[key] for key in sorted(selected)],
        }
        atomic_write_json(args.representative_raw_output, representative)
    print(
        json.dumps(
            {
                "summary": str(args.output),
                "gate": None if args.gate_output is None else str(args.gate_output),
                "gate_status": gate["gate_status"],
                "lock_created": False,
                "raw_rows": summary["totals"]["raw_rows"],
                "representative_raw": (
                    None
                    if args.representative_raw_output is None
                    else str(args.representative_raw_output)
                ),
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    # NOT_PASSED is a completed negative validation outcome.  Corrupt evidence
    # still raises above and returns non-zero; a failed performance gate does not.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

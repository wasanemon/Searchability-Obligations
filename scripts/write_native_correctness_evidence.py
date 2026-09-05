#!/usr/bin/env python3
"""Bind a passing native pytest JUnit file to code and binary identities."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import subprocess
import xml.etree.ElementTree as ET

from searchability.artifacts import (
    atomic_write_json,
    file_sha256,
    implementation_tree_sha256,
    native_test_tree_sha256,
)
from searchability.native import native_build_info


ROOT = Path(__file__).resolve().parents[1]
REQUIRED_NAME_FRAGMENTS = (
    "fixed_seed_10000",
    "duplicate_visibility",
    "population_below_k",
    "positive_beta",
    "equal_distance_identifier_ties",
)


def _test_tree_sha256() -> str:
    return native_test_tree_sha256(ROOT)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--junit", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--command", required=True)
    parser.add_argument("--pytest-exit-code", type=int, default=0)
    parser.add_argument(
        "--mark-running",
        action="store_true",
        help="Atomically invalidate any older pass before starting pytest",
    )
    args = parser.parse_args()

    build = dict(native_build_info())
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    ).stdout.strip() or None
    identity = {
        "schema_version": 1,
        "study_id": "issue-3-native-recheck",
        "command": args.command,
        "implementation_tree_sha256": implementation_tree_sha256(ROOT),
        "test_tree_sha256": _test_tree_sha256(),
        "native_shared_object_sha256": build["shared_object_sha256"],
        "native_build": build,
        "git_commit": commit,
    }
    if args.mark_running:
        atomic_write_json(
            args.output,
            {
                **identity,
                "status": "running_invalidates_prior_pass",
                "captured_at_utc": datetime.now(timezone.utc).isoformat(),
                "fixed_seed_cases": 0,
            },
        )
        return 0

    if args.junit is None:
        parser.error("--junit is required unless --mark-running is used")

    parse_error: str | None = None
    try:
        tree = ET.parse(args.junit)
        root = tree.getroot()
        suites = [root] if root.tag == "testsuite" else list(root.findall("testsuite"))
        testcases = list(root.iter("testcase"))
        names = [str(case.get("name", "")) for case in testcases]
        fixed_seed_properties = [
            str(prop.get("value", ""))
            for case in testcases
            if "fixed_seed_10000" in str(case.get("name", ""))
            for prop in case.findall("./properties/property")
            if prop.get("name") == "native_case_count"
        ]
        failures = sum(int(suite.get("failures", "0")) for suite in suites)
        errors = sum(int(suite.get("errors", "0")) for suite in suites)
        skipped = sum(int(suite.get("skipped", "0")) for suite in suites)
    except (OSError, ET.ParseError, ValueError) as error:
        parse_error = f"{type(error).__name__}: {error}"
        testcases = []
        names = []
        fixed_seed_properties = []
        failures = errors = skipped = 0
    missing = [
        fragment for fragment in REQUIRED_NAME_FRAGMENTS
        if not any(fragment in name for name in names)
    ]
    reasons = []
    if args.pytest_exit_code != 0:
        reasons.append(f"pytest_exit_code_{args.pytest_exit_code}")
    if parse_error is not None:
        reasons.append("junit_unreadable")
    if failures:
        reasons.append("pytest_failures")
    if errors:
        reasons.append("pytest_errors")
    if skipped:
        reasons.append("pytest_skips")
    if not testcases:
        reasons.append("no_testcases")
    if missing:
        reasons.append("required_test_names_missing")
    if fixed_seed_properties != ["10000"]:
        reasons.append("fixed_seed_case_count_property_missing_or_invalid")
    passed = not reasons
    evidence = {
        **identity,
        "status": "passed" if passed else "failed",
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "pytest_exit_code": args.pytest_exit_code,
        "junit_path": str(args.junit),
        "junit_sha256": file_sha256(args.junit) if args.junit.is_file() else None,
        "junit_parse_error": parse_error,
        "pytest_testcases": len(testcases),
        "pytest_failures": failures,
        "pytest_errors": errors,
        "pytest_skipped": skipped,
        "required_test_name_fragments": list(REQUIRED_NAME_FRAGMENTS),
        "missing_required_test_name_fragments": missing,
        "failure_reasons": reasons,
        "fixed_seed_case_count_properties": fixed_seed_properties,
        "fixed_seed_cases": 10_000 if passed else 0,
        "fixed_seed_policy": (
            "the required JUnit testcase whose name contains fixed_seed_10000 "
            "executes exactly 10,000 deterministic generated cases"
        ),
        "thread_policy": {
            "threads": 1,
            "environment_required_by_make": [
                "OMP_NUM_THREADS=1",
                "OPENBLAS_NUM_THREADS=1",
                "MKL_NUM_THREADS=1",
                "NUMEXPR_NUM_THREADS=1",
                "VECLIB_MAXIMUM_THREADS=1",
            ],
        },
    }
    atomic_write_json(args.output, evidence)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())

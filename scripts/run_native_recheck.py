#!/usr/bin/env python3
"""Run or resume the Issue #3 native-kernel recheck."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shlex

from searchability.artifacts import file_sha256, object_sha256
from searchability.benchmark import load_json_config
from searchability.native_benchmark import (
    apply_native_resource_limits,
    run_native_benchmark,
)


def _positive(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def _nonnegative(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be non-negative")
    return parsed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--run-id")
    parser.add_argument("--session-id")
    parser.add_argument("--stop-after-blocks", type=_positive)
    parser.add_argument("--only-experiment")
    parser.add_argument("--max-experiments", type=_positive)
    parser.add_argument("--max-base", type=_positive)
    parser.add_argument("--max-delta", type=_nonnegative)
    parser.add_argument("--max-validation-queries", type=_positive)
    parser.add_argument("--max-test-queries", type=_positive)
    parser.add_argument("--max-repetitions", type=_positive)
    args = parser.parse_args()

    source = load_json_config(args.config)
    effective = apply_native_resource_limits(
        source,
        only_experiment=args.only_experiment,
        max_experiments=args.max_experiments,
        max_base=args.max_base,
        max_delta=args.max_delta,
        max_validation_queries=args.max_validation_queries,
        max_test_queries=args.max_test_queries,
        max_repetitions=args.max_repetitions,
    )
    effective["source_config_object_hash"] = object_sha256(source)
    effective["source_config_file_sha256"] = file_sha256(args.config)
    if args.session_id is not None:
        effective["session_id"] = args.session_id
    outcome = run_native_benchmark(
        effective,
        resume=args.resume,
        run_id=args.run_id,
        stop_after_blocks=args.stop_after_blocks,
    )
    resume = [
        ".venv/bin/python",
        "scripts/run_native_recheck.py",
        "--config",
        str(args.config),
        "--resume",
        "--run-id",
        outcome.run_id,
    ]
    if args.session_id is not None:
        resume.extend(("--session-id", args.session_id))
    for flag, value in (
        ("--only-experiment", args.only_experiment),
        ("--max-experiments", args.max_experiments),
        ("--max-base", args.max_base),
        ("--max-delta", args.max_delta),
        ("--max-validation-queries", args.max_validation_queries),
        ("--max-test-queries", args.max_test_queries),
        ("--max-repetitions", args.max_repetitions),
    ):
        if value is not None:
            resume.extend((flag, str(value)))
    print(
        json.dumps(
            {
                "run_id": outcome.run_id,
                "run_dir": str(outcome.run_dir),
                "completed": outcome.completed,
                "completed_blocks": outcome.completed_blocks,
                "total_blocks": outcome.total_blocks,
                "resume_command": (
                    None
                    if outcome.completed
                    else " ".join(shlex.quote(value) for value in resume)
                ),
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if outcome.completed else 2


if __name__ == "__main__":
    raise SystemExit(main())

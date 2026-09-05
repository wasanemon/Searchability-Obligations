#!/usr/bin/env python3
"""Run or resume the JSON-configured common benchmark."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shlex

from searchability.artifacts import object_sha256
from searchability.benchmark import (
    apply_resource_limits,
    load_json_config,
    run_benchmark,
)


def _non_negative(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be non-negative")
    return parsed


def _positive(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run the Issue #1 benchmark. Raw query blocks and checkpoints are "
            "written atomically; completed run directories are immutable."
        )
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--run-id", help="Select an explicit new/incomplete run ID")
    parser.add_argument(
        "--stop-after-blocks",
        type=_positive,
        help="Controlled incomplete stop for checkpoint/restart validation",
    )
    parser.add_argument("--only-experiment")
    parser.add_argument("--max-experiments", type=_positive)
    parser.add_argument("--max-base", type=_positive)
    parser.add_argument("--max-delta", type=_non_negative)
    parser.add_argument("--max-validation-queries", type=_positive)
    parser.add_argument("--max-test-queries", type=_positive)
    parser.add_argument("--max-repetitions", type=_positive)
    args = parser.parse_args()

    source = load_json_config(args.config)
    effective = apply_resource_limits(
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
    outcome = run_benchmark(
        effective,
        resume=args.resume,
        run_id=args.run_id,
        stop_after_blocks=args.stop_after_blocks,
    )
    resume_parts = [
        ".venv/bin/python",
        "scripts/run_experiment.py",
        "--config",
        str(args.config),
        "--resume",
        "--run-id",
        outcome.run_id,
    ]
    for option, value in (
        ("--only-experiment", args.only_experiment),
        ("--max-experiments", args.max_experiments),
        ("--max-base", args.max_base),
        ("--max-delta", args.max_delta),
        ("--max-validation-queries", args.max_validation_queries),
        ("--max-test-queries", args.max_test_queries),
        ("--max-repetitions", args.max_repetitions),
    ):
        if value is not None:
            resume_parts.extend((option, str(value)))
    resume_command = " ".join(shlex.quote(part) for part in resume_parts)
    print(
        json.dumps(
            {
                "run_id": outcome.run_id,
                "run_dir": str(outcome.run_dir),
                "completed": outcome.completed,
                "completed_blocks": outcome.completed_blocks,
                "total_blocks": outcome.total_blocks,
                "resume_command": (
                    resume_command
                    if not outcome.completed
                    else None
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

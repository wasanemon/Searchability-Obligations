#!/usr/bin/env python3
"""Lock (or durably decline) the Issue #3 fresh-final evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from searchability.artifacts import atomic_write_json, file_sha256, object_sha256
from searchability.native_analysis import analyze_native_path, evaluate_validation_gate
from searchability.native_final import prepare_final_authorization, read_object


def _reject_negative_final_artifacts(paths: tuple[Path, ...]) -> None:
    existing = [str(path) for path in paths if path.exists()]
    if existing:
        raise RuntimeError(
            "validation is NOT_PASSED but final artifacts already exist: "
            + ", ".join(existing)
        )


def _verify_registered_hnsw_outputs(
    policy: dict[str, object],
    *,
    pre_hnsw_authorization_output: Path,
    hnsw_output: Path,
) -> None:
    if (
        Path(str(policy.get("pre_hnsw_authorization", "")))
        != pre_hnsw_authorization_output
        or Path(str(policy.get("hnsw_output_root", ""))) != hnsw_output
    ):
        raise RuntimeError(
            "pre-HNSW authorization/HNSW outputs differ from the registered policy"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gate", required=True, type=Path)
    parser.add_argument("--summary", required=True, type=Path)
    parser.add_argument("--validation-config", required=True, type=Path)
    parser.add_argument("--validation-input", required=True, type=Path)
    parser.add_argument("--correctness", required=True, type=Path)
    parser.add_argument("--holdout", required=True, type=Path)
    parser.add_argument("--policy", required=True, type=Path)
    parser.add_argument("--decision-output", required=True, type=Path)
    parser.add_argument("--lock-output", required=True, type=Path)
    parser.add_argument("--final-config-output", required=True, type=Path)
    parser.add_argument("--final-summary-output", required=True, type=Path)
    parser.add_argument("--final-gate-output", required=True, type=Path)
    parser.add_argument("--pre-hnsw-authorization-output", required=True, type=Path)
    parser.add_argument("--hnsw-output", required=True, type=Path)
    args = parser.parse_args()

    prior_time: str | None = None
    for path in (args.lock_output, args.decision_output):
        if path.is_file():
            prior = read_object(path)
            candidate = prior.get("decided_at_utc")
            if candidate is not None:
                prior_time = str(candidate)
                break
    saved_gate = read_object(args.gate)
    saved_summary = read_object(args.summary)
    policy = read_object(args.policy)
    _verify_registered_hnsw_outputs(
        policy,
        pre_hnsw_authorization_output=args.pre_hnsw_authorization_output,
        hnsw_output=args.hnsw_output,
    )
    bootstrap = saved_summary.get("bootstrap_policy")
    if not isinstance(bootstrap, dict):
        raise RuntimeError("validation summary has no bootstrap policy")
    recomputed_summary = analyze_native_path(
        args.validation_input,
        bootstrap_resamples=int(bootstrap["resamples"]),
        bootstrap_seed=int(bootstrap["base_seed"]),
    )
    recomputed_gate = evaluate_validation_gate(
        recomputed_summary, primary_metric=str(saved_gate.get("primary_metric", "api_wall"))
    )
    recomputed_summary["validation_gate"] = recomputed_gate
    if recomputed_gate != saved_gate or recomputed_summary != saved_summary:
        raise RuntimeError("saved validation summary/gate do not reproduce from raw manifests")
    decision, lock, final_config = prepare_final_authorization(
        gate=saved_gate,
        summary=saved_summary,
        validation_config=read_object(args.validation_config),
        correctness=read_object(args.correctness),
        holdout=read_object(args.holdout),
        policy=policy,
        gate_path=args.gate,
        summary_path=args.summary,
        validation_config_path=args.validation_config,
        correctness_path=args.correctness,
        holdout_path=args.holdout,
        policy_path=args.policy,
        decided_at_utc=prior_time,
    )

    if lock is None:
        _reject_negative_final_artifacts(
            (
                args.lock_output,
                args.final_config_output,
                args.final_summary_output,
                args.final_gate_output,
                args.pre_hnsw_authorization_output,
                args.hnsw_output,
            )
        )
        if args.decision_output.exists() and read_object(args.decision_output) != decision:
            raise RuntimeError("refusing to rewrite a different negative final decision")
        atomic_write_json(args.decision_output, decision)
        print(json.dumps(decision, ensure_ascii=False, indent=2, sort_keys=True))
        return 0

    assert final_config is not None
    if args.lock_output.exists() and read_object(args.lock_output) != lock:
        raise RuntimeError("refusing to rewrite a different final lock")
    atomic_write_json(args.lock_output, lock)
    lock_sha256 = file_sha256(args.lock_output)
    final_config = {
        **final_config,
        "authorization": {
            "final_lock_path": str(args.lock_output),
            "final_lock_sha256": lock_sha256,
            "validation_gate_sha256": file_sha256(args.gate),
            "validation_summary_sha256": file_sha256(args.summary),
            "correctness_sha256": file_sha256(args.correctness),
            "holdout_manifest_sha256": file_sha256(args.holdout),
        },
    }
    if (
        args.final_config_output.exists()
        and read_object(args.final_config_output) != final_config
    ):
        raise RuntimeError("refusing to rewrite a different locked final config")
    atomic_write_json(args.final_config_output, final_config)
    decision.update(
        {
            "final_lock_sha256": lock_sha256,
            "final_config_file_sha256": file_sha256(args.final_config_output),
            "final_config_object_sha256": object_sha256(final_config),
        }
    )
    # A previous completed final decision is immutable. Re-running the
    # authorization step verifies it, but must not roll it back to pending.
    if args.decision_output.exists():
        prior = read_object(args.decision_output)
        if str(prior.get("final_status", "")).startswith("COMPLETED_"):
            print(json.dumps(prior, ensure_ascii=False, indent=2, sort_keys=True))
            return 0
        if prior != decision:
            raise RuntimeError("refusing to rewrite a different final authorization")
    atomic_write_json(args.decision_output, decision)
    print(json.dumps(decision, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

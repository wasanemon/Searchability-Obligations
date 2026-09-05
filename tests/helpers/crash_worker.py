"""Subprocess entry point that deliberately terminates at a store fault point."""

from __future__ import annotations

import argparse

import numpy as np

from searchability.store import CommitOperation, LifecycleStore


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--store", required=True)
    parser.add_argument("--fault", required=True)
    parser.add_argument(
        "--action",
        choices=("insert", "update", "delete", "generation", "group"),
        required=True,
    )
    parser.add_argument("--logical-id", type=int, default=7)
    parser.add_argument("--generation-id", default="crash-generation")
    arguments = parser.parse_args()

    store = LifecycleStore(arguments.store, fault_point=arguments.fault, threads=1)
    if arguments.action == "insert":
        store.commit(
            [
                CommitOperation.insert(
                    arguments.logical_id, np.array([1.0, -2.0], dtype=np.float32)
                )
            ]
        )
    elif arguments.action == "update":
        store.commit(
            [
                CommitOperation.update(
                    arguments.logical_id, np.array([9.0, 3.0], dtype=np.float32)
                )
            ]
        )
    elif arguments.action == "delete":
        store.commit([CommitOperation.delete(arguments.logical_id)])
    elif arguments.action == "generation":
        store.build_generation(arguments.generation_id)
    else:
        store.group_pending(np.array([[0.0, 0.0]], dtype=np.float32))
    # Reaching here means the requested fault point was not exercised.
    store.close()
    raise SystemExit(3)


if __name__ == "__main__":
    main()

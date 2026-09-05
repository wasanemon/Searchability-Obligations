from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from searchability.store import LifecycleStore


WORKER = Path(__file__).parent / "helpers" / "crash_worker.py"
QUERY = np.array([0.0, 0.0], dtype=np.float32)


def _crash(path: Path, *, action: str, fault: str) -> subprocess.CompletedProcess[str]:
    environment = dict(os.environ)
    environment.update(
        {
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
        }
    )
    return subprocess.run(
        [
            sys.executable,
            str(WORKER),
            "--store",
            str(path),
            "--action",
            action,
            "--fault",
            fault,
        ],
        cwd=Path(__file__).parents[1],
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=30,
        check=False,
    )


@pytest.mark.crash
def test_process_exit_inside_transaction_leaves_neither_vector_nor_obligation(
    tmp_path: Path,
) -> None:
    path = tmp_path / "store"
    with LifecycleStore(path, dimension=2) as store:
        store.build_generation("empty-base")
    completed = _crash(
        path, action="insert", fault="commit_before_sql_commit"
    )
    assert completed.returncode == 86, completed.stderr
    with LifecycleStore(path) as recovered:
        assert recovered.snapshot() == 0
        assert recovered.visible_keys() == ()
        assert recovered.obligation_rows() == ()
        assert recovered.search(QUERY, 1).hits == ()


@pytest.mark.crash
def test_process_exit_immediately_after_commit_recovers_raw_searchable_obligation(
    tmp_path: Path,
) -> None:
    path = tmp_path / "store"
    with LifecycleStore(path, dimension=2) as store:
        store.build_generation("empty-base")
    completed = _crash(path, action="insert", fault="commit_after_sql_commit")
    assert completed.returncode == 86, completed.stderr
    with LifecycleStore(path) as recovered:
        assert recovered.snapshot() == 1
        assert recovered.visible_keys() == ((7, 1),)
        assert len(recovered.obligation_rows()) == 1
        assert tuple(record.key for record in recovered.raw_pending()) == ((7, 1),)
        result = recovered.search(QUERY, 1)
        assert tuple(hit.key for hit in result.hits) == ((7, 1),)
        assert result.receipt.raw_pending_scanned == 1


@pytest.mark.crash
def test_process_exit_mid_update_keeps_old_interval_and_has_no_new_obligation(
    tmp_path: Path,
) -> None:
    path = tmp_path / "store"
    with LifecycleStore(path, dimension=2) as store:
        store.insert(7, [1.0, -2.0])
        store.build_generation("old-base")
    completed = _crash(path, action="update", fault="commit_before_sql_commit")
    assert completed.returncode == 86, completed.stderr
    with LifecycleStore(path) as recovered:
        assert recovered.snapshot() == 1
        assert recovered.visible_keys() == ((7, 1),)
        assert [row["kind"] for row in recovered.obligation_rows()] == ["insert"]
        result = recovered.search(QUERY, 1)
        assert tuple(hit.key for hit in result.hits) == ((7, 1),)
        assert result.receipt.raw_pending_scanned == 0


@pytest.mark.crash
@pytest.mark.parametrize(
    ("action", "fault", "expected_snapshot", "expected_keys", "expected_kinds"),
    [
        (
            "insert",
            "commit_insert_after_version_before_obligation",
            0,
            (),
            [],
        ),
        (
            "update",
            "commit_update_after_versions_before_obligation",
            1,
            ((7, 1),),
            ["insert"],
        ),
        (
            "delete",
            "commit_delete_after_old_end_before_obligation",
            1,
            ((7, 1),),
            ["insert"],
        ),
    ],
)
def test_process_exit_between_vector_interval_mutation_and_obligation_rolls_back(
    tmp_path: Path,
    action: str,
    fault: str,
    expected_snapshot: int,
    expected_keys: tuple[tuple[int, int], ...],
    expected_kinds: list[str],
) -> None:
    path = tmp_path / action
    with LifecycleStore(path, dimension=2) as store:
        if action == "insert":
            store.build_generation("empty-base")
        else:
            store.insert(7, [1.0, -2.0])
            store.build_generation("old-base")

    completed = _crash(path, action=action, fault=fault)
    assert completed.returncode == 86, completed.stderr
    with LifecycleStore(path) as recovered:
        assert recovered.snapshot() == expected_snapshot
        assert recovered.visible_keys() == expected_keys
        assert [row["kind"] for row in recovered.obligation_rows()] == expected_kinds
        result = recovered.search(QUERY, 1)
        assert tuple(hit.key for hit in result.hits) == expected_keys


@pytest.mark.crash
@pytest.mark.parametrize(
    ("fault", "leftover_kind"),
    [
        ("generation_after_index_metadata", "temporary"),
        ("generation_after_files", "temporary"),
        ("generation_after_rename", "orphan"),
        ("generation_after_manifest", "orphan"),
        ("generation_before_catalog_commit", "orphan"),
        ("generation_after_catalog_commit", "published"),
    ],
)
def test_generation_crash_boundaries_use_old_plus_raw_or_valid_new(
    tmp_path: Path, fault: str, leftover_kind: str
) -> None:
    path = tmp_path / fault
    with LifecycleStore(path, dimension=2) as store:
        store.build_generation("empty-base")
        store.insert(7, [1.0, -2.0])

    completed = _crash(path, action="generation", fault=fault)
    assert completed.returncode == 86, completed.stderr
    with LifecycleStore(path) as recovered:
        report = recovered.last_recovery_report
        result = recovered.search(QUERY, 1)
        assert tuple(hit.key for hit in result.hits) == ((7, 1),)
        if leftover_kind == "temporary":
            assert report.temporary_directories
            assert report.current_generation == "empty-base"
            assert result.receipt.raw_pending_scanned == 1
        elif leftover_kind == "orphan":
            assert "crash-generation" in report.orphan_generations
            assert report.current_generation == "empty-base"
            assert result.receipt.raw_pending_scanned == 1
        else:
            assert report.current_generation == "crash-generation"
            assert "crash-generation" in report.valid_generations
            assert result.receipt.raw_pending_scanned == 0


@pytest.mark.crash
@pytest.mark.parametrize(
    ("fault", "expected_raw", "expected_groups"),
    [
        ("group_before_catalog_commit", 1, 0),
        ("group_after_catalog_commit", 0, 1),
    ],
)
def test_group_publish_crash_is_atomic_for_new_readers(
    tmp_path: Path, fault: str, expected_raw: int, expected_groups: int
) -> None:
    path = tmp_path / fault
    with LifecycleStore(path, dimension=2) as store:
        store.build_generation("empty-base")
        store.insert(7, [1.0, -2.0])
    completed = _crash(path, action="group", fault=fault)
    assert completed.returncode == 86, completed.stderr
    with LifecycleStore(path) as recovered:
        result = recovered.search(QUERY, 1)
        assert tuple(hit.key for hit in result.hits) == ((7, 1),)
        assert result.receipt.raw_pending_scanned == expected_raw
        assert result.receipt.groups_scanned == expected_groups

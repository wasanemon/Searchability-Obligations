from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from searchability.store import (
    CommitOperation,
    GC_NOT_IMPLEMENTED_CONSERVATIVE_RETENTION,
    LifecycleStore,
    PinnedGenerationError,
    PinnedViewError,
    SnapshotError,
)


QUERY = np.array([0.0, 0.0], dtype=np.float32)


def _keys(result: object) -> tuple[tuple[int, int], ...]:
    return tuple(hit.key for hit in result.hits)  # type: ignore[attr-defined]


def test_commit_keeps_vector_interval_and_obligation_in_one_sequence(tmp_path: Path) -> None:
    with LifecycleStore(tmp_path / "store", dimension=2) as store:
        first = store.commit(
            [
                {"op": "insert", "logical_id": 20, "vector": [3.0, 0.0]},
                CommitOperation.insert(10, [1.0, 0.0]),
            ]
        )
        second = store.update(10, [2.0, 0.0])
        third = store.delete(20)

        assert (first, second, third) == (1, 2, 3)
        assert store.visible_keys(first) == ((10, 2), (20, 1))
        assert store.visible_keys(second) == ((10, 3), (20, 1))
        assert store.visible_keys(third) == ((10, 3),)
        obligations = store.obligation_rows()
        assert [(row["kind"], row["commit_seq"]) for row in obligations] == [
            ("insert", 1),
            ("insert", 1),
            ("update", 2),
            ("delete", 3),
        ]
        assert obligations[2]["old_version_id"] == 2
        assert obligations[2]["new_version_id"] == 3
        assert obligations[3]["old_version_id"] == 1
        assert obligations[3]["new_version_id"] is None


def test_failed_multi_operation_commit_rolls_back_clock_vectors_and_obligations(
    tmp_path: Path,
) -> None:
    with LifecycleStore(tmp_path / "store", dimension=2) as store:
        store.insert(1, [0.0, 0.0])
        with pytest.raises(ValueError, match="already exists"):
            store.commit(
                [
                    CommitOperation.insert(2, [2.0, 0.0]),
                    CommitOperation.insert(1, [1.0, 0.0]),
                ]
            )
        assert store.snapshot() == 1
        assert store.visible_keys() == ((1, 1),)
        assert len(store.obligation_rows()) == 1


def test_reopen_reconstructs_committed_raw_pending_from_sqlite(tmp_path: Path) -> None:
    path = tmp_path / "store"
    with LifecycleStore(path, dimension=2) as store:
        store.build_generation("empty-base")
        snapshot = store.insert(8, [0.25, 0.0])
        assert tuple(record.key for record in store.raw_pending()) == ((8, 1),)
    with LifecycleStore(path) as reopened:
        assert reopened.snapshot() == snapshot
        assert tuple(record.key for record in reopened.raw_pending()) == ((8, 1),)
        result = reopened.search(QUERY, 1, beta=0.0)
        assert _keys(result) == ((8, 1),)
        assert result.receipt.raw_pending_scanned == 1
        assert result.receipt.base_generation == "empty-base"


def test_group_catalog_switch_has_raw_or_grouped_member_never_neither(
    tmp_path: Path,
) -> None:
    path = tmp_path / "store"
    writer = LifecycleStore(path, dimension=2)
    writer.build_generation("empty-base")
    writer.insert(5, [1.0, 0.0])
    reader = LifecycleStore(path)
    observations: list[tuple[int, int, tuple[tuple[int, int], ...]]] = []

    def observe(point: str) -> None:
        if point != "group_before_catalog_commit":
            return
        result = reader.search(QUERY, 1, beta=0.0)
        observations.append(
            (
                result.receipt.raw_pending_scanned,
                result.receipt.groups_scanned,
                _keys(result),
            )
        )

    writer._user_fault_injector = observe
    revision = writer.group_pending(np.array([[0.0, 0.0]], dtype=np.float32))
    after = reader.search(QUERY, 1, beta=0.0)
    assert revision == 1
    assert observations == [(1, 0, ((5, 1),))]
    assert after.receipt.raw_pending_scanned == 0
    assert after.receipt.groups_scanned == 1
    assert _keys(after) == ((5, 1),)
    reader.close()
    writer.close()


def test_pinned_old_catalog_remains_raw_after_new_catalog_publish(tmp_path: Path) -> None:
    with LifecycleStore(tmp_path / "store", dimension=2) as store:
        store.build_generation("empty-base")
        store.insert(1, [1.0, 0.0])
        with store.pin() as old_view:
            assert old_view.group_catalog_revision == 0
            store.group_pending(np.array([[0.0, 0.0]], dtype=np.float32))
            old_result = store.search(QUERY, 1, view=old_view)
            assert old_result.receipt.raw_pending_scanned == 1
            assert old_result.receipt.groups_scanned == 0
        new_result = store.search(QUERY, 1)
        assert new_result.receipt.raw_pending_scanned == 0
        assert new_result.receipt.groups_scanned == 1


def test_receipt_binds_group_revision_and_exact_delta_view_content(
    tmp_path: Path,
) -> None:
    with LifecycleStore(tmp_path / "store", dimension=2) as store:
        store.build_generation("empty-base")
        store.insert(1, [1.0, 0.0])
        first = store.search(QUERY, 1)
        repeated = store.search(QUERY, 1)

        assert first.receipt.group_catalog_revision == 0
        assert len(first.receipt.delta_view_hash or "") == 64
        assert first.receipt.delta_view_hash == repeated.receipt.delta_view_hash
        assert first.receipt.certificate_details is not None
        assert (
            first.receipt.certificate_details["delta_view_hash"]
            == first.receipt.delta_view_hash
        )

        store.insert(2, [2.0, 0.0])
        changed = store.search(QUERY, 1)
        assert changed.receipt.delta_view_hash != first.receipt.delta_view_hash


def test_invalid_group_center_falls_back_once_per_member_and_is_receipted(
    tmp_path: Path,
) -> None:
    with LifecycleStore(tmp_path / "store", dimension=2) as store:
        store.build_generation("empty-base")
        store.commit(
            [
                CommitOperation.insert(1, [1.0, 0.0]),
                CommitOperation.insert(2, [2.0, 0.0]),
            ]
        )
        revision = store.group_pending([[0.0, 0.0]])
        # Simulate catalog damage with a one-float center payload. The old
        # loader appended these members twice; a keyed raw fallback must expose
        # and scan each retained version exactly once.
        store._connection.execute(
            "UPDATE groups SET center = ? WHERE revision = ? AND group_id = 0",
            (b"\x00\x00\x00\x00", revision),
        )

        result = store.search(QUERY, 2)
        assert _keys(result) == ((1, 1), (2, 2))
        assert result.receipt.raw_pending_scanned == 2
        assert result.receipt.groups_scanned == 0
        assert result.receipt.group_catalog_revision == revision
        assert result.receipt.fallback_reason is not None
        assert "invalid_group_catalog_entries_scanned_raw" in (
            result.receipt.fallback_reason
        )
        details = result.receipt.certificate_details
        assert details is not None
        assert details["group_catalog_fallback_vector_count"] == 2
        assert details["group_catalog_fallback_group_count"] == 1
        assert details["group_catalog_fallback_details"] == [
            "group:0:invalid_center_shape_or_dimension"
        ]


def test_loaded_generation_initializes_integrity_metadata(tmp_path: Path) -> None:
    with LifecycleStore(tmp_path / "store", dimension=2, threads=1) as store:
        store.insert(1, [1.0, 0.0])
        store.build_generation("base")
        with store.pin() as view:
            loaded = store._generation_base(view)
        assert loaded.threads == 1
        assert len(loaded.universe_hash) == 64
        assert loaded._records_by_key == {(1, 1): loaded.records[0]}
        assert loaded._visible_latest_cache == {}
        assert loaded.vectors.dtype == np.float32
        assert loaded.vectors.flags.c_contiguous
        assert not loaded.vectors.flags.owndata
        with pytest.raises(ValueError, match="WRITEABLE"):
            loaded.vectors.setflags(write=True)


def test_loaded_generation_uses_current_runtime_thread_setting(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "store"
    with LifecycleStore(path, dimension=2, threads=1) as store:
        store.insert(1, [1.0, 0.0])
        store.build_generation("built-with-one-thread")

    observed: list[int] = []
    monkeypatch.setattr(
        "searchability.generations.faiss.omp_set_num_threads", observed.append
    )
    with LifecycleStore(path, threads=3) as reopened:
        with reopened.pin() as view:
            loaded = reopened._generation_base(view)
        assert loaded.threads == 3
        assert observed[-1] == 3


def test_reserved_temporary_generation_prefix_is_rejected(tmp_path: Path) -> None:
    with LifecycleStore(tmp_path / "store", dimension=2) as store:
        store.insert(1, [1.0, 0.0])
        with pytest.raises(ValueError, match="safe path component"):
            store.build_generation(".tmp-shadow")
        finals, temporaries = store._generations.directories()
        assert ".tmp-shadow" not in finals
        assert temporaries == ()


def test_update_delete_use_historical_intervals_not_current_latest(tmp_path: Path) -> None:
    with LifecycleStore(tmp_path / "store", dimension=2) as store:
        inserted = store.insert(42, [1.0, 0.0])
        generation = store.build_generation("at-insert")
        updated = store.update(42, [4.0, 0.0])
        deleted = store.delete(42)

        old = store.search(QUERY, 1, snapshot=inserted)
        middle = store.search(QUERY, 1, snapshot=updated)
        current = store.search(QUERY, 1, snapshot=deleted)
        assert generation == "at-insert"
        assert _keys(old) == ((42, 1),)
        assert _keys(middle) == ((42, 2),)
        assert _keys(current) == ()
        assert old.hits[0].distance == 1.0
        assert middle.hits[0].distance == 4.0
        # A latest-only table would incorrectly make both historical snapshots
        # empty after the delete; the retained intervals are authoritative.
        assert store.visible_keys(inserted) == ((42, 1),)
        assert store.visible_keys(updated) == ((42, 2),)


def test_snapshot_older_than_every_generation_uses_retained_exact_fallback(
    tmp_path: Path,
) -> None:
    with LifecycleStore(tmp_path / "store", dimension=2) as store:
        old_snapshot = store.insert(1, [2.0, 0.0])
        store.insert(2, [1.0, 0.0])
        store.build_generation("coverage-two")
        old = store.search(QUERY, 1, snapshot=old_snapshot)
        assert _keys(old) == ((1, 1),)
        assert old.receipt.base_generation == "retained-exact-fallback"
        assert old.receipt.covered_commit_seq is None
        assert old.receipt.fallback_reason == (
            "missing_compatible_generation_exact_retained_scan"
        )


def test_generation_manifest_and_corrupt_newest_fall_back_safely(tmp_path: Path) -> None:
    path = tmp_path / "store"
    with LifecycleStore(path, dimension=2) as store:
        store.insert(1, [1.0, 0.0])
        store.build_generation("good")
        store.insert(2, [0.5, 0.0])
        store.build_generation("will-corrupt")
        rows = store.generation_rows()
        corrupt_row = next(row for row in rows if row["generation_id"] == "will-corrupt")
        manifest_path = (
            store.generation_directory
            / str(corrupt_row["directory_name"])
            / "manifest.json"
        )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        required = {
            "generation_id",
            "covered_commit_seq",
            "build_snapshot",
            "metric",
            "dimension",
            "dtype",
            "numeric_contract_version",
            "hnsw_parameters",
            "vector_count",
            "ordinal_map_hash",
            "index_hash",
            "metadata_hash",
            "build_config_hash",
            "created_at",
        }
        assert required <= manifest.keys()
        index_path = manifest_path.with_name("index.faiss")
        payload = index_path.read_bytes()
        index_path.write_bytes(payload[:-1] + bytes([payload[-1] ^ 0xFF]))

    with LifecycleStore(path) as reopened:
        report = reopened.last_recovery_report
        assert "will-corrupt" in report.invalid_generations
        assert report.current_generation == "good"
        result = reopened.search(QUERY, 2, snapshot=2)
        assert set(_keys(result)) == {(1, 1), (2, 2)}
        # The older generation plus the durable obligation covers version 2.
        assert result.receipt.base_generation == "good"
        assert result.receipt.raw_pending_scanned == 1


def test_generation_pin_blocks_gc_and_old_view_survives_publish(tmp_path: Path) -> None:
    with LifecycleStore(tmp_path / "store", dimension=2) as store:
        store.insert(1, [1.0, 0.0])
        first = store.build_generation("first")
        with store.pin() as old_view:
            store.insert(2, [0.5, 0.0])
            second = store.build_generation("second")
            with pytest.raises(PinnedGenerationError):
                store.gc()
            old = store.search(QUERY, 1, view=old_view)
            assert old_view.generation_id == first
            assert _keys(old) == ((1, 1),)
        current = store.search(QUERY, 2)
        assert second != first
        assert set(_keys(current)) == {(1, 1), (2, 2)}
        gc_result = store.collect_garbage()
        assert gc_result["status"] == GC_NOT_IMPLEMENTED_CONSERVATIVE_RETENTION
        assert gc_result["versions_retained"] == 2
        assert gc_result["generations_retained"] == 2


def test_released_pin_and_out_of_range_snapshot_are_rejected(tmp_path: Path) -> None:
    with LifecycleStore(tmp_path / "store", dimension=2) as store:
        store.insert(1, [0.0, 0.0])
        with store.pin() as view:
            pass
        with pytest.raises(PinnedViewError):
            store.search(QUERY, 1, view=view)
        with pytest.raises(SnapshotError):
            store.search(QUERY, 1, snapshot=2)

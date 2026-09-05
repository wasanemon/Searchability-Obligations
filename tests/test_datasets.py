from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from searchability.datasets import read_fvecs, synthetic_dataset


@pytest.mark.parametrize(
    "kind", ["clustered", "isotropic", "delta_near_queries", "outlier_radius"]
)
def test_required_synthetic_families_are_deterministic_and_disjoint(kind: str) -> None:
    arguments = {
        "kind": kind,
        "n_base": 20,
        "n_delta": 8,
        "n_validation": 3,
        "n_test": 5,
        "dimension": 6,
        "seed": 417,
        "clusters": 4,
    }
    first = synthetic_dataset(**arguments)
    second = synthetic_dataset(**arguments)

    assert first.sha256() == second.sha256()
    assert first.base.shape == (20, 6)
    assert first.delta.shape == (8, 6)
    assert first.validation_queries.shape == (3, 6)
    assert first.test_queries.shape == (5, 6)
    assert set(first.base_ids).isdisjoint(set(first.delta_ids))
    assert not first.base.flags.writeable
    assert not first.delta.flags.writeable
    assert first.metadata["static_split_is_time_ordered"] is False


def test_dataset_hash_changes_with_seed_even_when_id_ranges_match() -> None:
    common = {
        "kind": "isotropic",
        "n_base": 8,
        "n_delta": 3,
        "n_validation": 2,
        "n_test": 2,
        "dimension": 4,
    }
    first = synthetic_dataset(**common, seed=1)
    second = synthetic_dataset(**common, seed=2)
    assert np.array_equal(first.base_ids, second.base_ids)
    assert first.sha256() != second.sha256()


def test_dataset_hash_accepts_zero_length_partitions() -> None:
    split = synthetic_dataset(
        kind="isotropic",
        n_base=8,
        n_delta=0,
        n_validation=0,
        n_test=0,
        dimension=4,
        seed=3,
    )

    observed = split.sha256()

    assert observed == split.sha256()
    assert len(observed) == 64
    assert split.delta.shape == (0, 4)
    assert split.validation_queries.shape == (0, 4)
    assert split.test_queries.shape == (0, 4)
    assert split.delta_ids.shape == (0,)


def test_fvecs_reader_respects_row_offset_and_limit(tmp_path: Path) -> None:
    vectors = np.asarray(
        [[1.5, -2.0, 3.0], [4.0, 5.25, -6.0], [7.0, 8.0, 9.0]],
        dtype="<f4",
    )
    rows = bytearray()
    for vector in vectors:
        rows.extend(np.asarray([3], dtype="<i4").tobytes())
        rows.extend(vector.tobytes())
    path = tmp_path / "tiny.fvecs"
    path.write_bytes(bytes(rows))

    observed = read_fvecs(path, offset=1, limit=1)
    assert observed.dtype == np.float32
    assert observed.shape == (1, 3)
    assert np.array_equal(observed[0], vectors[1])
    assert not observed.flags.writeable


def test_fvecs_reader_rejects_inconsistent_row_headers(tmp_path: Path) -> None:
    path = tmp_path / "bad.fvecs"
    path.write_bytes(
        np.asarray([2], dtype="<i4").tobytes()
        + np.asarray([1.0, 2.0], dtype="<f4").tobytes()
        + np.asarray([3], dtype="<i4").tobytes()
        + np.asarray([3.0, 4.0], dtype="<f4").tobytes()
    )
    with pytest.raises(ValueError, match="inconsistent"):
        read_fvecs(path)

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from searchability.datasets import (
    load_dataset,
    read_fvecs,
    synthetic_dataset,
    texmex_dataset,
)


def _write_fvecs(path: Path, vectors: np.ndarray) -> None:
    rows = bytearray()
    for vector in np.asarray(vectors, dtype="<f4"):
        rows.extend(np.asarray([vector.shape[0]], dtype="<i4").tobytes())
        rows.extend(vector.tobytes())
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(bytes(rows))


def _tiny_sift_root(tmp_path: Path) -> tuple[Path, np.ndarray]:
    root = tmp_path / "raw"
    base = np.arange(8 * 128, dtype=np.float32).reshape(8, 128)
    queries = (10_000 + np.arange(10 * 128, dtype=np.float32)).reshape(10, 128)
    _write_fvecs(root / "sift" / "sift_base.fvecs", base)
    _write_fvecs(root / "sift" / "sift_query.fvecs", queries)
    manifest = root.parent / "manifests" / "sift.json"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text('{"test_fixture":true}\n', encoding="utf-8")
    return root, queries


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


def test_limited_fvecs_read_does_not_touch_an_out_of_range_payload(
    tmp_path: Path,
) -> None:
    path = tmp_path / "partitioned.fvecs"
    _write_fvecs(path, np.arange(8, dtype=np.float32).reshape(2, 4))
    payload = bytearray(path.read_bytes())
    second_header_offset = 4 * (4 + 1)
    payload[second_header_offset : second_header_offset + 4] = np.asarray(
        [999], dtype="<i4"
    ).tobytes()
    path.write_bytes(payload)

    selected = read_fvecs(path, offset=0, limit=1)

    assert selected.shape == (1, 4)
    with pytest.raises(ValueError, match="inconsistent"):
        read_fvecs(path)


def test_texmex_explicit_query_offsets_are_absolute_and_read_separately(
    tmp_path: Path,
) -> None:
    root, queries = _tiny_sift_root(tmp_path)

    split = texmex_dataset(
        name="sift",
        root=root,
        n_base=4,
        n_delta=2,
        n_validation=2,
        n_test=3,
        validation_query_offset=1,
        test_query_offset=6,
    )

    assert split.validation_query_ids.tolist() == [1, 2]
    assert split.test_query_ids.tolist() == [6, 7, 8]
    assert np.array_equal(split.validation_queries, queries[1:3])
    assert np.array_equal(split.test_queries, queries[6:9])
    assert split.metadata["validation_query_offset"] == 1
    assert split.metadata["test_query_offset"] == 6
    assert split.metadata["query_ranges_disjoint"] is True


def test_texmex_default_offsets_preserve_the_issue_one_prefix_split(
    tmp_path: Path,
) -> None:
    root, _ = _tiny_sift_root(tmp_path)

    split = texmex_dataset(
        name="sift",
        root=root,
        n_base=4,
        n_delta=2,
        n_validation=2,
        n_test=3,
    )

    assert split.validation_query_ids.tolist() == [0, 1]
    assert split.test_query_ids.tolist() == [2, 3, 4]


@pytest.mark.parametrize(
    ("validation_offset", "test_offset", "message"),
    [
        (2, 3, "overlap"),
        (-1, 4, "non-negative"),
    ],
)
def test_texmex_rejects_overlapping_or_negative_query_offsets(
    tmp_path: Path,
    validation_offset: int,
    test_offset: int,
    message: str,
) -> None:
    root, _ = _tiny_sift_root(tmp_path)

    with pytest.raises(ValueError, match=message):
        texmex_dataset(
            name="sift",
            root=root,
            n_base=4,
            n_delta=2,
            n_validation=2,
            n_test=3,
            validation_query_offset=validation_offset,
            test_query_offset=test_offset,
        )


def test_texmex_rejects_query_range_past_the_file(tmp_path: Path) -> None:
    root, _ = _tiny_sift_root(tmp_path)

    with pytest.raises(ValueError, match="requested split sizes"):
        texmex_dataset(
            name="sift",
            root=root,
            n_base=4,
            n_delta=2,
            n_validation=2,
            n_test=3,
            validation_query_offset=0,
            test_query_offset=9,
        )


def test_load_dataset_forwards_explicit_texmex_query_offsets(tmp_path: Path) -> None:
    root, _ = _tiny_sift_root(tmp_path)

    split = load_dataset(
        {
            "type": "texmex",
            "name": "sift",
            "root": str(root),
            "n_base": 4,
            "n_delta": 2,
            "n_validation": 2,
            "n_test": 2,
            "validation_query_offset": 1,
            "test_query_offset": 7,
        }
    )

    assert split.validation_query_ids.tolist() == [1, 2]
    assert split.test_query_ids.tolist() == [7, 8]

from __future__ import annotations

from fractions import Fraction
import math

import numpy as np
import pytest

from searchability.groups import GroupDirectory
from searchability.models import VectorRecord, as_float32_vector
from searchability.numerics import (
    distance_intervals,
    exact_squared_l2 as implementation_squared_l2,
    lower_bound_fraction,
    sqrt_fraction_bracket,
    validate_beta,
    validate_k,
)
from searchability.oracle import (
    exact_group_radius_squared,
    exact_squared_l2,
    radius_covers,
    sqrt_bracket,
)


def _squared_float(value: float) -> Fraction:
    as_fraction = Fraction.from_float(float(value))
    return as_fraction * as_fraction


def _assert_interval_contains_exact(
    query: np.ndarray, vectors: np.ndarray
) -> None:
    intervals = distance_intervals(query, vectors)
    assert intervals.estimate.shape == (len(vectors),)
    assert np.all(intervals.lower >= 0.0)
    assert np.all(intervals.lower <= intervals.upper)
    for index, vector in enumerate(vectors):
        exact = exact_squared_l2(query, vector)
        assert _squared_float(intervals.lower[index]) <= exact
        assert exact <= _squared_float(intervals.upper[index])


def test_distance_intervals_contain_exact_stored_float32_distances() -> None:
    rng = np.random.default_rng(0x1A2B3C4D)
    scales = (0.0, 2.0**-100, 2.0**-20, 1.0, 2.0**20, 1.0e14)
    for dimension in (1, 2, 3, 7, 31, 128, 257):
        for scale in scales:
            query = np.clip(
                rng.normal(size=dimension) * scale, -9.0e14, 9.0e14
            ).astype(np.float32)
            vectors = np.clip(
                rng.normal(size=(9, dimension)) * scale,
                -9.0e14,
                9.0e14,
            ).astype(np.float32)
            vectors[0] = query
            _assert_interval_contains_exact(query, vectors)


def test_intervals_cover_cancellation_outliers_and_long_accumulation() -> None:
    query = np.array(
        [1.0e14, -1.0e14] + [1.0 + (index % 2) * 2.0**-20 for index in range(1022)],
        dtype=np.float32,
    )
    vectors = np.stack(
        [
            query,
            np.nextafter(query, np.float32(math.inf), dtype=np.float32),
            -query,
            np.zeros_like(query),
        ]
    )
    _assert_interval_contains_exact(query, vectors)


@pytest.mark.parametrize(
    "value",
    [
        Fraction(0),
        Fraction(1, 3),
        Fraction(2),
        Fraction(2**100 + 1, 2**70),
        Fraction(17**9, 13**7),
    ],
)
def test_sqrt_brackets_independently_enclose_exact_value(value: Fraction) -> None:
    for lower, upper in (sqrt_bracket(value), sqrt_fraction_bracket(value)):
        assert lower >= 0.0
        assert lower <= upper
        assert _squared_float(lower) <= value <= _squared_float(upper)


def test_implementation_exact_distance_agrees_with_independent_oracle() -> None:
    left = np.array([0.1, -0.0, 2.0**-100, 1.0e14], dtype=np.float32)
    right = np.array([-0.2, 0.0, -2.0**-100, -1.0e14], dtype=np.float32)
    assert implementation_squared_l2(left, right) == exact_squared_l2(left, right)


def test_built_radii_cover_every_member_and_lbs_are_member_lower_bounds() -> None:
    centers = np.array([[0.0, 0.0], [20.0, -10.0], [-5.0, 4.0]], dtype=np.float32)
    vectors = np.array(
        [
            [0.0, 0.0],
            [3.0, 4.0],
            [-1.0, 2.0],
            [20.0, -10.0],
            [23.0, -6.0],
            [-5.0, 4.0],
            [1.0e7, -1.0e7],
        ],
        dtype=np.float32,
    )
    records = tuple(
        VectorRecord(logical_id=index, version_id=0, vector=vector)
        for index, vector in enumerate(vectors)
    )
    directory = GroupDirectory.from_centers(centers, records, threads=1)
    query = np.array([7.5, -3.25], dtype=np.float32)

    for group in directory.groups:
        member_vectors = tuple(record.vector for record in group.members)
        assert radius_covers(group.center, member_vectors, group.radius_upper)
        exact_radius_squared = exact_group_radius_squared(group.center, member_vectors)
        assert exact_radius_squared <= _squared_float(group.radius_upper)

        lb, _, _ = lower_bound_fraction(query, group.center, group.radius_upper)
        assert lb >= 0
        for record in group.members:
            member_squared = exact_squared_l2(query, record.vector)
            assert lb == 0 or lb * lb <= member_squared


def test_zero_radius_and_zero_lower_bound() -> None:
    point = np.array([2.0, -3.0, 5.0], dtype=np.float32)
    assert exact_group_radius_squared(point, [point, point]) == 0
    assert radius_covers(point, [point], 0.0)
    lower, center_lower, center_upper = lower_bound_fraction(point, point, 0.0)
    assert lower == 0
    assert center_lower == center_upper == 0.0


def test_independent_radius_check_rejects_too_small_radius() -> None:
    center = np.array([0.0, 0.0], dtype=np.float32)
    member = np.array([3.0, 4.0], dtype=np.float32)
    assert not radius_covers(center, [member], math.nextafter(5.0, 0.0))
    assert radius_covers(center, [member], 5.0)


@pytest.mark.parametrize("beta", [-1.0, -math.inf, math.inf, math.nan])
def test_invalid_beta_is_rejected(beta: float) -> None:
    with pytest.raises(ValueError):
        validate_beta(beta)


@pytest.mark.parametrize("k", [0, -1])
def test_nonpositive_k_is_rejected(k: int) -> None:
    with pytest.raises(ValueError):
        validate_k(k)


@pytest.mark.parametrize("k", [True, False, 1.0, "1", None])
def test_noninteger_k_is_rejected(k: object) -> None:
    with pytest.raises(TypeError):
        validate_k(k)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "bad_vector",
    [
        np.array([math.nan], dtype=np.float64),
        np.array([math.inf], dtype=np.float64),
        np.array([-math.inf], dtype=np.float64),
        np.array([1.0e16], dtype=np.float64),
    ],
)
def test_invalid_numeric_vectors_are_rejected(bad_vector: np.ndarray) -> None:
    with pytest.raises(ValueError):
        as_float32_vector(bad_vector)
    with pytest.raises(ValueError):
        distance_intervals(bad_vector, np.zeros((1, 1), dtype=np.float32))


def test_complex_vectors_are_rejected_instead_of_discarding_the_imaginary_part() -> None:
    complex_vector = np.array([1.0 + 2.0j], dtype=np.complex64)
    with pytest.raises(TypeError):
        as_float32_vector(complex_vector)
    with pytest.raises(TypeError):
        exact_squared_l2(complex_vector, np.zeros(1, dtype=np.float32))


def test_distance_input_shape_and_radius_validation() -> None:
    query = np.zeros(2, dtype=np.float32)
    with pytest.raises(ValueError, match="two-dimensional"):
        distance_intervals(query, np.zeros(2, dtype=np.float32))
    with pytest.raises(ValueError, match="dimension mismatch"):
        distance_intervals(query, np.zeros((1, 3), dtype=np.float32))
    with pytest.raises(ValueError, match="NaN or infinity"):
        distance_intervals(query, np.array([[math.nan, 0.0]], dtype=np.float64))
    for radius in (-1.0, math.nan, math.inf):
        with pytest.raises(ValueError):
            lower_bound_fraction(query, query, radius)


def test_exact_distance_rejects_mismatched_dimensions() -> None:
    with pytest.raises(ValueError, match="dimension mismatch"):
        exact_squared_l2(
            np.zeros(2, dtype=np.float32), np.zeros(3, dtype=np.float32)
        )

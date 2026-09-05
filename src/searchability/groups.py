"""Immutable fixed-center Delta groups and raw-pending ownership."""

from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction
import time
from typing import Sequence

import faiss
import numpy as np

from .models import VectorRecord, as_float32_matrix, as_float32_vector
from .numerics import distance_intervals, exact_squared_l2


@dataclass(frozen=True, slots=True)
class Group:
    group_id: int
    center: np.ndarray = field(compare=False, repr=False)
    radius_upper: float
    members: tuple[VectorRecord, ...]
    vectors: np.ndarray = field(compare=False, repr=False)

    def __post_init__(self) -> None:
        if isinstance(self.group_id, bool) or not isinstance(
            self.group_id, (int, np.integer)
        ):
            raise TypeError("group_id must be an integer")
        if int(self.group_id) < 0:
            raise ValueError("group_id must be non-negative")
        center = as_float32_vector(self.center, name="center")
        members = tuple(self.members)
        if any(not isinstance(member, VectorRecord) for member in members):
            raise TypeError("group members must be VectorRecord instances")
        if len({member.key for member in members}) != len(members):
            raise ValueError("group contains duplicate version keys")
        matrix = as_float32_matrix(self.vectors, name="group vectors")
        if matrix.shape != (len(members), center.shape[0]):
            raise ValueError("group member matrix shape mismatch")
        if not np.isfinite(self.radius_upper) or self.radius_upper < 0.0:
            raise ValueError("radius_upper must be finite and non-negative")
        expected = (
            np.ascontiguousarray(
                np.stack([member.vector for member in members]), dtype=np.float32
            )
            if members
            else np.empty(matrix.shape, dtype=np.float32)
        )
        if matrix.tobytes(order="C") != expected.tobytes(order="C"):
            raise ValueError("group vectors do not bit-match group members")
        if members:
            interval_required = float(distance_intervals(center, matrix).upper.max())
            if self.radius_upper < interval_required:
                # A mathematically valid tight radius can fall inside the bulk
                # interval.  Resolve only this construction-time ambiguity
                # exactly; an actually under-covering/corrupt group is rejected.
                radius_squared = Fraction.from_float(float(self.radius_upper)) ** 2
                for member_vector in matrix:
                    if exact_squared_l2(center, member_vector) > radius_squared:
                        raise ValueError("radius_upper does not cover every group member")
        object.__setattr__(self, "group_id", int(self.group_id))
        object.__setattr__(self, "center", center)
        object.__setattr__(self, "members", members)
        object.__setattr__(self, "vectors", matrix)

    def visible(self, snapshot_id: int) -> tuple[tuple[VectorRecord, ...], np.ndarray]:
        indices = [
            index for index, record in enumerate(self.members) if record.visible_at(snapshot_id)
        ]
        if not indices:
            return (), as_float32_matrix(
                np.empty((0, self.center.shape[0]), dtype=np.float32),
                name="visible group vectors",
            )
        return (
            tuple(self.members[index] for index in indices),
            as_float32_matrix(
                self.vectors[indices], name="visible group vectors"
            ),
        )


@dataclass(frozen=True, slots=True)
class GroupBuildStats:
    center_training_ns: int
    assignment_ns: int
    packing_and_radius_ns: int
    member_bytes: int
    metadata_bytes: int


@dataclass(frozen=True, slots=True)
class GroupDirectory:
    groups: tuple[Group, ...]
    dimension: int
    build_stats: GroupBuildStats

    def __post_init__(self) -> None:
        if isinstance(self.dimension, bool) or not isinstance(
            self.dimension, (int, np.integer)
        ):
            raise TypeError("group directory dimension must be an integer")
        if int(self.dimension) <= 0:
            raise ValueError("group directory dimension must be positive")
        groups = tuple(self.groups)
        if any(not isinstance(group, Group) for group in groups):
            raise TypeError("groups must be Group instances")
        identifiers = [group.group_id for group in groups]
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("group directory contains duplicate group IDs")
        if any(group.center.shape[0] != int(self.dimension) for group in groups):
            raise ValueError("group/directory dimension mismatch")
        keys = [member.key for group in groups for member in group.members]
        if len(set(keys)) != len(keys):
            raise ValueError("version key is assigned to more than one group")
        object.__setattr__(self, "groups", groups)
        object.__setattr__(self, "dimension", int(self.dimension))

    @staticmethod
    def train_centers(
        training_vectors: np.ndarray,
        n_groups: int,
        *,
        seed: int = 0,
        iterations: int = 20,
        threads: int = 1,
    ) -> tuple[np.ndarray, int]:
        training = as_float32_matrix(training_vectors, name="training_vectors")
        if training.shape[0] == 0:
            raise ValueError("training_vectors must be a non-empty matrix")
        if n_groups <= 0 or n_groups > training.shape[0]:
            raise ValueError("n_groups must be in [1, number of training vectors]")
        if iterations <= 0:
            raise ValueError("iterations must be positive")
        faiss.omp_set_num_threads(threads)
        start = time.perf_counter_ns()
        kmeans = faiss.Kmeans(
            training.shape[1],
            int(n_groups),
            niter=int(iterations),
            verbose=False,
            seed=int(seed),
            min_points_per_centroid=1,
            max_points_per_centroid=max(256, training.shape[0]),
        )
        kmeans.train(training)
        elapsed = time.perf_counter_ns() - start
        return as_float32_matrix(kmeans.centroids, name="trained centers"), elapsed

    @classmethod
    def from_centers(
        cls,
        centers: np.ndarray,
        records: Sequence[VectorRecord],
        *,
        center_training_ns: int = 0,
        threads: int = 1,
    ) -> "GroupDirectory":
        center_matrix = as_float32_matrix(centers, name="centers")
        if center_matrix.shape[0] == 0:
            raise ValueError("centers must be a non-empty matrix")
        if any(record.vector.shape[0] != center_matrix.shape[1] for record in records):
            raise ValueError("record/center dimension mismatch")
        member_matrix = (
            np.ascontiguousarray(np.stack([record.vector for record in records]), dtype=np.float32)
            if records
            else np.empty((0, center_matrix.shape[1]), dtype=np.float32)
        )
        faiss.omp_set_num_threads(threads)
        assignment_start = time.perf_counter_ns()
        assignments = np.empty(0, dtype=np.int64)
        if records:
            center_index = faiss.IndexFlatL2(center_matrix.shape[1])
            center_index.add(center_matrix)
            _, nearest = center_index.search(member_matrix, 1)
            assignments = nearest[:, 0].astype(np.int64, copy=False)
        assignment_ns = time.perf_counter_ns() - assignment_start

        packing_start = time.perf_counter_ns()
        groups: list[Group] = []
        for group_id, center in enumerate(center_matrix):
            indices = np.flatnonzero(assignments == group_id)
            group_records = tuple(records[index] for index in indices.tolist())
            vectors = (
                np.ascontiguousarray(member_matrix[indices], dtype=np.float32)
                if indices.size
                else np.empty((0, center_matrix.shape[1]), dtype=np.float32)
            )
            radius_upper = 0.0
            if indices.size:
                radius_upper = float(distance_intervals(center, vectors).upper.max())
            groups.append(
                Group(
                    group_id=group_id,
                    center=center,
                    radius_upper=radius_upper,
                    members=group_records,
                    vectors=vectors,
                )
            )
        packing_ns = time.perf_counter_ns() - packing_start
        metadata_bytes = len(groups) * (center_matrix.shape[1] * 4 + 8 + 8)
        return cls(
            groups=tuple(groups),
            dimension=center_matrix.shape[1],
            build_stats=GroupBuildStats(
                center_training_ns=int(center_training_ns),
                assignment_ns=int(assignment_ns),
                packing_and_radius_ns=int(packing_ns),
                member_bytes=int(member_matrix.nbytes),
                metadata_bytes=int(metadata_bytes),
            ),
        )

    @classmethod
    def build(
        cls,
        training_vectors: np.ndarray,
        records: Sequence[VectorRecord],
        n_groups: int,
        *,
        seed: int = 0,
        iterations: int = 20,
        threads: int = 1,
    ) -> "GroupDirectory":
        centers, training_ns = cls.train_centers(
            training_vectors,
            n_groups,
            seed=seed,
            iterations=iterations,
            threads=threads,
        )
        return cls.from_centers(
            centers, records, center_training_ns=training_ns, threads=threads
        )


@dataclass(frozen=True, slots=True)
class DeltaStore:
    """A snapshot-safe immutable view of raw and grouped pending records."""

    raw_records: tuple[VectorRecord, ...]
    grouped: GroupDirectory | None = None

    def __post_init__(self) -> None:
        raw = tuple(self.raw_records)
        if any(not isinstance(record, VectorRecord) for record in raw):
            raise TypeError("raw_records must contain VectorRecord instances")
        if self.grouped is not None and not isinstance(self.grouped, GroupDirectory):
            raise TypeError("grouped must be a GroupDirectory or None")
        dimensions = {record.vector.shape[0] for record in raw}
        if self.grouped is not None:
            dimensions.add(self.grouped.dimension)
        if len(dimensions) > 1:
            raise ValueError("Delta records/groups have inconsistent dimensions")

        # Raw/group overlap is expected during an atomic handoff and is handled
        # conservatively by SearchEngine's duplicate-visible fallback.  The
        # same version key must nevertheless describe the same stored value;
        # otherwise full and grouped scans could rank different mathematics.
        by_key: dict[tuple[int, int], VectorRecord] = {}
        all_records = list(raw)
        if self.grouped is not None:
            all_records.extend(
                member
                for group in self.grouped.groups
                for member in group.members
            )
        for record in all_records:
            previous = by_key.get(record.key)
            if previous is None:
                by_key[record.key] = record
                continue
            if (
                previous.begin_seq != record.begin_seq
                or previous.end_seq != record.end_seq
                or previous.commit_seq != record.commit_seq
                or previous.vector.tobytes(order="C")
                != record.vector.tobytes(order="C")
            ):
                raise ValueError(
                    "duplicate Delta version key has conflicting stored content"
                )
        object.__setattr__(self, "raw_records", raw)

    @property
    def dimension(self) -> int | None:
        if self.raw_records:
            return self.raw_records[0].vector.shape[0]
        if self.grouped is not None:
            return self.grouped.dimension
        return None

    def group_all(
        self,
        training_vectors: np.ndarray,
        n_groups: int,
        *,
        seed: int = 0,
        iterations: int = 20,
        threads: int = 1,
    ) -> "DeltaStore":
        """Build completely before atomically returning the new immutable view."""

        prior_members: list[VectorRecord] = []
        if self.grouped is not None:
            for group in self.grouped.groups:
                prior_members.extend(group.members)
        all_records = tuple(prior_members) + self.raw_records
        directory = GroupDirectory.build(
            training_vectors,
            all_records,
            n_groups,
            seed=seed,
            iterations=iterations,
            threads=threads,
        )
        return DeltaStore(raw_records=(), grouped=directory)

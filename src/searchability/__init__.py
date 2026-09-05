"""Searchability Obligations research prototype."""

from .base import BaseIndex
from .groups import DeltaStore, Group, GroupDirectory
from .models import CandidateSet, Receipt, SearchHit, SearchResult, VectorRecord
from .store import (
    CommitOperation,
    LifecycleStore,
    PinnedGenerationError,
    PinnedView,
    RecoveryReport,
    SnapshotError,
)
from .search import SearchEngine

__all__ = [
    "BaseIndex",
    "CandidateSet",
    "CommitOperation",
    "DeltaStore",
    "Group",
    "GroupDirectory",
    "LifecycleStore",
    "PinnedGenerationError",
    "PinnedView",
    "RecoveryReport",
    "Receipt",
    "SearchEngine",
    "SearchHit",
    "SearchResult",
    "SnapshotError",
    "VectorRecord",
]

__version__ = "0.1.0"

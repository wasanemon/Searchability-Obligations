#!/usr/bin/env python3
"""Emit the exact runtime/CPU/library/thread environment as JSON."""

from __future__ import annotations

import argparse
import json
import os
import platform
from pathlib import Path
import subprocess
import sys

import faiss
import numpy as np


def _command(command: list[str]) -> str | None:
    try:
        return subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def _faiss_hnsw_smoke() -> dict[str, object]:
    """Execute a real one-thread add/search, not merely an import check."""

    vectors = np.asarray([[0.0, 0.0], [1.0, 0.0], [3.0, 0.0]], dtype=np.float32)
    query = np.asarray([[0.0, 0.0]], dtype=np.float32)
    index = faiss.IndexHNSWFlat(2, 8, faiss.METRIC_L2)
    index.hnsw.efConstruction = 32
    index.hnsw.efSearch = 16
    index.add(vectors)
    squared_l2, labels = index.search(query, 2)
    passed = labels[0].tolist() == [0, 1] and np.allclose(
        squared_l2[0], np.asarray([0.0, 1.0], dtype=np.float32)
    )
    if not passed:
        raise RuntimeError(
            "Faiss HNSW add/search smoke returned unexpected labels or squared-L2"
        )
    return {
        "passed": True,
        "metric": "Faiss squared-L2",
        "labels": labels[0].tolist(),
        "squared_l2": squared_l2[0].tolist(),
        "vectors": int(index.ntotal),
    }


def collect() -> dict[str, object]:
    faiss.omp_set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "1")))
    memory = {}
    meminfo = Path("/proc/meminfo")
    if meminfo.exists():
        for line in meminfo.read_text(encoding="utf-8").splitlines():
            key, value = line.split(":", 1)
            if key in {"MemTotal", "MemAvailable", "SwapTotal", "SwapFree"}:
                memory[key] = value.strip()
    return {
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "faiss_version": faiss.__version__,
        "faiss_compile_options": faiss.get_compile_options(),
        "faiss_threads": faiss.omp_get_max_threads(),
        "faiss_hnsw_smoke": _faiss_hnsw_smoke(),
        "numpy_version": np.__version__,
        "thread_environment": {
            name: os.environ.get(name)
            for name in (
                "OMP_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS",
            )
        },
        "cpu_count": os.cpu_count(),
        "lscpu": _command(["lscpu"]),
        "memory": memory,
        "git_commit": _command(["git", "rev-parse", "HEAD"]),
        "git_status_porcelain": _command(["git", "status", "--porcelain"]),
        "pip_freeze": _command([sys.executable, "-m", "pip", "freeze"]),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    payload = json.dumps(collect(), indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    else:
        print(payload, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

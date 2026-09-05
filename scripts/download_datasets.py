#!/usr/bin/env python3
"""Resumable TEXMEX downloads with checksums and safe extraction."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import tarfile
import traceback
import urllib.request

from searchability.artifacts import atomic_write_json, file_sha256


def _file_md5(path: Path) -> str:
    digest = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _download(url: str, archive: Path) -> dict[str, object]:
    archive.parent.mkdir(parents=True, exist_ok=True)
    partial = archive.with_suffix(archive.suffix + ".part")
    existing = partial.stat().st_size if partial.exists() else 0
    headers = {"User-Agent": "Searchability-Obligations-Issue-1/0.1"}
    if existing:
        headers["Range"] = f"bytes={existing}-"
    request = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(request, timeout=120) as response:
        status = getattr(response, "status", response.getcode())
        content_range = response.headers.get("Content-Range")
        append = existing > 0 and status == 206 and content_range is not None
        mode = "ab" if append else "wb"
        if existing and not append:
            existing = 0
        with partial.open(mode) as handle:
            while True:
                block = response.read(8 * 1024 * 1024)
                if not block:
                    break
                handle.write(block)
            handle.flush()
            os.fsync(handle.fileno())
    os.replace(partial, archive)
    return {
        "http_status": status,
        "content_range": content_range,
        "resumed_from_bytes": existing,
        "downloaded_size": archive.stat().st_size,
    }


def _safe_extract(archive: Path, destination: Path) -> list[str]:
    destination.mkdir(parents=True, exist_ok=True)
    root = destination.resolve()
    extracted: list[str] = []
    with tarfile.open(archive, "r:*") as bundle:
        for member in bundle.getmembers():
            target = (destination / member.name).resolve()
            if target != root and root not in target.parents:
                raise ValueError(f"archive path escapes destination: {member.name}")
            if member.issym() or member.islnk():
                raise ValueError(f"archive links are not accepted: {member.name}")
            if not (member.isfile() or member.isdir()):
                raise ValueError(
                    f"archive special entries are not accepted: {member.name}"
                )
        bundle.extractall(destination)
        extracted = [member.name for member in bundle.getmembers() if member.isfile()]
    return extracted


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--only", action="append", default=[])
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    root = Path(config.get("root", "data/raw"))
    manifest_dir = Path(config.get("manifest_dir", "data/manifests"))
    manifest_dir.mkdir(parents=True, exist_ok=True)
    failures: list[dict[str, object]] = []

    for dataset in config["datasets"]:
        name = str(dataset["name"])
        if args.only and name not in args.only:
            continue
        archive = root / str(dataset["archive"])
        manifest_path = manifest_dir / f"{name}.json"
        try:
            transfer = (
                {"reused_existing_archive": True, "downloaded_size": archive.stat().st_size}
                if archive.exists()
                else _download(str(dataset["url"]), archive)
            )
            checksum = file_sha256(archive)
            md5 = _file_md5(archive)
            expected = dataset.get("sha256")
            if expected and checksum.lower() != str(expected).lower():
                raise ValueError(
                    f"checksum mismatch for {name}: expected {expected}, got {checksum}"
                )
            expected_md5 = dataset.get("md5")
            if expected_md5 and md5.lower() != str(expected_md5).lower():
                raise ValueError(
                    f"publisher MD5 mismatch for {name}: expected {expected_md5}, got {md5}"
                )
            previous = None
            if manifest_path.exists():
                previous = json.loads(manifest_path.read_text(encoding="utf-8"))
                observed = previous.get("observed_sha256")
                if observed and observed != checksum:
                    raise ValueError(
                        f"archive changed since prior manifest: {observed} != {checksum}"
                    )
            extracted = _safe_extract(archive, root)
            file_manifest = []
            for relative in sorted(extracted):
                location = root / relative
                file_manifest.append(
                    {
                        "path": relative,
                        "bytes": location.stat().st_size,
                        "sha256": file_sha256(location),
                    }
                )
            atomic_write_json(
                manifest_path,
                {
                    "schema_version": 1,
                    "name": name,
                    "url": dataset["url"],
                    "source_page": dataset["source_page"],
                    "terms_note": dataset["terms_note"],
                    "archive": str(archive),
                    "observed_sha256": checksum,
                    "pinned_sha256": expected,
                    "sha256_source": dataset.get("sha256_source"),
                    "publisher_sha256": dataset.get("publisher_sha256"),
                    "observed_md5": md5,
                    "publisher_md5": expected_md5,
                    "retrieved_at": datetime.now(timezone.utc).isoformat(),
                    "transfer": transfer,
                    "files": file_manifest,
                    "previous_manifest_present": previous is not None,
                },
            )
            print(
                f"{name}: ready (sha256={checksum}, md5={md5}, "
                f"{archive.stat().st_size} bytes)"
            )
        except BaseException as error:
            failure = {
                "name": name,
                "url": dataset.get("url"),
                "time": datetime.now(timezone.utc).isoformat(),
                "error_type": type(error).__name__,
                "error": str(error),
                "traceback": traceback.format_exc(),
                "resume_command": (
                    f".venv/bin/python scripts/download_datasets.py --config "
                    f"{args.config} --only {name}"
                ),
            }
            failures.append(failure)
            print(f"{name}: FAILED: {error}")
    atomic_write_json(manifest_dir / "last_attempt.json", {"failures": failures})
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Parquet storage with a content hash and a metadata JSON sidecar."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd


def data_hash(df: pd.DataFrame) -> str:
    """Deterministic sha256 of values and index (timestamps normalized to nanoseconds)."""
    h = hashlib.sha256()
    h.update(df.index.as_unit("ns").asi8.tobytes())
    h.update(df.to_numpy(dtype="float64").tobytes())
    h.update(",".join(map(str, df.columns)).encode())
    return h.hexdigest()


def save_dataset(df: pd.DataFrame, name: str, directory: str | Path, **meta) -> dict:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    df.to_parquet(directory / f"{name}.parquet")
    info = {
        "name": name,
        "rows": len(df),
        "first": str(df.index.min()),
        "last": str(df.index.max()),
        "sha256": data_hash(df),
        "created_utc": datetime.now(UTC).isoformat(),
        **meta,
    }
    (directory / f"{name}.meta.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    return info


def load_dataset(
    name: str, directory: str | Path, verify: bool = True
) -> tuple[pd.DataFrame, dict]:
    directory = Path(directory)
    df = pd.read_parquet(directory / f"{name}.parquet")
    info = json.loads((directory / f"{name}.meta.json").read_text(encoding="utf-8"))
    if verify and data_hash(df) != info["sha256"]:
        raise ValueError(f"hash mismatch for dataset '{name}': file changed since it was saved")
    return df, info

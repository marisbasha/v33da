#!/usr/bin/env python3
"""Prepare standalone v33da experiment split shards from the pooled dataset."""

from __future__ import annotations

import json
import os
import shutil
from collections import Counter
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parent.parent
DATASET_ROOT = Path(os.environ.get("V33DA_DATA_DIR", ROOT / "data" / "v33da"))
SPLIT_ROOT = Path(os.environ.get("V33DA_SPLIT_DIR", ROOT / "splits"))
ROWS_PER_SHARD = 1500

SESSION_SPLIT_BY_EXP_DATE = {
    ("juvExpBP01", "2021-07-31"): "train",
    ("juvExpBP01", "2021-08-01"): "train",
    ("juvExpBP01", "2021-08-02"): "train",
    ("juvExpBP01", "2021-08-03"): "train",
    ("juvExpBP01", "2021-08-04"): "train",
    ("juvExpBP01", "2021-08-05"): "train",
    ("juvExpBP01", "2021-08-06"): "train",
    ("juvExpBP01", "2021-08-07"): "val",
    ("juvExpBP01", "2021-08-08"): "test",
    ("juvExpBP01", "2021-08-09"): "test",
    ("juvExpBP02", "2022-05-24"): "train",
    ("juvExpBP02", "2022-05-25"): "val",
    ("juvExpBP02", "2022-05-26"): "test",
    ("juvExpBP05", "2023-07-10"): "train",
    ("juvExpBP05", "2023-07-11"): "val",
    ("juvExpBP05", "2023-07-12"): "test",
}

GENERALIZATION_SPLIT_BY_EXP_DATE = {
    ("juvExpBP01", "2021-07-31"): "train",
    ("juvExpBP01", "2021-08-01"): "train",
    ("juvExpBP01", "2021-08-02"): "train",
    ("juvExpBP01", "2021-08-03"): "train",
    ("juvExpBP01", "2021-08-04"): "train",
    ("juvExpBP01", "2021-08-05"): "train",
    ("juvExpBP01", "2021-08-06"): "train",
    ("juvExpBP01", "2021-08-07"): "val",
    ("juvExpBP01", "2021-08-08"): "val",
    ("juvExpBP01", "2021-08-09"): "val",
    ("juvExpBP02", "2022-05-24"): "train",
    ("juvExpBP02", "2022-05-25"): "train",
    ("juvExpBP02", "2022-05-26"): "val",
    ("juvExpBP05", "2023-07-10"): "test",
    ("juvExpBP05", "2023-07-11"): "test",
    ("juvExpBP05", "2023-07-12"): "test",
}


def load_pooled_table() -> pa.Table:
    shard_files = sorted(DATASET_ROOT.glob("v33da-*.parquet"))
    if not shard_files:
        raise FileNotFoundError(f"No pooled shards found in {DATASET_ROOT}")
    return pa.concat_tables([pq.read_table(path) for path in shard_files], promote_options="default")


def assign_splits(table: pa.Table) -> tuple[list[str], list[str]]:
    session_splits: list[str] = []
    generalization_splits: list[str] = []
    missing_keys = set()
    for exp, date in zip(table.column("experiment").to_pylist(), table.column("date").to_pylist()):
        key = (exp, date)
        session = SESSION_SPLIT_BY_EXP_DATE.get(key)
        generalization = GENERALIZATION_SPLIT_BY_EXP_DATE.get(key)
        if session is None or generalization is None:
            missing_keys.add(key)
            continue
        session_splits.append(session)
        generalization_splits.append(generalization)
    if missing_keys:
        missing = ", ".join(f"{exp}|{date}" for exp, date in sorted(missing_keys))
        raise ValueError(f"Missing split assignments for: {missing}")
    return session_splits, generalization_splits


def reset_split_dirs() -> None:
    for split_name in ("train", "val", "test"):
        split_dir = SPLIT_ROOT / split_name
        if split_dir.exists():
            shutil.rmtree(split_dir)
        split_dir.mkdir(parents=True, exist_ok=True)


def write_split_shards(table: pa.Table, split_name: str) -> None:
    split_dir = SPLIT_ROOT / split_name
    for shard_idx, start in enumerate(range(0, table.num_rows, ROWS_PER_SHARD)):
        shard = table.slice(start, ROWS_PER_SHARD)
        pq.write_table(shard, split_dir / f"{split_name}-{shard_idx:05d}.parquet")


def main() -> None:
    table = load_pooled_table()
    session_splits, generalization_splits = assign_splits(table)
    enriched = (
        table.append_column("split", pa.array(session_splits))
        .append_column("split_generalization", pa.array(generalization_splits))
    )

    reset_split_dirs()

    session_counts = Counter(session_splits)
    generalization_counts = Counter(generalization_splits)
    session_by_experiment: dict[str, dict[str, int]] = {}
    generalization_by_experiment: dict[str, dict[str, int]] = {}

    for split_name in ("train", "val", "test"):
        session_mask = pa.array([split_name == s for s in session_splits])
        split_table = enriched.filter(session_mask)
        write_split_shards(split_table, split_name)
        session_by_experiment[split_name] = dict(
            sorted(Counter(split_table.column("experiment").to_pylist()).items())
        )
        generalization_by_experiment[split_name] = dict(
            sorted(
                Counter(
                    exp
                    for exp, gen in zip(table.column("experiment").to_pylist(), generalization_splits)
                    if gen == split_name
                ).items()
            )
        )
        print(f"{split_name}: {split_table.num_rows} rows", flush=True)

    summary = {
        "dataset_root": str(DATASET_ROOT),
        "split_root": str(SPLIT_ROOT),
        "total_rows": enriched.num_rows,
        "session_split_counts": dict(sorted(session_counts.items())),
        "generalization_split_counts": dict(sorted(generalization_counts.items())),
        "session_by_experiment": session_by_experiment,
        "generalization_by_experiment": generalization_by_experiment,
    }
    SPLIT_ROOT.mkdir(parents=True, exist_ok=True)
    summary_path = SPLIT_ROOT / "split_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"wrote {summary_path}", flush=True)


if __name__ == "__main__":
    main()

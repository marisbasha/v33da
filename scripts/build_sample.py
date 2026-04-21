"""Build a tiny self-contained V33DA sample for reviewers.

Selects N events stratified across the three experiments, copies only the
parquet rows and per-event asset files those events reference, and mirrors
the calibrations folder so the standard loaders work without modification.

Run from the repo root:
    python scripts/build_sample.py \
        --src /path/to/full/v33da \
        --dst data/v33da_sample \
        --n 100

The source must be a local V33DA layout (parquet shards + audio/ +
accelerometer/ + calibrations/). Download it with:

    huggingface-cli download songbirdini/V33DA --repo-type dataset \
        --local-dir /tmp/v33da_full
"""

from __future__ import annotations

import argparse
import shutil
from collections import Counter
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", type=Path, required=True, help="Full V33DA root")
    ap.add_argument("--dst", type=Path, required=True, help="Sample root to create")
    ap.add_argument("--n", type=int, default=100, help="Total events")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--include-video", action="store_true",
                    help="Also copy clips/*.mp4 (adds ~7x size)")
    return ap.parse_args()


def select_rows(src: Path, n_total: int, seed: int) -> pa.Table:
    shards = sorted(src.glob("v33da-*.parquet"))
    if not shards:
        raise SystemExit(f"No parquet shards under {src}")
    full = pa.concat_tables([pq.read_table(s) for s in shards])
    experiments = sorted(set(full["experiment"].to_pylist()))
    per_exp = n_total // len(experiments)
    remainder = n_total - per_exp * len(experiments)

    selected = []
    for i, exp in enumerate(experiments):
        mask = pa.compute.equal(full["experiment"], exp)
        sub = full.filter(mask)
        k = per_exp + (1 if i < remainder else 0)
        k = min(k, sub.num_rows)
        # Deterministic selection: evenly spaced indices through the sorted shard order.
        idx = [round(j * (sub.num_rows - 1) / max(k - 1, 1)) for j in range(k)]
        selected.append(sub.take(idx))
    return pa.concat_tables(selected)


def copy_assets(rows: pa.Table, src: Path, dst: Path, include_video: bool) -> Counter:
    counts: Counter = Counter()
    for col in ("audio_path", "accelerometer_path", "video_path"):
        if col == "video_path" and not include_video:
            continue
        for rel in rows[col].to_pylist():
            if not rel:
                continue
            src_file = src / rel
            dst_file = dst / rel
            if not src_file.exists():
                counts[f"missing_{col}"] += 1
                continue
            dst_file.parent.mkdir(parents=True, exist_ok=True)
            if not dst_file.exists():
                shutil.copy2(src_file, dst_file)
                counts[f"copied_{col}"] += 1
    return counts


def main() -> None:
    args = parse_args()
    args.dst.mkdir(parents=True, exist_ok=True)

    rows = select_rows(args.src, args.n, args.seed)
    print(f"Selected {rows.num_rows} rows across experiments:")
    for exp, count in Counter(rows["experiment"].to_pylist()).items():
        print(f"  {exp}: {count}")

    out_parquet = args.dst / "v33da_sample.parquet"
    pq.write_table(rows, out_parquet)
    print(f"Wrote {out_parquet}")

    stats = copy_assets(rows, args.src, args.dst, args.include_video)
    for key, value in stats.items():
        print(f"  {key}: {value}")

    calib_src = args.src / "calibrations"
    if calib_src.exists():
        shutil.copytree(calib_src, args.dst / "calibrations", dirs_exist_ok=True)
        print("Copied calibrations/")

    for doc in ("README.md", "metadata.json", "split_summary.json"):
        src_doc = args.src / doc
        if src_doc.exists():
            shutil.copy2(src_doc, args.dst / doc)

    total_bytes = sum(p.stat().st_size for p in args.dst.rglob("*") if p.is_file())
    print(f"Sample size: {total_bytes / 1024 / 1024:.1f} MB")


if __name__ == "__main__":
    main()

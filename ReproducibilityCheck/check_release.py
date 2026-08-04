#!/usr/bin/env python3
"""Check a released V33DA build against its own metadata, and against a re-run
of the real selection stage.

Two independent checks:

  release   Does the published dataset agree with its own metadata.json and
            split_summary.json, and is there one audio, accelerometer and clip
            file per event? Needs only the dataset.

  selection Re-run build_v33da.py --dry-run and compare its per-group counts to
            the release. This runs the project's OWN selection code, so it
            applies the same filters the real build does: overlap removal at
            +/-10ms, motion filter, NaN keypoints, missing tracking payloads,
            missing frame keypoints. Skipped if the builder is not available.

The selection check does not expect equality. --dry-run stops before media
extraction and the published build keeps only samples whose media extracted, so
the release is legitimately smaller. What the check looks at is the shape of the
difference: a broadly uniform shortfall is expected, a group the release has
MORE of is impossible, and two groups whose magnitudes invert mean the colour
labels are transposed.

Usage:
    python ReproducibilityCheck/check_release.py --dataset data/v33da
    python ReproducibilityCheck/check_release.py --dataset data/v33da \\
        --builder ../v33da/build_dataset/build_v33da.py --out report.json
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

# A group falling short of the selection by more than this is worth a look.
MAX_SHORTFALL = 0.40
# Two groups whose ratio inverts by more than this are treated as transposed.
SWAP_RATIO = 10.0

# The juvExpBP05 detector reports yellow and brown swapped relative to the
# physical backpack tags, so build_v33da.py remaps them for the PUBLISHED
# colour while keeping detector convention internally (PUBLISHED_COLOR_REMAP in
# build_dataset/build_v33da.py). metadata.json reports published colours; the
# builder's dry-run prints detector ones. Apply the same remap before compare.
PUBLISHED_COLOR_REMAP = {
    "juvExpBP05": {"yellow": "brown", "brown": "yellow"},
}


def published_bird_color(experiment: str, bird_color: str) -> str:
    return PUBLISHED_COLOR_REMAP.get(experiment, {}).get(bird_color, bird_color)


def check_release(root: Path) -> dict:
    out = {"name": "release", "problems": []}
    meta = json.loads((root / "metadata.json").read_text())
    out["metadata_num_rows"] = meta["num_rows"]

    shards = sorted(root.glob("v33da-*.parquet"))
    out["shards"] = len(shards)
    if not shards:
        out["problems"].append("no parquet shards found")
        out["status"] = "FAIL"
        return out

    try:
        import pyarrow.parquet as pq
    except ImportError:
        out["status"] = "SKIP"
        out["reason"] = "pyarrow not installed"
        return out

    rows = sum(pq.ParquetFile(s).metadata.num_rows for s in shards)
    out["parquet_rows"] = rows
    if rows != meta["num_rows"]:
        out["problems"].append(
            f"parquet holds {rows} rows, metadata.json claims {meta['num_rows']}")

    ss = root / "split_summary.json"
    if ss.exists():
        s = json.loads(ss.read_text())
        if s.get("total_rows") != meta["num_rows"]:
            out["problems"].append(
                f"split_summary total_rows {s.get('total_rows')} != "
                f"metadata num_rows {meta['num_rows']}")
        for key in ("session_split_counts", "generalization_split_counts"):
            if key in s and sum(s[key].values()) != meta["num_rows"]:
                out["problems"].append(
                    f"{key} sums to {sum(s[key].values())}, "
                    f"expected {meta['num_rows']}")

    counts = {d: sum(1 for p in (root / d).rglob("*") if p.is_file())
              for d in ("audio", "accelerometer", "clips") if (root / d).is_dir()}
    out["media_counts"] = counts
    for d, n in counts.items():
        if n != meta["num_rows"]:
            out["problems"].append(
                f"{d}/ has {n} files, expected {meta['num_rows']} "
                f"({meta['num_rows'] - n} missing)")

    out["status"] = "PASS" if not out["problems"] else "FAIL"
    return out


GROUP_RE = re.compile(r"^\s+(\S+)\s*/\s*(\S+):\s*(\d+)\s*$")
STAT_RE = re.compile(r"^\s+(selected|overlap-removed)\s+(\d+)\s+samples\s*$")


def check_selection(root: Path, builder: Path) -> dict:
    """Run the project's own selection code and compare it to the release."""
    out = {"name": "selection", "problems": []}
    if not builder.exists():
        out["status"] = "SKIP"
        out["reason"] = f"builder not found at {builder}"
        return out

    proc = subprocess.run(
        [sys.executable, str(builder), "--dry-run"],
        cwd=str(builder.resolve().parents[1]),
        capture_output=True, text=True, timeout=3600)
    if proc.returncode != 0:
        out["status"] = "FAIL"
        out["problems"].append(f"builder --dry-run exited {proc.returncode}")
        out["stderr_tail"] = proc.stderr[-2000:]
        return out

    sel, stats = {}, {}
    for line in proc.stdout.splitlines():
        m = STAT_RE.match(line)
        if m:
            stats[m.group(1)] = int(m.group(2))
            continue
        m = GROUP_RE.match(line)
        if m:
            exp, col, n = m.group(1), m.group(2), int(m.group(3))
            sel[f"{exp}/{published_bird_color(exp, col)}"] = n
    if not sel:
        out["status"] = "FAIL"
        out["problems"].append("could not parse group counts from --dry-run")
        return out

    out["builder_stats"] = stats
    out["selected_groups"] = dict(sorted(sel.items()))

    meta = json.loads((root / "metadata.json").read_text())
    rel = meta["groups"]

    # The release covers only some experiments; ignore any it never claimed.
    released_exps = {g.rsplit("/", 1)[0] for g in rel}
    ignored = sorted(g for g in sel if g.rsplit("/", 1)[0] not in released_exps)
    if ignored:
        out["ignored_not_in_release"] = ignored

    deltas = {}
    for g in sorted(set(sel) | set(rel)):
        if g.rsplit("/", 1)[0] not in released_exps:
            continue
        s, r = sel.get(g, 0), rel.get(g, 0)
        frac = (s - r) / s if s else None
        deltas[g] = {"selected": s, "released": r, "shortfall": s - r,
                     "shortfall_frac": round(frac, 4) if frac is not None else None}
        if r > s:
            out["problems"].append(
                f"{g}: release has MORE than selection ({r} > {s})")
        elif frac is not None and frac > MAX_SHORTFALL:
            out["problems"].append(
                f"{g}: {frac:.0%} short of selection (> {MAX_SHORTFALL:.0%})")
    out["per_group"] = deltas

    by_exp = {}
    for g, d in deltas.items():
        exp, col = g.rsplit("/", 1)
        by_exp.setdefault(exp, []).append((col, d))
    for exp, items in by_exp.items():
        for i, (c1, d1) in enumerate(items):
            for c2, d2 in items[i + 1:]:
                if all((d1["selected"], d2["selected"], d1["released"], d2["released"])):
                    if (d1["selected"] / max(d2["selected"], 1) > SWAP_RATIO
                            and d2["released"] / max(d1["released"], 1) > SWAP_RATIO):
                        out["problems"].append(
                            f"{exp}: '{c1}' and '{c2}' look TRANSPOSED "
                            f"(selected {d1['selected']}/{d2['selected']}, "
                            f"released {d1['released']}/{d2['released']})")

    out["status"] = "PASS" if not out["problems"] else "FAIL"
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=Path, required=True)
    ap.add_argument("--builder", type=Path, default=None,
                    help="path to build_dataset/build_v33da.py")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    checks = [check_release(args.dataset)]
    if args.builder:
        checks.append(check_selection(args.dataset, args.builder))

    status = "FAIL" if any(c["status"] == "FAIL" for c in checks) else "PASS"
    report = {"pass_fail_status": status, "dataset": str(args.dataset),
              "checks": checks}
    if args.out:
        args.out.write_text(json.dumps(report, indent=2))

    for c in checks:
        print(f"[{c['status']:4s}] {c['name']}")
        for p in c.get("problems", []):
            print(f"         - {p}")
        if c.get("reason"):
            print(f"         ({c['reason']})")
    print(f"\noverall: {status}")
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())

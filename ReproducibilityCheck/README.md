# Reproducibility check

Verifies that a released V33DA build is internally consistent, and that it
still agrees with the inputs it was built from.

```bash
# consistency only — needs just the dataset
python ReproducibilityCheck/check_release.py --dataset data/v33da

# also re-run selection and compare — needs the build inputs
python ReproducibilityCheck/check_release.py \
    --dataset data/v33da \
    --build-inputs /path/to/v33da/data \
    --out ReproducibilityCheck/report.json
```

Runs in seconds. No GPU, no network, no lab drive.

## The two checks

**release** — parquet row count against `metadata.json`, `split_summary.json`
totals against both, and one `audio/`, `accelerometer/` and `clips/` file per
event. Catches a truncated or partially-uploaded release.

**selection** — re-derives which samples should be in the dataset from
`validations.json`, `all_candidates.csv` and `frame_keypoints.json`, then
compares per-group counts against `metadata.json`.

This one does **not** expect the numbers to match. Media extraction fails on
some samples, so the release is legitimately smaller than the selection. What
matters is the *shape* of the gap:

- a few percent lost per group, fairly uniform — normal
- a group the release has **more** of than the selection — impossible, flagged
- a group that lost most of its samples while another in the same experiment
  gained — the colour labels are transposed, flagged

## Colour conventions

`metadata.json` reports **published** bird colours. The builder works in
**detector** convention. For juvExpBP05 those differ: the detector reports
yellow and brown swapped relative to the physical backpack tags, and
`build_v33da.py` corrects this via `PUBLISHED_COLOR_REMAP` when writing the
published colour. The check applies the same remap before comparing. Without
it juvExpBP05 looks transposed when it is only the two conventions meeting.

## Result, 2026-08-04

`report_2026-08-04.json`, run against the current release.

```
[PASS] release
[PASS] selection
overall: PASS
```

The builder selects 38,033 samples and removes 3,784 overlaps. Per group,
against the release:

| group | selected | released | shortfall |
|---|---|---|---|
| juvExpBP01 / blue | 6,771 | 5,735 | 15.3% |
| juvExpBP01 / peach | 2,580 | 2,277 | 11.7% |
| juvExpBP01 / red | 4,494 | 3,963 | 11.8% |
| juvExpBP01 / white | 4,804 | 4,244 | 11.7% |
| juvExpBP02 / blue | 5,314 | 4,923 | 7.4% |
| juvExpBP02 / peach | 5,049 | 4,589 | 9.1% |
| juvExpBP02 / red | 2,245 | 2,065 | 8.0% |
| juvExpBP05 / brown | 68 | 56 | 17.6% |
| juvExpBP05 / purple | 2,449 | 2,140 | 12.6% |
| juvExpBP05 / yellow | 4,259 | 3,633 | 14.7% |

`--dry-run` stops before media extraction, so the shortfall is the samples the
published build dropped when cutting audio, accelerometer and clips. It is
7-18% and broadly even across groups, with no outlier.

## Note on the release check

It passes against a local build. It will fail against the copy published on the
Hub, which is missing 7,819 `audio/` and 7,817 `accelerometer/` files out of
33,625 — that is what the media-count check is for.

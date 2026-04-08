#!/usr/bin/env python3
"""
Compute V33DL dataset statistics and save figures.

1. Reprojection error distribution per keypoint (histogram + stats)
2. Inter-frame displacement per keypoint
3. Beak reprojection error (spatial precision of ground truth)
4. Per-bird sample counts per split (stacked bar)
5. Vocalization duration distribution
6. Per-experiment per-date sample counts
"""

import io
import os
import sys
import pickle
from pathlib import Path
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import cv2
import soundfile as sf

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from v33da.data.dataset import V33DLDataset

# ── Config ───────────────────────────────────────────────────────────────────
FIG_DIR = Path(os.environ.get("V33DA_FIG_DIR", "./figures"))
FIG_DIR.mkdir(parents=True, exist_ok=True)

CAL_BASE = Path(os.environ.get("V33DA_CAL_DIR", "./data/calibrations"))
KEYPOINT_NAMES = ["beak", "head", "backpack", "tail_base", "tail_end"]
KP_COLORS = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12", "#9b59b6"]

# Experiment name -> calibration folder
EXP_TO_CAL = {
    "juvExpBP01": "juvExpBP01",
    "juvExpBP02": "VOCIM_juvExpBP02",
    "juvExpBP05": "VOCIM_juvExpBP05",
}

# Z-up -> Y-up rotation (inverse of R_ZUP)
R_ZUP_INV = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], dtype=np.float64)

FRAME_RATE = 47.6837158203125  # fps


def _load_np(raw_bytes):
    return np.load(io.BytesIO(raw_bytes))


# ── Load calibrations ────────────────────────────────────────────────────────
def load_calibrations():
    cals = {}
    for exp, cal_folder in EXP_TO_CAL.items():
        cals[exp] = {}
        for view in ["top", "back"]:
            cal_dir = CAL_BASE / cal_folder
            npz = np.load(str(cal_dir / f"calibration_{view}.npz"))
            with open(str(cal_dir / f"camera_pose_{view}.pkl"), "rb") as f:
                pose = pickle.load(f)
            cals[exp][view] = {
                "K": npz["mtx"],
                "rvec": pose["rvec"],
                "tvec": pose["tvec"],
            }
    return cals


def project_3d_to_2d(pts_3d_zup, cal):
    """Project 3D points (Z-up, mm) to 2D pixels using camera calibration.

    pts_3d_zup: (..., 3) array in Z-up convention
    cal: dict with K, rvec, tvec
    Returns: (..., 2) projected 2D coordinates
    """
    shape = pts_3d_zup.shape[:-1]
    flat = pts_3d_zup.reshape(-1, 3).astype(np.float64)
    # Rotate to Y-up for camera projection
    flat_yup = (R_ZUP_INV @ flat.T).T
    proj, _ = cv2.projectPoints(
        flat_yup.reshape(-1, 1, 3),
        cal["rvec"], cal["tvec"], cal["K"], np.zeros(5)
    )
    return proj.reshape(*shape, 2)


def save_fig(fig, name):
    """Save figure as both PDF and PNG."""
    fig.savefig(FIG_DIR / f"{name}.pdf", bbox_inches="tight", dpi=150)
    fig.savefig(FIG_DIR / f"{name}.png", bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"  Saved {name}.pdf + .png")


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    print("Loading calibrations...")
    cals = load_calibrations()

    # Collect statistics across all splits
    reproj_errors = {view: {kp: [] for kp in KEYPOINT_NAMES} for view in ["top", "back"]}
    displacement = {kp: [] for kp in KEYPOINT_NAMES}
    beak_reproj_all = []  # combined top+back for beak

    split_bird_counts = defaultdict(lambda: defaultdict(int))  # split -> bird_name -> count
    exp_date_counts = defaultdict(lambda: defaultdict(int))    # exp -> date -> count
    voc_durations = []

    split_sample_counts = {}

    for split in ["train", "val", "test"]:
        print(f"\nProcessing split: {split}...")
        ds = V33DLDataset(split=split)
        n = len(ds.table)
        split_sample_counts[split] = n

        exps = ds.table.column("experiment").to_pylist()
        dates = ds.table.column("date").to_pylist()
        voc_idxs = ds.table.column("vocalizer_idx").to_pylist()
        audio_paths = ds.table.column("audio_path").to_pylist()

        # Load bird names mapping
        import json
        meta_path = ds.dataset_dir / "metadata.json"
        exp_bird_names = {}
        if meta_path.exists():
            with open(meta_path) as f:
                meta = json.load(f)
            exp_bird_names = {
                exp: info["bird_names"]
                for exp, info in meta["experiments"].items()
            }

        kp3d_col = ds.table.column("keypoints_3d")
        kp2d_top_col = ds.table.column("keypoints_2d_top")
        kp2d_back_col = ds.table.column("keypoints_2d_back")

        for i in range(n):
            if i % 2000 == 0:
                print(f"  {i}/{n}...")

            exp = exps[i]
            date = dates[i]
            voc_idx = voc_idxs[i]

            # Per-experiment per-date counts
            exp_date_counts[exp][date] += 1

            # Per-bird counts
            bird_names = exp_bird_names.get(exp, [])
            if bird_names and voc_idx < len(bird_names):
                bird_label = f"{exp}/{bird_names[voc_idx]}"
            else:
                bird_label = f"{exp}/bird_{voc_idx}"
            split_bird_counts[split][bird_label] += 1

            # Load arrays
            kp3d = _load_np(kp3d_col[i].as_py())    # (n_frames, n_birds, 5, 3)
            kp2d_top = _load_np(kp2d_top_col[i].as_py())  # (n_frames, n_birds, 5, 2)
            kp2d_back = _load_np(kp2d_back_col[i].as_py())  # (n_frames, n_birds, 5, 2)

            n_frames, n_birds, n_kps, _ = kp3d.shape

            # Vocalization duration (from audio file)
            audio_path = ds.dataset_dir / audio_paths[i]
            if audio_path.exists():
                try:
                    info = sf.info(str(audio_path))
                    voc_durations.append(info.duration)
                except Exception:
                    pass

            # ── Reprojection error ──
            if exp in cals:
                cal_top = cals[exp]["top"]
                cal_back = cals[exp]["back"]

                for bird_j in range(n_birds):
                    for frame_f in range(n_frames):
                        pts3d = kp3d[frame_f, bird_j]  # (5, 3)
                        pts2d_top_obs = kp2d_top[frame_f, bird_j]  # (5, 2)
                        pts2d_back_obs = kp2d_back[frame_f, bird_j]  # (5, 2)

                        # Skip if any NaN
                        valid_top = ~np.any(np.isnan(pts3d), axis=1) & ~np.any(np.isnan(pts2d_top_obs), axis=1)
                        valid_back = ~np.any(np.isnan(pts3d), axis=1) & ~np.any(np.isnan(pts2d_back_obs), axis=1)

                        if np.any(valid_top):
                            proj_top = project_3d_to_2d(pts3d[valid_top], cal_top)
                            errs = np.linalg.norm(proj_top - pts2d_top_obs[valid_top], axis=1)
                            for ki_local, ki_global in enumerate(np.where(valid_top)[0]):
                                reproj_errors["top"][KEYPOINT_NAMES[ki_global]].extend(errs[ki_local:ki_local+1].tolist())
                                if KEYPOINT_NAMES[ki_global] == "beak":
                                    beak_reproj_all.extend(errs[ki_local:ki_local+1].tolist())

                        if np.any(valid_back):
                            proj_back = project_3d_to_2d(pts3d[valid_back], cal_back)
                            errs = np.linalg.norm(proj_back - pts2d_back_obs[valid_back], axis=1)
                            for ki_local, ki_global in enumerate(np.where(valid_back)[0]):
                                reproj_errors["back"][KEYPOINT_NAMES[ki_global]].extend(errs[ki_local:ki_local+1].tolist())
                                if KEYPOINT_NAMES[ki_global] == "beak":
                                    beak_reproj_all.extend(errs[ki_local:ki_local+1].tolist())

            # ── Displacement (3D, between consecutive frames) ──
            for bird_j in range(n_birds):
                for frame_f in range(1, n_frames):
                    pts_prev = kp3d[frame_f - 1, bird_j]  # (5, 3)
                    pts_curr = kp3d[frame_f, bird_j]       # (5, 3)
                    valid = ~np.any(np.isnan(pts_prev), axis=1) & ~np.any(np.isnan(pts_curr), axis=1)
                    if np.any(valid):
                        disp = np.linalg.norm(pts_curr[valid] - pts_prev[valid], axis=1)
                        for ki_local, ki_global in enumerate(np.where(valid)[0]):
                            displacement[KEYPOINT_NAMES[ki_global]].append(disp[ki_local])

    # ═════════════════════════════════════════════════════════════════════════
    # FIGURE 1: Reprojection error per keypoint (histogram)
    # ═════════════════════════════════════════════════════════════════════════
    print("\n── Figure 1: Reprojection errors ──")
    fig, axes = plt.subplots(2, 5, figsize=(18, 7), sharey="row")
    fig.suptitle("Reprojection Error per Keypoint (pixels)", fontsize=14, fontweight="bold")

    for row, view in enumerate(["top", "back"]):
        for col, kp in enumerate(KEYPOINT_NAMES):
            ax = axes[row, col]
            data = np.array(reproj_errors[view][kp])
            data = data[np.isfinite(data)]
            if len(data) == 0:
                ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center")
                continue
            # Clip for visualization
            clip_val = np.percentile(data, 99)
            ax.hist(data[data <= clip_val], bins=60, color=KP_COLORS[col], alpha=0.8, edgecolor="white", linewidth=0.3)
            ax.axvline(np.median(data), color="black", ls="--", lw=1.2, label=f"med={np.median(data):.2f}")
            ax.axvline(np.percentile(data, 90), color="gray", ls=":", lw=1, label=f"p90={np.percentile(data, 90):.2f}")
            ax.legend(fontsize=7)
            if row == 0:
                ax.set_title(kp, fontweight="bold")
            if col == 0:
                ax.set_ylabel(f"{view} view\nCount")
            ax.set_xlabel("Error (px)")
            ax.tick_params(labelsize=8)

    fig.tight_layout()
    save_fig(fig, "fig_reproj_error_per_keypoint")

    # Print stats
    print("\n  Reprojection error statistics (pixels):")
    print(f"  {'View':<6} {'Keypoint':<12} {'N':>8} {'Mean':>8} {'Median':>8} {'P90':>8} {'P95':>8}")
    print("  " + "-" * 60)
    for view in ["top", "back"]:
        for kp in KEYPOINT_NAMES:
            data = np.array(reproj_errors[view][kp])
            data = data[np.isfinite(data)]
            if len(data) > 0:
                print(f"  {view:<6} {kp:<12} {len(data):>8} {data.mean():>8.2f} {np.median(data):>8.2f} "
                      f"{np.percentile(data, 90):>8.2f} {np.percentile(data, 95):>8.2f}")

    # ═════════════════════════════════════════════════════════════════════════
    # FIGURE 2: Displacement per keypoint (histogram)
    # ═════════════════════════════════════════════════════════════════════════
    print("\n── Figure 2: Inter-frame displacement ──")
    fig, axes = plt.subplots(1, 5, figsize=(18, 3.5), sharey=True)
    fig.suptitle("Inter-frame 3D Displacement per Keypoint (mm)", fontsize=14, fontweight="bold")

    for col, kp in enumerate(KEYPOINT_NAMES):
        ax = axes[col]
        data = np.array(displacement[kp])
        data = data[np.isfinite(data)]
        if len(data) == 0:
            continue
        clip_val = np.percentile(data, 99)
        ax.hist(data[data <= clip_val], bins=60, color=KP_COLORS[col], alpha=0.8, edgecolor="white", linewidth=0.3)
        ax.axvline(np.median(data), color="black", ls="--", lw=1.2, label=f"med={np.median(data):.1f}")
        ax.axvline(np.percentile(data, 90), color="gray", ls=":", lw=1, label=f"p90={np.percentile(data, 90):.1f}")
        ax.legend(fontsize=7)
        ax.set_title(kp, fontweight="bold")
        ax.set_xlabel("Displacement (mm)")
        if col == 0:
            ax.set_ylabel("Count")

    fig.tight_layout()
    save_fig(fig, "fig_displacement_per_keypoint")

    print("\n  Displacement statistics (mm, per frame @ {:.1f} fps):".format(FRAME_RATE))
    print(f"  {'Keypoint':<12} {'N':>10} {'Mean':>8} {'Median':>8} {'P90':>8} {'P95':>8}")
    print("  " + "-" * 56)
    for kp in KEYPOINT_NAMES:
        data = np.array(displacement[kp])
        data = data[np.isfinite(data)]
        if len(data) > 0:
            print(f"  {kp:<12} {len(data):>10} {data.mean():>8.2f} {np.median(data):>8.2f} "
                  f"{np.percentile(data, 90):>8.2f} {np.percentile(data, 95):>8.2f}")

    # ═════════════════════════════════════════════════════════════════════════
    # FIGURE 3: Beak reprojection error (spatial precision claim)
    # ═════════════════════════════════════════════════════════════════════════
    print("\n── Figure 3: Beak reprojection error (spatial precision) ──")
    beak_data = np.array(beak_reproj_all)
    beak_data = beak_data[np.isfinite(beak_data)]

    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    clip_val = np.percentile(beak_data, 99)
    ax.hist(beak_data[beak_data <= clip_val], bins=80, color="#e74c3c", alpha=0.85, edgecolor="white", linewidth=0.3)
    ax.axvline(np.median(beak_data), color="black", ls="--", lw=1.5,
               label=f"Median: {np.median(beak_data):.2f} px")
    ax.axvline(np.mean(beak_data), color="navy", ls="-.", lw=1.2,
               label=f"Mean: {np.mean(beak_data):.2f} px")
    ax.axvline(np.percentile(beak_data, 90), color="gray", ls=":", lw=1.2,
               label=f"P90: {np.percentile(beak_data, 90):.2f} px")
    ax.legend(fontsize=10)
    ax.set_xlabel("Beak Reprojection Error (pixels)", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title("Beak Reprojection Error (Top + Back Views)", fontsize=13, fontweight="bold")
    fig.tight_layout()
    save_fig(fig, "fig_beak_reproj_error")

    print(f"\n  Beak reprojection error (N={len(beak_data)}):")
    print(f"    Mean:   {beak_data.mean():.3f} px")
    print(f"    Median: {np.median(beak_data):.3f} px")
    print(f"    P90:    {np.percentile(beak_data, 90):.3f} px")
    print(f"    P95:    {np.percentile(beak_data, 95):.3f} px")
    print(f"    P99:    {np.percentile(beak_data, 99):.3f} px")

    # ═════════════════════════════════════════════════════════════════════════
    # FIGURE 4: Per-bird sample counts per split (stacked bar)
    # ═════════════════════════════════════════════════════════════════════════
    print("\n── Figure 4: Per-bird sample counts per split ──")
    all_birds = sorted(set().union(*[set(v.keys()) for v in split_bird_counts.values()]))

    fig, ax = plt.subplots(1, 1, figsize=(max(8, len(all_birds) * 0.8), 5))
    x = np.arange(len(all_birds))
    width = 0.7
    bottom = np.zeros(len(all_birds))
    split_colors = {"train": "#3498db", "val": "#f39c12", "test": "#e74c3c"}

    for split in ["train", "val", "test"]:
        counts = [split_bird_counts[split].get(b, 0) for b in all_birds]
        ax.bar(x, counts, width, bottom=bottom, label=split, color=split_colors[split], edgecolor="white", linewidth=0.5)
        bottom += np.array(counts)

    # Add total labels
    for xi, b in enumerate(all_birds):
        total = sum(split_bird_counts[s].get(b, 0) for s in ["train", "val", "test"])
        ax.text(xi, bottom[xi] + 20, str(total), ha="center", va="bottom", fontsize=7, fontweight="bold")

    ax.set_xticks(x)
    short_names = [b.split("/")[-1] for b in all_birds]
    exp_labels = [b.split("/")[0].replace("juvExp", "") for b in all_birds]
    combined_labels = [f"{n}\n({e})" for n, e in zip(short_names, exp_labels)]
    ax.set_xticklabels(combined_labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Number of Vocalizations")
    ax.set_title("Vocalization Samples per Bird per Split", fontweight="bold")
    ax.legend()
    fig.tight_layout()
    save_fig(fig, "fig_perbird_split_counts")

    print("\n  Per-bird sample counts:")
    print(f"  {'Bird':<30} {'Train':>7} {'Val':>7} {'Test':>7} {'Total':>7}")
    print("  " + "-" * 58)
    for b in all_birds:
        tr = split_bird_counts["train"].get(b, 0)
        va = split_bird_counts["val"].get(b, 0)
        te = split_bird_counts["test"].get(b, 0)
        print(f"  {b:<30} {tr:>7} {va:>7} {te:>7} {tr+va+te:>7}")

    # ═════════════════════════════════════════════════════════════════════════
    # FIGURE 5: Vocalization duration distribution
    # ═════════════════════════════════════════════════════════════════════════
    print("\n── Figure 5: Vocalization duration distribution ──")
    dur = np.array(voc_durations) * 1000  # ms

    fig, ax = plt.subplots(1, 1, figsize=(7, 4))
    ax.hist(dur, bins=80, color="#2ecc71", alpha=0.85, edgecolor="white", linewidth=0.3)
    ax.axvline(np.median(dur), color="black", ls="--", lw=1.5,
               label=f"Median: {np.median(dur):.1f} ms")
    ax.axvline(np.mean(dur), color="navy", ls="-.", lw=1.2,
               label=f"Mean: {np.mean(dur):.1f} ms")
    ax.legend(fontsize=10)
    ax.set_xlabel("Duration (ms)", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title("Vocalization Duration Distribution", fontsize=13, fontweight="bold")
    fig.tight_layout()
    save_fig(fig, "fig_vocalization_duration")

    print(f"\n  Vocalization durations (N={len(dur)}):")
    print(f"    Mean:   {np.mean(dur):.1f} ms")
    print(f"    Median: {np.median(dur):.1f} ms")
    print(f"    Std:    {np.std(dur):.1f} ms")
    print(f"    Min:    {np.min(dur):.1f} ms")
    print(f"    Max:    {np.max(dur):.1f} ms")
    print(f"    P5:     {np.percentile(dur, 5):.1f} ms")
    print(f"    P95:    {np.percentile(dur, 95):.1f} ms")

    # ═════════════════════════════════════════════════════════════════════════
    # FIGURE 6: Per-experiment per-date sample counts
    # ═════════════════════════════════════════════════════════════════════════
    print("\n── Figure 6: Per-experiment per-date sample counts ──")

    # Sort experiments and dates
    experiments = sorted(exp_date_counts.keys())
    all_dates = sorted(set().union(*[set(v.keys()) for v in exp_date_counts.values()]))

    fig, ax = plt.subplots(1, 1, figsize=(max(12, len(all_dates) * 0.5), 5))
    x = np.arange(len(all_dates))
    width = 0.8 / len(experiments)
    exp_colors = ["#3498db", "#e74c3c", "#2ecc71", "#f39c12"]

    for ei, exp in enumerate(experiments):
        counts = [exp_date_counts[exp].get(d, 0) for d in all_dates]
        offset = (ei - len(experiments) / 2 + 0.5) * width
        bars = ax.bar(x + offset, counts, width, label=exp, color=exp_colors[ei % len(exp_colors)],
                       edgecolor="white", linewidth=0.3)

    ax.set_xticks(x)
    ax.set_xticklabels(all_dates, rotation=90, fontsize=6)
    ax.set_ylabel("Number of Vocalizations")
    ax.set_title("Sample Counts per Experiment per Recording Date", fontweight="bold")
    ax.legend()
    fig.tight_layout()
    save_fig(fig, "fig_experiment_date_counts")

    print("\n  Per-experiment per-date counts:")
    for exp in experiments:
        dates_sorted = sorted(exp_date_counts[exp].keys())
        total = sum(exp_date_counts[exp].values())
        print(f"  {exp}: {len(dates_sorted)} dates, {total} total samples")
        for d in dates_sorted:
            print(f"    {d}: {exp_date_counts[exp][d]}")

    # ═════════════════════════════════════════════════════════════════════════
    # FINAL SUMMARY
    # ═════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("DATASET STATISTICS SUMMARY")
    print("=" * 70)
    total = sum(split_sample_counts.values())
    print(f"\nTotal samples: {total}")
    for s in ["train", "val", "test"]:
        print(f"  {s}: {split_sample_counts[s]} ({100*split_sample_counts[s]/total:.1f}%)")

    print(f"\nExperiments: {len(experiments)}")
    for exp in experiments:
        n_dates = len(exp_date_counts[exp])
        n_samp = sum(exp_date_counts[exp].values())
        print(f"  {exp}: {n_dates} recording dates, {n_samp} samples")

    print(f"\nBirds: {len(all_birds)}")
    print(f"Vocalization durations: mean={np.mean(dur):.1f}ms, median={np.median(dur):.1f}ms, "
          f"range=[{np.min(dur):.1f}, {np.max(dur):.1f}]ms")

    print(f"\nBeak reprojection error (spatial precision):")
    print(f"  Mean: {beak_data.mean():.2f} px, Median: {np.median(beak_data):.2f} px, "
          f"P90: {np.percentile(beak_data, 90):.2f} px")

    # Combined reproj error across all keypoints
    all_reproj = []
    for view in ["top", "back"]:
        for kp in KEYPOINT_NAMES:
            all_reproj.extend(reproj_errors[view][kp])
    all_reproj = np.array(all_reproj)
    all_reproj = all_reproj[np.isfinite(all_reproj)]
    print(f"\nOverall reprojection error (all keypoints, both views, N={len(all_reproj)}):")
    print(f"  Mean: {all_reproj.mean():.2f} px, Median: {np.median(all_reproj):.2f} px, "
          f"P90: {np.percentile(all_reproj, 90):.2f} px")

    all_disp = []
    for kp in KEYPOINT_NAMES:
        all_disp.extend(displacement[kp])
    all_disp = np.array(all_disp)
    all_disp = all_disp[np.isfinite(all_disp)]
    print(f"\nOverall inter-frame displacement (all keypoints, N={len(all_disp)}):")
    print(f"  Mean: {all_disp.mean():.2f} mm, Median: {np.median(all_disp):.2f} mm, "
          f"P90: {np.percentile(all_disp, 90):.2f} mm")

    print(f"\nFigures saved to: {FIG_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()

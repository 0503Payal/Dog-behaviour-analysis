"""
Tracker diagnostics for the dog-tracking pipeline.

Counting track IDs tells you how badly a tracker fragments, but not *why* or
whether the fragmentation is safe to ignore. This script answers three questions
that decide what downstream behavioural analysis is valid:

  1. How concentrated is the tracking?  (a few long tracks, or uniform debris?)
  2. Is fragmentation random, or correlated with what the dog is doing?
  3. Are the derived kinematics physically plausible?

Question 2 is the one that matters. Missing data that correlates with behaviour
biases every behavioural statistic computed from it; missing data that doesn't,
mostly averages out.

Usage:
    python tracker_diagnostics.py --traj path/to/trajectories_per_frame.csv.gz \
                                  --summary path/to/summary_metrics_per_dog.csv \
                                  --outdir figures --fps 49.95 --dogs 3
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

EDGE_FRAMES = 10      # frames at each end of a track treated as "birth"/"death"
FRAME_WIDTH_PX = 720  # video is 720x576


def load(traj_path: Path, summary_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    traj = pd.read_csv(traj_path).sort_values(["track_id", "frame"])
    summary = pd.read_csv(summary_path)
    traj["gap"] = traj.groupby("track_id").frame.diff()
    return traj, summary


def concentration(summary: pd.DataFrame, n_dogs: int) -> dict:
    """How much of the data sits in how few tracks."""
    lengths = summary.total_frames_tracked.sort_values(ascending=False)
    total = lengths.sum()
    out = {
        "n_track_ids": int(len(lengths)),
        "ids_per_dog": round(len(lengths) / n_dogs, 1),
        "longest_track_frames": int(lengths.iloc[0]),
        "detection_share_top3_pct": round(lengths.head(3).sum() / total * 100, 1),
        "detection_share_top5_pct": round(lengths.head(5).sum() / total * 100, 1),
    }
    for thr in (5, 25, 250):
        short = lengths[lengths <= thr]
        out[f"tracks_le_{thr}f_pct"] = round(len(short) / len(lengths) * 100, 1)
        out[f"detections_in_le_{thr}f_pct"] = round(short.sum() / total * 100, 2)
    return out


def break_bias(traj: pd.DataFrame) -> dict:
    """Do tracks start and end at different speeds than they live at?

    If yes, fragmentation is behaviour-dependent: fast bouts get chopped into
    fragments while slow ones survive intact, so any bout-level statistic is
    biased even though the per-frame detections are fine.
    """
    grouped = traj.groupby("track_id")
    birth = grouped.head(EDGE_FRAMES)
    death = grouped.tail(EDGE_FRAMES)
    middle = traj.drop(index=birth.index.union(death.index))

    speed = "speed_smoothed_px_per_sec"
    med = lambda frame, col: float(frame[col].median())
    mid_speed = med(middle, speed)
    return {
        "birth_speed_px_s": round(med(birth, speed), 1),
        "death_speed_px_s": round(med(death, speed), 1),
        "mid_track_speed_px_s": round(mid_speed, 1),
        "break_speed_ratio": round(med(death, speed) / mid_speed, 1),
        "birth_confidence": round(med(birth, "confidence"), 3),
        "death_confidence": round(med(death, "confidence"), 3),
        "mid_track_confidence": round(med(middle, "confidence"), 3),
    }


def kinematic_validity(traj: pd.DataFrame, fps: float) -> dict:
    """Check the speed column against physics, and against its own gaps.

    Speed is computed as step_distance / (1/fps). That is only correct when
    consecutive rows are consecutive frames. Where the tracker bridged a gap,
    several frames elapsed but one frame's worth of time is used, so the
    reported speed is inflated by the gap length.
    """
    bridged = traj[traj.gap > 1].copy()
    bridged["speed_corrected"] = bridged.step_distance_px / (bridged.gap / fps)

    contiguous = traj[traj.gap == 1]
    jump = (
        contiguous.groupby("track_id").center_x.diff().abs()
        + contiguous.groupby("track_id").center_y.diff().abs()
    )

    # A dog crossing a quarter of the frame in one 20 ms step is not a dog.
    implausible = FRAME_WIDTH_PX / 4
    return {
        "gap_bridged_rows": int(len(bridged)),
        "gap_bridged_pct": round(len(bridged) / len(traj) * 100, 2),
        "median_gap_frames": float(bridged.gap.median()) if len(bridged) else 0.0,
        "max_gap_frames": float(bridged.gap.max()) if len(bridged) else 0.0,
        "speed_inflation_at_gaps_x": round(
            float((bridged.speed_px_per_sec / bridged.speed_corrected).median()), 1
        ),
        "reported_max_speed_px_s": round(float(traj.speed_px_per_sec.max()), 1),
        "corrected_max_speed_px_s": round(float(bridged.speed_corrected.max()), 1),
        "contiguous_jumps_over_quarter_frame": int((jump > implausible).sum()),
        "max_contiguous_jump_px": round(float(jump.max()), 1),
    }


def coverage(traj: pd.DataFrame, n_dogs: int) -> dict:
    """Are the right number of dogs detected at any instant?

    This is independent of identity. If instantaneous counts are right, the
    detector is sound and only the across-time linking is broken.
    """
    per_frame = traj.groupby("frame").track_id.nunique()
    return {
        "mean_ids_alive": round(float(per_frame.mean()), 2),
        "median_ids_alive": int(per_frame.median()),
        "max_ids_alive": int(per_frame.max()),
        "frames_at_expected_count_pct": round(float((per_frame == n_dogs).mean() * 100), 1),
        "frames_undercounting_pct": round(float((per_frame < n_dogs).mean() * 100), 1),
        "frames_overcounting_pct": round(float((per_frame > n_dogs).mean() * 100), 1),
    }


def plot_break_bias(traj: pd.DataFrame, outdir: Path) -> None:
    grouped = traj.groupby("track_id")
    birth = grouped.head(EDGE_FRAMES)
    death = grouped.tail(EDGE_FRAMES)
    middle = traj.drop(index=birth.index.union(death.index))
    speed = "speed_smoothed_px_per_sec"

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))

    groups = [("mid-track", middle, "#4C72B0"),
              ("track birth", birth, "#DD8452"),
              ("track death", death, "#C44E52")]

    positions = range(len(groups))
    ax1.boxplot([np.log10(g[speed].clip(lower=1)) for _, g, _ in groups],
                positions=list(positions), widths=0.55, showfliers=False,
                medianprops=dict(color="black", linewidth=1.8),
                boxprops=dict(linewidth=1.1), whiskerprops=dict(linewidth=1.1),
                capprops=dict(linewidth=1.1))
    for i, (_, g, colour) in enumerate(groups):
        ax1.scatter([i], [np.log10(max(g[speed].median(), 1))], s=70, color=colour,
                    zorder=5, edgecolor="white", linewidth=1.3)
    ax1.set_xticks(list(positions))
    ax1.set_xticklabels([label for label, _, _ in groups])
    ax1.set_ylabel("speed  (log$_{10}$ px/s)")
    ax1.set_title("Tracks break while the dog is moving fast", fontsize=11, loc="left")

    ax2.boxplot([g.confidence for _, g, _ in groups],
                positions=list(positions), widths=0.55, showfliers=False,
                medianprops=dict(color="black", linewidth=1.8),
                boxprops=dict(linewidth=1.1), whiskerprops=dict(linewidth=1.1),
                capprops=dict(linewidth=1.1))
    for i, (_, g, colour) in enumerate(groups):
        ax2.scatter([i], [g.confidence.median()], s=70, color=colour,
                    zorder=5, edgecolor="white", linewidth=1.3)
    ax2.set_xticks(list(positions))
    ax2.set_xticklabels([label for label, _, _ in groups])
    ax2.set_ylabel("YOLO detection confidence")
    ax2.set_title("...because confidence collapses there", fontsize=11, loc="left")

    for ax in (ax1, ax2):
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=0.25)

    fig.suptitle("Fragmentation is behaviour-dependent, not random",
                 fontsize=13, x=0.01, ha="left", weight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(outdir / "break_bias.png", dpi=150)
    plt.close(fig)


def plot_concentration(summary: pd.DataFrame, outdir: Path) -> None:
    lengths = summary.total_frames_tracked.sort_values(ascending=False).values
    cumulative = np.cumsum(lengths) / lengths.sum() * 100
    rank = np.arange(1, len(lengths) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))

    ax1.bar(rank, lengths, color="#4C72B0", width=1.0)
    ax1.set_yscale("log")
    ax1.set_xlabel("track rank")
    ax1.set_ylabel("frames tracked (log)")
    ax1.set_title(f"{len(lengths)} track IDs, heavily skewed", fontsize=11, loc="left")

    ax2.plot(rank, cumulative, color="#C44E52", linewidth=2)
    ax2.axhline(90, ls="--", lw=1, color="grey")
    n90 = int(np.searchsorted(cumulative, 90) + 1)
    ax2.axvline(n90, ls="--", lw=1, color="grey")
    ax2.annotate(f"90% of detections\nin {n90} tracks",
                 xy=(n90, 90), xytext=(n90 + len(lengths) * 0.12, 62),
                 fontsize=9, arrowprops=dict(arrowstyle="->", lw=1, color="grey"))
    ax2.set_xlabel("number of tracks (longest first)")
    ax2.set_ylabel("cumulative % of detections")
    ax2.set_ylim(0, 102)
    ax2.set_title("Most IDs are debris", fontsize=11, loc="left")

    for ax in (ax1, ax2):
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=0.25)

    fig.suptitle("Where the tracking data actually lives",
                 fontsize=13, x=0.01, ha="left", weight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(outdir / "track_concentration.png", dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--traj", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, default=Path("figures"))
    parser.add_argument("--fps", type=float, default=49.95)
    parser.add_argument("--dogs", type=int, default=3,
                        help="expected number of dogs simultaneously visible")
    args = parser.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    traj, summary = load(args.traj, args.summary)

    report = {
        "concentration": concentration(summary, args.dogs),
        "break_bias": break_bias(traj),
        "kinematic_validity": kinematic_validity(traj, args.fps),
        "coverage": coverage(traj, args.dogs),
    }

    plot_break_bias(traj, args.outdir)
    plot_concentration(summary, args.outdir)

    (args.outdir / "diagnostics.json").write_text(json.dumps(report, indent=2))

    for section, values in report.items():
        print(f"\n{section.replace('_', ' ').upper()}")
        for key, value in values.items():
            print(f"  {key:<42} {value}")


if __name__ == "__main__":
    main()

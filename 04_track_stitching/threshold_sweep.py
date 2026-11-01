"""
Choose the stitching threshold by evidence rather than by taste.

Without ground-truth identities, "how many tracks" cannot be the criterion — a
loose threshold always produces fewer. These three checks are what is available:

  dup-frames     a stitched track holding two detections in the same frame means
                 two animals were merged. Must be zero.
  max stitch spd the fastest movement any bridged gap implies. A dog cannot
                 exceed roughly 1000 px/s in this footage.
  break ratio    speed at track ends divided by speed mid-track. 1.0 would mean
                 fragmentation is no longer behaviour-dependent.

Usage:  python threshold_sweep.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from stitch_tracks import stitch, recompute_kinematics

FPS = 49.95
EDGE = 10
TRAJ = "../02_tracking/results/trajectories_per_frame.csv.gz"


def evaluate(traj: pd.DataFrame, max_cost: float) -> dict:
    stitched, _ = stitch(traj, 750, FPS, max_cost)
    stitched = recompute_kinematics(stitched, FPS)

    grouped = stitched.groupby("stitched_id")
    duplicates = int((stitched.groupby(["stitched_id", "frame"]).size() > 1).sum())
    bridged = stitched[grouped.frame.diff() > 1]

    speed = "speed_smoothed_px_per_sec"
    birth, death = grouped.head(EDGE), grouped.tail(EDGE)
    middle = stitched.drop(index=birth.index.union(death.index))

    sizes = grouped.size().sort_values(ascending=False)
    cumulative = np.cumsum(sizes) / sizes.sum() * 100

    return {
        "max_cost": max_cost,
        "track_ids": stitched.stitched_id.nunique(),
        "duplicate_frames": duplicates,
        "max_bridged_speed": round(float(bridged.speed_px_per_sec.max()), 0),
        "break_ratio": round(float(death[speed].median() / middle[speed].median()), 1),
        "tracks_holding_90pct": int(np.searchsorted(cumulative, 90) + 1),
    }


if __name__ == "__main__":
    traj = pd.read_csv(TRAJ)
    results = pd.DataFrame([evaluate(traj, c)
                            for c in (400, 600, 800, 1100, 1500, 2200)])
    print(results.to_string(index=False))
    results.to_csv("threshold_sweep.csv", index=False)

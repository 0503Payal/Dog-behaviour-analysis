"""
Offline track stitching: link fragments of the same animal back together.

BoT-SORT runs online — at frame t it may only use frames up to t, so when a dog
is lost to motion blur the tracker must decide immediately whether a detection
twelve frames later is the same animal. Offline, the whole session is available:
a track that dies at frame 1000 can be matched against one born at frame 1012
using evidence from both sides of the gap.

Each candidate link is scored on three things:
  - where the dying track was heading (constant-velocity prediction, damped)
  - how similar the two bounding boxes are in size
  - how long the gap is

and rejected outright if it would require an implausible speed, or if the two
tracks are alive at the same time (then they are different animals).

Links are solved globally with the Hungarian algorithm rather than greedily, and
the process repeats so that chains A -> B -> C close up.

Usage:
    python stitch_tracks.py --traj ../02_tracking/results/trajectories_per_frame.csv.gz \
                            --out stitched_trajectories.csv --fps 49.95
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

# A dog cannot plausibly sustain more than this. Derived from the data: the 99th
# percentile of single-frame movement is ~15 px (≈745 px/s), so anything above
# this is the tracker, not the animal.
MAX_PLAUSIBLE_SPEED_PX_S = 1000.0

# Velocity decays across a gap — a dog that was sprinting when we lost it has
# usually slowed by the time we see it again, so an undamped constant-velocity
# prediction overshoots badly on long gaps.
VELOCITY_DAMPING = 0.5

EDGE_FRAMES = 5     # frames used to estimate velocity at each track end
BIG = 1e9           # forbidden-link cost


def track_endpoints(traj: pd.DataFrame) -> pd.DataFrame:
    """Summarise each track by how it starts and how it ends."""
    rows = []
    for track_id, group in traj.groupby("track_id"):
        group = group.sort_values("frame")
        head, tail = group.head(EDGE_FRAMES), group.tail(EDGE_FRAMES)

        def velocity(edge: pd.DataFrame) -> tuple[float, float]:
            if len(edge) < 2:
                return 0.0, 0.0
            span = edge.frame.iloc[-1] - edge.frame.iloc[0]
            if span <= 0:
                return 0.0, 0.0
            return ((edge.center_x.iloc[-1] - edge.center_x.iloc[0]) / span,
                    (edge.center_y.iloc[-1] - edge.center_y.iloc[0]) / span)

        vx, vy = velocity(tail)
        rows.append({
            "track_id": track_id,
            "start_frame": group.frame.iloc[0],
            "end_frame": group.frame.iloc[-1],
            "start_x": group.center_x.iloc[0], "start_y": group.center_y.iloc[0],
            "end_x": group.center_x.iloc[-1], "end_y": group.center_y.iloc[-1],
            "exit_vx": vx, "exit_vy": vy,
            "start_w": head.box_width.median(), "start_h": head.box_height.median(),
            "end_w": tail.box_width.median(), "end_h": tail.box_height.median(),
            "n_frames": len(group),
        })
    return pd.DataFrame(rows).set_index("track_id")

def link_cost(dying: pd.Series, newborn: pd.Series, max_gap: int,
              fps: float) -> float:
    """Cost of claiming `newborn` continues `dying`. BIG means forbidden."""
    gap = newborn.start_frame - dying.end_frame
    if gap < 1 or gap > max_gap:
        return BIG

    # Overlapping in time means two animals were visible at once.
    if newborn.start_frame <= dying.end_frame:
        return BIG

    predicted_x = dying.end_x + dying.exit_vx * gap * VELOCITY_DAMPING
    predicted_y = dying.end_y + dying.exit_vy * gap * VELOCITY_DAMPING
    miss = float(np.hypot(newborn.start_x - predicted_x,
                          newborn.start_y - predicted_y))

    # Reject links the animal could not physically have made.
    travelled = float(np.hypot(newborn.start_x - dying.end_x,
                               newborn.start_y - dying.end_y))
    if travelled / (gap / fps) > MAX_PLAUSIBLE_SPEED_PX_S:
        return BIG

    # Apparent size is a weak identity cue, but a cheap one: a dog does not
    # change size across a twelve-frame gap.
    size_a = np.array([dying.end_w, dying.end_h])
    size_b = np.array([newborn.start_w, newborn.start_h])
    size_mismatch = float(np.abs(size_a - size_b).sum() / (size_a.sum() + 1e-6))

    # Scale each term to roughly comparable units before summing.
    return miss + 200.0 * size_mismatch + 0.5 * gap


def stitch_once(endpoints: pd.DataFrame, max_gap: int, fps: float,
                max_cost: float) -> dict[int, int]:
    """One round of global assignment. Returns {newborn_id: dying_id}."""
    ids = endpoints.index.to_numpy()
    cost = np.full((len(ids), len(ids)), BIG)
    for i, dying_id in enumerate(ids):
        for j, newborn_id in enumerate(ids):
            if dying_id != newborn_id:
                cost[i, j] = link_cost(endpoints.loc[dying_id],
                                       endpoints.loc[newborn_id], max_gap, fps)

    rows, cols = linear_sum_assignment(cost)
    return {ids[j]: ids[i] for i, j in zip(rows, cols)
            if cost[i, j] < max_cost}


def stitch(traj: pd.DataFrame, max_gap: int, fps: float, max_cost: float,
           max_rounds: int = 12) -> tuple[pd.DataFrame, list[int]]:
    """Repeatedly merge tracks until no link is good enough. Chains close up."""
    traj = traj.copy()
    traj["stitched_id"] = traj.track_id
    history = [traj.stitched_id.nunique()]

    for _ in range(max_rounds):
        working = traj.rename(columns={"track_id": "_orig",
                                       "stitched_id": "track_id"})
        links = stitch_once(track_endpoints(working), max_gap, fps, max_cost)
        if not links:
            break
        traj["stitched_id"] = traj.stitched_id.map(lambda x: links.get(x, x))
        history.append(traj.stitched_id.nunique())
        if history[-1] == history[-2]:
            break

    return traj, history


def recompute_kinematics(traj: pd.DataFrame, fps: float) -> pd.DataFrame:
    """Recompute speed with the TRUE elapsed time between rows.

    The original pipeline divided by a fixed 1/fps, which is only correct when
    consecutive rows are consecutive frames. Stitching creates far more gaps,
    so getting this right matters more after stitching than before.
    """
    traj = traj.sort_values(["stitched_id", "frame"]).copy()
    grouped = traj.groupby("stitched_id")

    elapsed_frames = grouped.frame.diff()
    step = np.hypot(grouped.center_x.diff(), grouped.center_y.diff())

    traj["step_distance_px"] = step.fillna(0.0)
    traj["elapsed_sec"] = elapsed_frames / fps
    traj["speed_px_per_sec"] = (traj.step_distance_px / traj.elapsed_sec).fillna(0.0)
    traj["cumulative_dist_px"] = grouped.step_distance_px.cumsum()
    traj["speed_smoothed_px_per_sec"] = (
        traj.groupby("stitched_id").speed_px_per_sec
            .transform(lambda s: s.rolling(5, center=True, min_periods=1).mean())
    )
    return traj


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--traj", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("stitched_trajectories.csv"))
    parser.add_argument("--fps", type=float, default=49.95)
    parser.add_argument("--max-gap", type=int, default=750,
                        help="longest gap to bridge, in frames (750 ≈ 15 s)")
    parser.add_argument("--max-cost", type=float, default=800.0,
                        help="reject links above this cost; lower is stricter")
    args = parser.parse_args()

    traj = pd.read_csv(args.traj)
    before = traj.track_id.nunique()

    traj, history = stitch(traj, args.max_gap, args.fps, args.max_cost)
    traj = recompute_kinematics(traj, args.fps)

    after = traj.stitched_id.nunique()
    print(f"track IDs: {before} -> {after}   ({history})")

    sizes = traj.groupby("stitched_id").size().sort_values(ascending=False)
    print(f"90% of detections now in "
          f"{int(np.searchsorted(np.cumsum(sizes) / sizes.sum() * 100, 90) + 1)} tracks")
    print("\nlongest tracks (frames):")
    print(sizes.head(8).to_string())

    traj.to_csv(args.out, index=False)
    print(f"\nwritten: {args.out}")


if __name__ == "__main__":
    main()

# The timestep bug, and the two-line fix

## What is wrong

The kinematics cell computes speed as

```python
dt = 1.0 / fps                      # a constant
df["speed_px_per_sec"] = df["step_distance_px"] / dt
```

`dt` is a constant one frame. That is correct **only when consecutive rows of a
track are consecutive frames** — and they often are not. `track_buffer: 150`
lets BoT-SORT lose a dog for up to 150 frames and then resume the same track ID.
When it does, the next row is many frames later, but it is still divided by one
frame's worth of time.

Raising `track_buffer` from 30 to 150 made this worse, because bridging gaps is
exactly what the longer buffer is for. **There are 910 such rows**, with a median
gap of 5 frames and a maximum of 139.

## What it costs

| | value |
|---|---|
| affected rows | 910 (0.75% of all rows) |
| median speed inflation on those rows | **5×** |
| reported maximum speed | **11,603 px/s** |
| corrected maximum speed | 2,450 px/s |
| tracks whose reported max speed is >1.5× too high | 18 of 127 |
| worst single track | **13× too high** |

Mean speed barely moves (83.1 → 76.4 px/s), so group averages survive. But
every `max_speed_px_s` in `summary_metrics_per_dog.csv` is unreliable, and
11,603 px/s is a dog crossing the 720-pixel frame sixteen times per second —
a number a reviewer can falsify with one division.

## The fix

Compute elapsed time from the frame numbers instead of assuming it:

```python
# BEFORE
dt = 1.0 / fps
df["step_distance_px"] = np.hypot(df.groupby("track_id").center_x.diff(),
                                  df.groupby("track_id").center_y.diff())
df["speed_px_per_sec"] = df["step_distance_px"] / dt

# AFTER
grouped = df.groupby("track_id")
df["elapsed_frames"] = grouped["frame"].diff()          # usually 1, sometimes more
df["elapsed_sec"]    = df["elapsed_frames"] / fps       # the real time that passed
df["step_distance_px"] = np.hypot(grouped.center_x.diff(),
                                  grouped.center_y.diff()).fillna(0.0)
df["speed_px_per_sec"] = (df["step_distance_px"] / df["elapsed_sec"]).fillna(0.0)
```

Two lines changed: `elapsed_sec` replaces the constant `dt`, and it is derived
from `frame.diff()`.

## One thing to decide

After a long gap, speed becomes an *average over the gap*, not an instantaneous
speed — correct, but a different quantity. For a 139-frame gap that is an
average over 2.8 seconds.

The cleaner choice for behavioural analysis is to keep the corrected speed but
**flag** the row, so that windowed features can skip windows containing a long
bridge rather than silently averaging across them:

```python
df["bridged"] = df["elapsed_frames"] > 1
df["long_bridge"] = df["elapsed_frames"] > 0.25 * fps   # gaps over 250 ms
```

Also worth fixing in the same cell: `locomotion_time_pct` currently reads 100%
for 72 short tracks with a median length of 16 frames. A fragment carved out of
a fast bout is 100% locomotion by construction. Report it as `NaN` below a
minimum track length rather than as a real value.

```python
MIN_FRAMES_FOR_BUDGET = 50   # ~1 s; below this a time budget is meaningless
summary.loc[summary.total_frames_tracked < MIN_FRAMES_FOR_BUDGET,
            "locomotion_time_pct"] = np.nan
```

# Offline track stitching

Links fragments of the same animal back together after tracking has finished.

## Why this works when the tracker could not

BoT-SORT runs **online**: at frame *t* it may only use frames up to *t*. When a
dog is lost to motion blur, the tracker has to decide immediately whether a
detection twelve frames later is the same animal, using only what it has seen.

Offline, the whole session is available. A track that dies at frame 1000 can be
matched against one born at frame 1012 using evidence from **both sides** of the
gap — where the first track was heading, and where the second came from.

## How a link is scored

Each candidate pair (dying track, newborn track) is scored on:

- **Motion** — constant-velocity prediction from the dying track's exit velocity,
  damped by 0.5 because a dog that was sprinting when it was lost has usually
  slowed by the time it reappears.
- **Apparent size** — a weak identity cue, but free: a dog does not change size
  across a twelve-frame gap.
- **Gap length** — longer gaps are weaker evidence.

A link is **forbidden outright** if:

- the two tracks are alive at the same time (then they are different animals), or
- it would require a speed above 1000 px/s, derived from the data: the 99th
  percentile of single-frame movement is ~15 px, or about 745 px/s.

Assignment is solved globally with the Hungarian algorithm rather than greedily,
and repeated so that chains A → B → C close up.

## Results

```bash
python stitch_tracks.py --traj ../02_tracking/results/trajectories_per_frame.csv.gz \
                        --out stitched_trajectories.csv.gz
```

**127 → 39 track IDs**, with 90% of detections in 16 tracks (from 31).

## Choosing the threshold honestly

Without ground-truth identities, "fewer tracks" cannot be the criterion — a
looser threshold always produces fewer. `threshold_sweep.py` scores each setting
on checks that can actually fail:

| max_cost | track IDs | duplicate frames | max bridged speed | break ratio | 90% in |
|---|---|---|---|---|---|
| 400 | 77 | 0 | 2450 | 4.0 | 22 |
| 600 | 50 | 0 | 2450 | 4.2 | 20 |
| **800** | **39** | **0** | **2450** | **4.4** | **16** |
| 1100 | 33 | 0 | 2450 | 4.2 | 14 |
| 1500 | 31 | 0 | 2450 | 4.3 | 12 |
| 2200 | 28 | 0 | 2450 | 3.7 | 12 |

**No setting produced a single duplicate frame** — no stitched track ever holds
two detections at the same instant, so no two dogs were merged. 800 is the
default: past it, track count keeps falling but no check improves, which means
the extra merges are being bought with less evidence rather than more.

The residual 2450 px/s comes from gaps BoT-SORT bridged internally, not from
stitched links, and is unchanged by the threshold.

## What this does *not* fix

**The break ratio only falls from 6.2 to about 4.4.** Tracks still end while the
dog is moving roughly four times faster than mid-track.

That is the honest limit of stitching: it repairs identity *continuity*, but the
breaks happen because the **detector** loses blurred dogs, and no amount of
post-hoc linking recovers a detection that was never made. Closing that gap
needs better detection — see
[`../01_detection/scripts/train_yolo_blur_robust.py`](../01_detection/scripts/train_yolo_blur_robust.py)
— or appearance re-identification enabled in the tracker.

So: bout-level statistics are now **usable** but still carry a known bias toward
under-representing fast behaviour, and that caveat belongs in any result derived
from them.

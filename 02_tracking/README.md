# Tracking and kinematics

Turns per-frame detections into per-animal trajectories and movement measurements.

## What this does

1. Runs the tuned YOLOv8-nano detector over all 44,954 frames.
2. Associates detections across frames with BoT-SORT to produce track IDs.
3. Derives kinematics per track: step distance, cumulative distance, instantaneous
   speed, a 5-frame smoothed speed, and inter-dog distance for proximity events.

## Tracker configuration

Stock `botsort.yaml` assumes roughly 30 fps. This video runs at **49.95 fps**, so the
default 30-frame buffer allowed only **0.6 seconds** to re-acquire a dog through a blur
or an occlusion. [`config/custom_botsort.yaml`](config/custom_botsort.yaml) raises it to
150 frames (3.0 s) and tightens the association thresholds, cutting unique track IDs
from **377 to 127**.

| parameter | default | used | effect |
|---|---|---|---|
| `track_buffer` | 30 | **150** | 0.6 s → 3.0 s to re-acquire. The main win. |
| `match_thresh` | 0.8 | 0.95 | more permissive association across a gap |
| `track_high_thresh` | 0.25 | 0.5 | **discarded 5,409 low-confidence detections** |
| `new_track_thresh` | 0.25 | 0.6 | fewer spurious new tracks |
| `with_reid` | false | false | appearance matching still off |
| `gmc_method` | sparseOptFlow | sparseOptFlow | camera-motion compensation |

### The cost of the confidence thresholds

Total detections fell from 128,188 to 122,014. **5,409 of those 6,174 lost detections
(88%) were below 0.5 confidence** — which is where `track_high_thresh` was set.

Low confidence in this footage means motion blur, and motion blur means a fast-moving
dog. So this setting discarded precisely the fast behaviour the project exists to
measure, which is why the speed-dependent fragmentation bias survived the tuning.
[`config/botsort_recommended.yaml`](config/botsort_recommended.yaml) proposes reverting
those two thresholds and enabling `with_reid`.

| metric | default | tuned |
|---|---|---|
| unique track IDs | 377 | 127 |
| new IDs per minute | 25.1 | 8.5 |
| longest unbroken track | 18,458 frames | 21,424 frames |
| tracks ≤25 frames | 57.3% | 37.8% |
| confidence at a track break | 0.57 | 0.73 |

## Outputs

| file | contents |
|---|---|
| `results/trajectories_per_frame.csv.gz` | one row per detection per frame: position, box, confidence, kinematics (122,014 rows) |
| `results/summary_metrics_per_dog.csv` | one row per track: duration, distance, mean and max speed, locomotion time |
| `results/spatial_trajectories_plot.png` | movement paths in the enclosure |
| `results/velocity_profiles_plot.png` | speed over time |
| `results_baseline/` | the same four files from the untuned default config, for comparison |

## Known issues in this output

- **`max_speed_px_s` is unreliable.** Speed assumes consecutive rows are consecutive
  frames; on the 910 gap-bridged rows it is inflated by a median of 5×, up to 13× on one
  track. Mean speed is largely unaffected. Diagnosis and the two-line fix are in
  [`kinematics_fix.md`](kinematics_fix.md).
- **`locomotion_time_pct` reads 100% for 72 short tracks** (median 16 frames). These are
  fragments carved out of fast bouts — they exist *because* the dog was moving — and
  should not be read as "this dog never rested".
- **Track IDs are not animal identities.** 127 IDs correspond to 3 dogs.
  [`../04_track_stitching/`](../04_track_stitching) reduces this to 39 offline.

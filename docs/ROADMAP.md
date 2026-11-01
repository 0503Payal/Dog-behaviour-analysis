# Roadmap

Where this project is going, why each step comes when it does, and what counts as done.

The ordering is driven by one finding from
[`03_tracker_diagnostics/`](../03_tracker_diagnostics): fragmentation in the current
tracking is **correlated with speed**, so analyses that need continuous identity are
biased while analyses over short windows are not. That splits the work cleanly into
what can be done now and what needs identity fixed first.

---

## Phase 1 — Offline track stitching

**Problem.** 127 track IDs for 3 dogs. Tracks break during fast motion, where motion
blur drives detection confidence from 0.90 down to ~0.72.

**Why this is tractable.** BoT-SORT runs *online*: at frame *t* it may only use frames
up to *t*. Post-hoc, the whole session is available, so a track that dies at frame 1000
can be matched against a track that is born at frame 1012 using evidence from both
sides of the gap. The gaps are mostly small — 910 bridging events with a median length
of 5 frames.

**Approach.**
- Build a cost matrix over (dying track, newborn track) pairs from the time gap, the
  predicted-versus-observed position gap, and colour-histogram similarity of the crops.
- Forbid links that would require implausible speed, or that overlap in time.
- Solve the global assignment (Hungarian), iterating until no link falls below threshold.

**Done when.** Track count is under 10, no stitched link implies a speed above a
physically plausible ceiling, and the break-speed ratio from the diagnostics falls
towards 1.

**Also in this phase.** Fix the gap-aware speed bug: divide by the true elapsed time
`gap / fps` rather than assuming one frame.

---

## Phase 2 — Unsupervised behavioural state discovery

**Goal.** Find recurring behavioural states without hand-labelling any.

**Approach.**
- Sliding windows (~1 s, overlapping) over each track.
- Per window: mean and variance of speed, turning angle, path straightness, bounding-box
  aspect ratio and area, and distance to the nearest other dog.
- Standardise, reduce with UMAP, cluster with HDBSCAN — which can leave ambiguous windows
  unassigned instead of forcing every window into a state.

**Making it explainable.** A cluster index means nothing to a biologist. Fit a shallow
decision tree on the cluster labels so each state becomes a readable rule, and report
the tree's fidelity to the clustering. A state that cannot be summarised by a simple
rule is flagged rather than named.

**Done when.** States are stable under resampling, each has a rule a biologist can read,
and the state sequence for a known segment matches what the video shows.

> Validity note: this works on the *current* data because a behavioural state is a
> short-timescale property and windows sit inside tracks. Transition matrices between
> states do **not**, and wait for Phase 1.

---

## Phase 3 — Posture classification with explainability

**Goal.** Distinguish standing / sitting / lying — a distinction kinematics alone cannot
make, since a lying dog and a standing-still dog both have zero speed.

**Approach.** Crop detections, hand-label a few hundred, fine-tune a small CNN. Class
imbalance is expected: resting dominates.

**The explainability part.** Grad-CAM attribution on the trained classifier, to check it
responds to the dog rather than the background. With three dogs in one enclosure and a
fixed camera, a classifier can learn "this corner of the floor means lying down" and
score well for the wrong reason. Attribution maps concentrated off the animal mean the
result is an artefact.

**Done when.** Held-out accuracy is reported per class, and attribution maps are shown
to sit on the animal.

**Dependency.** Needs the source video for crop extraction.

---

## Phase 4 — Validation against human scoring

**Goal.** Establish that the pipeline measures behaviour, not itself.

**Approach.** Manually score a held-out window frame by frame and compare against the
pipeline: Cohen's κ for state agreement, and a comparison of time budgets.

**Why it is a phase and not a footnote.** Everything before this produces numbers whose
correctness is assumed. This is the step that makes them evidence. It is also the step
most portfolio projects skip.

---

## Phase 5 — Interactive dashboard

**Goal.** Make the pipeline usable by someone who does not write Python.

Upload a video, run detection and tracking, show trajectories, the state timeline,
per-dog time budgets, and the diagnostics panel — so a user can see when the pipeline's
own output should not be trusted.

---

## Ideas beyond the roadmap

- **Social behaviour.** Inter-dog distance time series is already computed; proximity
  bouts and approach/avoid asymmetry follow from it once identity is stable.
- **Higher-resolution detection** to attack the blur limit directly: train at a larger
  input size, or add motion-blur augmentation, and re-measure the break-speed ratio.
- **Cross-site generalisation.** The honest test of the detector is a different
  enclosure with different dogs.

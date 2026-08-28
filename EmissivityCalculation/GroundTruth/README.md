# Ground truth for material classification

`annotate.py` records what each region actually is; `evaluate.py` scores any run
against that. Together they replace reading overlays and forming an impression.

`../Mask2Former/README.md`'s prior-on/off table was 24 regions on two frames,
read by hand. It is an anecdote — it cannot say by how much, it cannot be
repeated when the table or the crop mode changes, and it cannot compare a third
thing. These two scripts make the same question answerable in one command.

```
# once -- pin the eval frames
py annotate.py --init-frames --run ...\fullrate\material_map_m2f --n 20

# label (this is the hour of work; it is done once per segmenter)
py annotate.py --session-dir ...\ZED\20260730_161223\fullrate ^
               --run ...\fullrate\material_map_m2f ^
               --out ground_truth_m2f.csv

# score, as often as you like
py evaluate.py --ground-truth ground_truth_m2f.csv ^
               --runs ...\fullrate\material_map_m2f ...\fullrate\material_map_m2f_noprior ^
               --names prior-on prior-off
```

Runs in `C:\venvs\emissivity`, but does not need it — see `requirements.txt`.
Neither script loads a model.

## The one rule: region ids are positional

A ground-truth row says *"frame `20250906_233144_R`, region 3 is glass"*. Region
ids are handed out by connected-component order, so that row is only true for a
run whose `labels.npy` puts region 3 on the same pixels. Point it at a different
segmentation and the label silently describes a different object — and you get a
plausible accuracy number that means nothing.

Measured on frame `20250906_233144_R`, of the four runs on disk:

| pair | same `labels.npy`? |
|---|---|
| `material_map_sam` vs `material_map_consensus` | **yes** — consensus re-votes materials over SAM's segmentation |
| every other pair | no |

So annotation is **per segmenter, not per run**:

- **prior on vs off** — one pass covers both. Segmentation happens before and
  independently of the prior (`classify_session_m2f.py:405` vs `:440`;
  `classify_session.py:303` vs `:311`).
- **SAM vs Mask2Former** — two passes, over *the same photographs*. That is why
  the frame set is pinned in `eval_frames.txt` rather than sampled per run.

None of this has to be remembered. `evaluate.py` records the `labels.npy` md5 at
annotation time and refuses any run that does not match.

## `eval_frames.txt`

20 FLIR stems, chosen as `sorted(stems)[::5][:20]` — a **stride, not an even
spread**. Neither classifier has a `--frames` flag; they can only subset with
`--every-n` and `--limit`. A stride-5 set of 20 is therefore reproducible when
the runs are regenerated:

```
--every-n 5 --limit 20
```

Verified: that expression over `sync_manifest.json`'s triplets reproduces
`eval_frames.txt` exactly.

## Annotating

The window shows the region zoomed on the left — surroundings dimmed rather
than blacked out, because a thin window reveal is only identifiable from what is
beside it — and the whole frame on the right with the region tinted, because a
crop alone loses *which* object you are looking at.

| key | |
|---|---|
| `1`–`0` | paint, plaster, concrete, brick, wood, glass, ceramic, plastic, rubber, fabric |
| `a`–`c` | painted_metal, cardboard, asphalt |
| `d`–`j` | the bare metals |
| `m` | **mixed** — the region genuinely spans two materials |
| `u` | **unclear** — too small, dark or blurred to judge |
| `←` `→` | back / skip without labelling |
| `Esc` | quit (saved) |

The key map is built from `../emissivity_table.csv` and drawn in the window, so
a table edit cannot leave the legend disagreeing with the bindings, and no
material can become unbindable.

`mixed` and `unclear` are recorded and counted, but excluded from every metric.
A region with no right answer must not be able to score as a classifier error.
`--min-area` records anything smaller as `unclear` without showing it —
`material_map_sam`'s smallest region on this session is **3 px**.

Saved after **every single label**, via temp-file-then-replace. An hour of work
must not depend on a clean exit, and a process killed mid-append leaves a
truncated line that `csv` reads as a valid short row.

### The prediction is hidden

The run's own answer is **not** shown while labelling. If the screen says
`glass 0.94` while you decide, you will agree with it more often than you
should, and a ground truth anchored on the thing it is meant to judge measures
nothing. `--show-prediction` exists for reviewing a finished pass.

### `ground_truth.csv`

```
session, run_used, frame, region_id, area_px, ade, status, ground_truth, labels_md5, annotated_utc
```

`status` is `labeled` / `mixed` / `unclear`; `ground_truth` holds a material from
`../emissivity_table.csv` when `labeled`. `labels_md5` is the fingerprint of that
frame's segmentation at annotation time — kept per row rather than in a sidecar
so the file cannot be separated from its provenance. Pointing a second pass at
an existing file annotated on a different run is refused.

## Scoring

`evaluate.py` prints a table with the runs as columns, writes `comparison.csv`
with the same content, and one `confusion_<name>.csv` per run.

**The compatibility gate runs first and is fatal, with no `--force`.** Scoring a
run against labels drawn on a different segmentation produces a wrong number
that looks right, which is worse than no number. On mismatch it prints, per
frame, both `n_segments`, both crop boxes and both `source_zed_frame` values:

```
material_map_sam was segmented differently from the run the ground truth was
annotated on -- 20/20 frame(s) differ. ...

  20250906_233144_R
    this run : n_segments=28  crop=(624, 178, 1375, 911) source_zed_frame=right_000148.png
    annotated: n_segments=12  crop=(515, 67, 1494, 1019) source_zed_frame=right_000149.png
```

That is not a hypothetical. **`material_map_sam` and `material_map` on this
session are stale and must not be used as a baseline:**

- generated 2026-08-07, before commit `a664518` *"Fix ZED intrinsics in
  rig_calibration.yaml: use MATLAB result, not stale 1280x720"* (2026-08-25).
  The intrinsics set the FLIR-FOV box, so theirs is 751×733 where the corrected
  one is 979×952.
- also before the sync manifest was regenerated, so they classified
  `right_000148.png` where the m2f run classified `right_000149.png` — **100 of
  107 frames disagree** on which photograph was used.

Nothing about them looks wrong from the outside. Regenerate both SAM runs
against the current calibration before comparing anything to them:

```
py ..\classify_session.py --session-dir ...\fullrate --every-n 5 --limit 20 ^
    --calibration ..\..\SensorFusionLoader\rig_calibration.yaml --out-dir ...\material_map_sam_v2
```

(`classify_session.py:253` hardcodes `../Calibration/rig_calibration.yaml`,
which does not exist in this repo — hence the explicit `--calibration`.)

### Read the emissivity error, not only the accuracy

Accuracy counts every mistake once. The pipeline does not — it divides by ε, and
every non-metal in the table sits between 0.90 and 0.95. So the report also
gives mean |Δε|, its area-weighted version, how many regions are wrong by more
than 0.10, and the temperature error that implies.

From the first real 20-frame ground truth:

| | prior-on | prior-off |
|---|---|---|
| accuracy | 0.696 | 0.522 |
| regions with \|Δε\| > 0.10 | **0** | **12** |
| worst \|ΔT\| | **3 K** | **316 K** |

17 accuracy points understates it. What the prior actually removes is
`windowpane → aluminum_polished` (ε 0.05) and `windowpane → steel_polished`
(ε 0.07) — both of which passed the low-emissivity gate at confidence 0.50 and
0.54, just over its 0.50 threshold, plus nine `steel_oxidized` (ε 0.79) that sit
above the gate's cutoff and that it cannot see at all. The prior makes them
impossible by construction; the gate demonstrably does not catch them.

ΔT is the naive graybody bound, `(ε_true/ε_pred)^¼` at 22 °C. It **ignores
reflected ambient radiation**, which cancels much of the error in a corridor
where walls, air and objects are at similar temperatures —
`../RadiometricCalibration/correct_session.py` does model it. Use ΔT to rank
runs, not to predict the pipeline's output.

### Macro-averaging, and its trap

The macro average is over **classes present in the ground truth**, not all 20 in
`../emissivity_table.csv`. Most of the table never occurs indoors here, and a
macro over 20 classes would be dominated by classes with no support.

The cost of that choice: predictions naming a class that occurs nowhere in the
ground truth are all wrong, but **none of them enter macro precision**. On the
real run this makes macro precision 0.881 against an accuracy of 0.696. The
report prints `predictions outside GT cls` as its own row and warns explicitly
when the gap opens. **Read accuracy and macro F1, not macro precision.**

Precision/recall/F1 are computed with numpy rather than sklearn — a dozen lines,
against a new dependency. Zero denominators give 0.0, the same convention as
sklearn's `zero_division=0`.

## Deliberately not included

Ask if you want either; they were left out rather than silently added.

- **Plots** and **per-frame breakdowns**.
- **Area-weighted accuracy.** Region-count accuracy treats a 1,728 px sliver and
  a 205,719 px wall as equal. |Δε| is reported area-weighted, but accuracy is
  not.

## Layout

```
GroundTruth/
  gt_common.py        schema reading, labels_md5, the compatibility gate
  annotate.py         Tkinter annotator; --init-frames pins eval_frames.txt
  evaluate.py         scoring, confusion matrices, the comparison table
  eval_frames.txt     the 20 pinned frames
  requirements.txt
```

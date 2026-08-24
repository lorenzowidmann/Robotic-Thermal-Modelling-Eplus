# LivoxLidarOdometryLoader

Rebuilds full-FOV world clouds from a rosbag's **raw** `/livox/lidar` topic and
its `/Odometry`, bypassing FAST-LIO's `/cloud_registered` — which on session 9
throws away 92% of what the sensor recorded.

Drop-in for `EmissivityCalculation/project_to_flir.py::nearest_clouds_for_targets`.

## The problem

Measured on `rosbag2_2026_07_30-18_12_20`, first triplet:

| | raw `/livox/lidar` | FAST-LIO `/cloud_registered` |
|---|---|---|
| points / message | **83 096** | 6 465 (−92%) |
| azimuth | −59.5° … +60.8° | **−17.2° … +17.4°** |
| elevation | −13.4° … +13.2° | −4.3° … +8.7° |
| min range | **1.02 m** | **4.05 m** |

Projected into the ZED frame, `/cloud_registered` covers u ∈ [666, 1309],
v ∈ [369, 628] — about **8% of the frame** — and *all* 6465 points already land
inside the image, so the camera clips nothing. The cloud arrives pre-cropped.
The 4.05 m floor looks like FAST-LIO's `preprocess/blind`, the ±17° cone like
`mapping/fov_degree`. The sensor is a Livox **HAP** (120° × 25°), not a Mid-360.

This starves anything that votes per LiDAR point.
`WindowsDoorsDetection/opening_voxel_consensus.py` produced **0 opening voxels**
because both window bays sat outside the cone.

## What it does

`/Odometry`'s pose maps lidar-local straight to world — no lidar→body extrinsic
to compose. Verified, not assumed: transforming a raw scan by the manifest pose
puts `/cloud_registered`'s own points 4.9 cm away at the median, 93.6% within
10 cm. Read that as agreement, not as an error bar — it is a nearest-neighbour
distance into a cloud 13× denser, so it mostly measures surface sampling.

Each point is transformed by the pose interpolated at **its own** timestamp. A
CustomMsg spans **200 ms** here, not 100. On session 9's opening frames that is
worth almost nothing (the rover is nearly still — 0.6 cm between poses, and
deskew on-vs-off moves points by 0.2 cm median); it matters on the walking
segments. `--no-deskew` to compare.

## Measured effect

Coverage of the kept opening regions, frame `20250906_233144_R`:

```
                       in frame   footprint            window14   window15
/cloud_registered          6 465  u 666..1309              0          0
rebuilt raw               77 894  u   0..1920          5 833      5 897
```

Both bays go from **zero returns** to ~5 800 each, at 1.6–5.3 m. Full frame
width recovered.

### This disproves a claim WindowsDoorsDetection was built on

That module's README has long stated *"glazing returns no LiDAR — 11 of 19
opening detections got ZERO points, including the 315×735 px glass wall at conf
0.98–0.99."* It is the stated reason metric checks are kept off windows and the
metric gate lives at stage 2.

It was measured on the cropped cloud, which cannot tell glass apart from
out-of-footprint. On the rebuilt cloud those same bays return **thousands of
points at 1.6–2.5 m**. The premise is false as stated; the window rules that
rest on it need re-deriving.

## Usage

```
:: what is actually in the bag
py livox_odometry_loader.py --bag ...\rosbag2_2026_07_30-18_12_20 --inspect

:: rebuild the clouds a session's manifest asks for, and check them
py livox_odometry_loader.py --bag ... --session-dir ...\fullrate --limit 5 --compare

:: dump one for CloudCompare
py livox_odometry_loader.py --bag ... --session-dir ...\fullrate --limit 1 --export-ply scan0.ply
```

As a library:

```python
import livox_odometry_loader as lol
clouds = lol.nearest_clouds_for_targets(bag, target_epochs)   # [(t, points_world) | None]
```

Options: `--min-range`, `--max-range`, `--point-filter-num` (decimate),
`--no-deskew`, `--lidar-topic`, `--odom-topic`, `--store`.

## `--fix-digests`: reading a bag MATLAB wrote

```
py livox_odometry_loader.py --bag ...\SavedBag_Odometry\<stamp> --fix-digests
```

`3DModelPointCloudExtraction/BagFilter*.m` export their drift-corrected
trajectory as a rosbag2 so this loader can read it. MATLAB's `ros2bagwriter`
writes the message definition but leaves `type_description_hash` **empty**, and
`rosbags` asserts that digest against one it recomputes from the definition —
so the bag fails to open with

```
AssertionError: Failed to parse nav_msgs/msg/Odometry
```

before a single message is read. Nothing is wrong with the data; only the hash
is absent. `--fix-digests` recomputes it exactly as `rosbags` would and writes
it back, to the sqlite tables and to `metadata.yaml`. It fills **only** empty
digests, so it is idempotent and cannot paper over a genuine type mismatch.
Run once per bag.

Measured on a synthetic 10 s / 101-pose trajectory carrying a deliberate 0.8°
yaw drift: before, the bag does not open at all; after, 101 poses read at the
full 10 Hz and the injected drift comes back out at 0.35 mm.

**Drop-in gotcha.** The third positional is the *cloud* topic, which here is
`/livox/lidar`; the odometry topic is a separate keyword. Both
WindowsDoorsDetection stages call their cloud-topic argument `--odom-topic`
(a pre-existing misnomer), so wiring this in means passing `/livox/lidar`
there.

## Performance

Two passes over the bag. The first reads only `header.stamp` out of the CDR
bytes to pick which messages are wanted; the second deserialises just those,
and reads the point array with `np.frombuffer` on a 20-byte packed struct
rather than letting rosbags materialise 90 000 Python objects per message
(~128 ms → ~1 ms). 5 targets in **4.1 s**, most of it bag I/O.

The struct offsets are verified per message against the sequence-length field
and fall back to full deserialisation on mismatch — a driver that changes the
layout gets slow, not wrong.

## Caveat

Even at full FOV the HAP's **25° vertical** against the ZED's 54° means the
LiDAR reaches roughly the middle half of the image height (measured: v ∈ [275,
898] of 1080). A `windowpane` mask spanning the full frame height still cannot
be measured over its whole extent — only over the part the laser reaches.

## Requirements

`rosbags`, `numpy`. No scipy — the slerp is written out so this imports in the
consensus venv too.

# PointCloudFilterGUI

Tkinter GUI: load a rosbag2 point cloud, tune filters, preview two ways, save
the filtered cloud as a new rosbag2 `.db3`.

Same filters as [`PointCloudView/view_pointcloud.py`](../PointCloudView/view_pointcloud.py)
(SOR, declutter, voxel downsample), plus the ROI crop from
[`MATLAB_PointCloudVisualization/ROS2_PointVisualization.m`](../../MATLAB_PointCloudVisualization/ROS2_PointVisualization.m).
Pipeline order, same as those:

```
read all frames -> merge -> ROI crop -> SOR -> declutter -> voxel downsample
```

## Requirements

```
pip install -r requirements.txt
```

`C:\venvs\planefit` already has everything (`rosbags`, `numpy`, `scipy`,
`pyvista`, `matplotlib`, `tkinter`) — you can point at it directly instead of
making a new venv:

```
C:\venvs\planefit\Scripts\python.exe filter_gui.py
```

## Usage

```
python filter_gui.py [bag_folder]
```

`bag_folder` (optional) preloads a rosbag2 folder path into the "Bag" field —
you still need to click **Load**. Or leave it empty and use **Browse...**.

1. **Browse...** → pick the rosbag2 folder (the one with `metadata.yaml`,
   e.g. `rosbag2_2026_07_30-18_02_41`), not the `.db3` file itself.
2. **Load** — reads and merges every frame on `--topic` (default
   `/cloud_registered`). Can take a while on multi-GB bags (session 8's
   `.db3` is ~3 GB); the GUI stays responsive (loading runs in a background
   thread), progress prints in the log panel.
3. Set filter parameters (ROI / SOR / declutter / voxel — see table below),
   click **Apply filters / Preview**. Filtering runs in a background thread
   (log panel shows progress), then the **embedded view** (right panel)
   updates immediately — quick CPU-rendered glance, capped at "embedded max
   pts" (default 30,000; matplotlib has no GPU accel, so rotating gets
   sluggish well past that).
4. Want to actually rotate/inspect closely, or read off metric coordinates to
   pick a cut? Click **"Open smooth 3D view (PyVista)..."** — opens the
   full-resolution filtered cloud in a *separate process* with a
   GPU-accelerated VTK renderer (same one `view_pointcloud.py` uses) plus a
   metre-scale grid on every axis. Doesn't block the control panel — keep
   tuning params and re-opening the preview. Opening a new one **closes the
   previous PyVista window** automatically (one at a time, no pile-up).
   Capped by
   "pyvista max pts" (default 500,000, mostly a backstop for huge raw
   clouds — VTK handles far more than matplotlib comfortably).
5. Happy with it → **Save filtered bag...**, pick a folder name that doesn't
   already exist. Writes a new rosbag2 folder (`metadata.yaml` + `.db3`) with
   the **merged, filtered cloud as a single `PointCloud2` message** (xyz
   only, float32) on the same topic — openable by `view_pointcloud.py`,
   `fit_planes.py`, or the MATLAB viewers exactly like a normal bag (they all
   merge every frame into one cloud anyway).

## Why one output frame instead of per-frame filtering

Voxel downsampling and declutter operate on the *merged* cloud, mixing points
from different original frames into the same kept/dropped decision — there is
no consistent way to split that back into the original per-frame message
sequence. The saved bag is one baked-down `PointCloud2` message instead of
pretending to preserve frame-by-frame structure it can't actually keep.
Intensity/normal/curvature fields present in the source (e.g. FAST-LIO's
`/cloud_registered`) are dropped — none of the existing tools that read these
bags (`view_pointcloud.py`, `fit_planes.py`, the MATLAB scripts) use anything
but x/y/z.

## Filters

| Section | Param | Meaning |
|---|---|---|
| ROI crop | xmin/xmax/ymin/ymax/zmin/zmax | Keep points inside this box only. `inf`/`-inf` allowed (leave a side unbounded). Off by default. |
| SOR | k | Neighbours per point for outlier scoring (default 16). |
| | std | Std-dev multiplier, lower = more aggressive (default 1.5). |
| Declutter | cluster-gap | Max gap (m) for points to count as one cluster (default 0.30). |
| | min-cluster | Keep every cluster with ≥ N points (0 = off, keep largest only). |
| | cluster-dist | Keep clusters within this distance (m) of the main cloud (0 = off). |
| Voxel | size | Voxel edge (m) to downsample density before saving (0 = off). |
| Preview | embedded max pts | Cap on points drawn in the embedded matplotlib view. |
| | pyvista max pts | Cap on points drawn in the PyVista window (0 = no cap). |

## Gotchas

- **Save** target must not already exist — `rosbags.Writer` creates the
  folder itself; pick a fresh name (the dialog defaults to
  `<bag_name>_filtered`).
- The PyVista window runs in its own subprocess (not embedded in the Tk
  panel — pyvista has no official Tkinter-embeddable widget, and nesting its
  window loop in-process is what caused an earlier "window doesn't appear"
  bug this design avoids). The control panel stays usable while it's open.
- Declutter with no `min-cluster`/`cluster-dist` keeps **only the largest**
  cluster, same default as `view_pointcloud.py`.
- Without `scipy` installed, SOR falls back to a cruder voxel-density filter
  (same fallback as `view_pointcloud.py`) — install scipy for the real thing.

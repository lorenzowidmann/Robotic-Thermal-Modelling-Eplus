# MATLAB_PointCloudVisualization

MATLAB viewers for a raw LiDAR rosbag, used for a first look at a recording
before anything is processed in Python. Requires the Computer Vision Toolbox
and the ROS Toolbox.

| file | what it does |
|---|---|
| `ROS2_PointVisualization.m` | reads a rosbag2 folder, merges the frames and applies the standard filter chain — ROI crop → statistical outlier removal → declutter → voxel downsample — then displays the result. The ROI crop here is the one `../PointCloudElaboration/PointCloudFilterGUI/filter_gui.py` reimplements in Python. |
| `ROS2_PointVisualization_NoFilters.m` | same, with every filter off — the unfiltered cloud as recorded. |
| `ROS1_PointVisualization.m` | ROS 1 (`.bag`) equivalent, kept for older recordings. |

These are inspection tools only; nothing downstream consumes their output. The
filtered cloud the pipeline actually runs on is produced by
`../PointCloudElaboration/PointCloudFilterGUI/filter_gui.py`, which writes a new
rosbag2 `.db3`.

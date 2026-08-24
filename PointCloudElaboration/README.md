# PointCloudElaboration

Everything that happens to the LiDAR point cloud between the raw recording and
the OpenStudio model. Each subfolder is a step with its own README, venv and
`requirements.txt`.

Run in this order:

| # | folder | what it produces |
|---|---|---|
| 1 | **`LivoxLidarOdometryLoader/`** | reads the Livox/FAST-LIO rosbag2 and its `/Odometry`, and fixes MATLAB-written bags whose `type_description_hash` is empty (`--fix-digests`) so Python's `rosbags` can open them at all. |
| 2 | **`PointCloudFilterGUI/`** | interactive filtering with live preview — ROI crop → statistical outlier removal → declutter → voxel downsample — saved as a new rosbag2 `.db3`. **This produces the `_filtered` bag every later step starts from.** |
| 3 | **`OcTreeVoxel/`** | `fit_closed_planes.py` fits the closed 6-plane box of the corridor; `aligned_octree.py` levels the cloud into that building frame and voxelises it. Outputs `voxels.npz`, `transform.json`, `planes_aligned.json`. |
| 4 | **`WindowsDoorsDetection/`** | finds windows and doors: ZED segmentation masks projected onto the cloud via LiDAR, voted per voxel, regularised into rectangles → `openings.json`, consumed by `../3DModelPointCloudExtraction/to_openstudio.py --openings`. |
| 5 | **`SolarIrradianceCorrection/`** | per-voxel U-value with the solar gain handled by sol-air on the exterior walls → `thermal_voxels_u_solair.csv`, consumed by `../3DModelPointCloudExtraction/assign_u_to_osm.py`. Also holds `Vostok/`, an unconnected shadow-mask branch. |

Steps 4 and 5 both need the thermal/material side of the pipeline to have run
first — see `../RadiometricCalibration/` and `../EmissivityCalculation/`.

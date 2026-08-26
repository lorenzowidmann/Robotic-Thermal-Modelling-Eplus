# LVTCalibConversion

Tools that get recorded sensor data (Livox LiDAR ROS2 bag, ZED frames,
FLIR RJPG frames) into a form **LVT2Calib** (ROS1/Noetic, runs in the
`lvt2calib_gui` container) can actually consume.

## Contents

| Folder | What it is | Runs where |
|---|---|---|
| `convert_livox_bag.py` (+ `requirements.txt`) | Standalone Python script. Converts a ROS2 bag with `livox_ros_driver2/msg/CustomMsg` into a ROS1 `.bag`, remapping the message namespace to `livox_ros_driver` (v1) so `koide3/livox_to_pointcloud2`'s ROS1 branch accepts it. | Windows host, plain `pip` venv |
| `zed_frame_publisher/` | ROS1 catkin package. Publishes ZED 2i frames recorded by `zed_record.py` (PNG session + `metadata.json`) as `sensor_msgs/Image`, for LVT2Calib's RGB `cam_pattern` node. | Inside `lvt2calib_gui` container |
| `flir_frame_publisher/` | ROS1 catkin package. Publishes FLIR radiometric JPEG (RJPG) frames as `sensor_msgs/Image`, for LVT2Calib's thermal `cam_pattern` node. Sibling of `zed_frame_publisher`. | Inside `lvt2calib_gui` container |
| `livox_hap_pattern.launch` | Launch file adding **Livox HAP** support to LVT2Calib, which upstream does not ship. Copy into `lvt2calib/launch/lidar/livox/`. | Inside `lvt2calib_gui` container |

Each has its own README with full details (flags, defaults, why they were
chosen against the lvt2calib source); this file is the map between them.

## Why these exist

LVT2Calib expects live ROS1 topics (camera images + LiDAR cloud), not files.
None of our raw captures are already in that form:

- The Livox LiDAR is recorded as a **ROS2** bag (HAP driver) → `convert_livox_bag.py`
  turns it into a ROS1 bag so it can be played back and converted to
  `PointCloud2` inside ROS1 (`rosrun livox_to_pointcloud2 livox_to_pointcloud2_node`).
- ZED and FLIR are recorded to **plain files** (PNG/mp4 + RJPG), not bags →
  `zed_frame_publisher` / `flir_frame_publisher` replay those files as live
  `sensor_msgs/Image` topics that LVT2Calib's `cam_pattern` nodes subscribe to.
- The **HAP is not an upstream-supported LiDAR** in LVT2Calib (only Horizon,
  Mid-70, Mid-40 and Avia are) → `livox_hap_pattern.launch` adds it. It is the
  Horizon wrapper with `ns_=livox_hap` and `cloud_tp=/livox/points`; the Horizon
  wrapper carries no hardware-specific parameters, so the copy is safe. The input
  topic is `/livox/points` (`PointCloud2`, out of `livox_to_pointcloud2`), **not**
  `/livox/lidar`. `livox_pattern.launch` also loads `rviz/$(arg ns_)_pattern.rviz`,
  so a matching `livox_hap_pattern.rviz` must exist there or the rviz node errors.

## Typical flow

```
1) Livox bag (ROS2)  --convert_livox_bag.py-->  ROS1 bag
                                                   |
                                                   v (inside container)
                                        rosbag play + livox_to_pointcloud2_node
                                                   |
2) ZED session (PNG+metadata.json) --zed_frame_publisher-->  /zed_right/image_raw
3) FLIR RJPG folder                --flir_frame_publisher--> /thermal_cam/thermal_image
                                                   |
                                                   v
                                    lvt2calib rgb_cam_pattern.launch /
                                    lvt2calib thermal_cam_pattern.launch /
                                    pattern_collection_lc (laser<->cam calibration)
```

## Setup

**`convert_livox_bag.py`** — runs on the host, plain pip:
```
pip install -r requirements.txt
py convert_livox_bag.py --src C:\path\to\ros2_bag_dir --dst C:\path\to\output_ros1.bag
```
See `convert_livox_bag.py`'s own docstring / `--help` for all options.

**`zed_frame_publisher/` and `flir_frame_publisher/`** — ROS1 catkin packages,
built and run inside the `lvt2calib_gui` container (they need `rospy`,
`cv_bridge`, ROS message types):
```bash
# from the Windows host (this folder's parent)
docker cp zed_frame_publisher  lvt2calib_gui:/home/catkin_ws/src/
docker cp flir_frame_publisher lvt2calib_gui:/home/catkin_ws/src/

# inside the container
docker exec -it lvt2calib_gui bash
  cd /home/catkin_ws
  catkin build zed_frame_publisher flir_frame_publisher
  source devel/setup.bash
```
Then run each with `rosrun`/`roslaunch` as described in their own READMEs
(`zed_frame_publisher/README.md`, `flir_frame_publisher/README.md`).

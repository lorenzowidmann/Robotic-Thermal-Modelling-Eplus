# zed_frame_publisher

ROS1 (Noetic) node that publishes ZED 2i frames recorded by
`SensorFusion/zed_record.py` (UVC backend) as `sensor_msgs/Image` (`bgr8`),
so **LVT2Calib**'s `cam_pattern` node can consume them (it wants a raw ROS
image topic, not files).

## Why these defaults (verified against lvt2calib source)

- **`sensor_msgs/Image` raw, `bgr8`** — `cam_pattern.cpp` subscribes with
  `image_transport` and does `cv_bridge::toCvShare(msg, "bgr8")`. PNGs from
  `cv2.imwrite` are already BGR → no channel conversion.
- **No CameraInfo topic needed** — `cam_pattern` reads intrinsics from a FILE
  (`data/camera_info/<cam_info_filename>`), not a topic. `--camera-info-file`
  here is optional (rviz/other consumers only).
- **`--stamp-mode absolute` default** — `pattern_collection_lc.cpp` uses two
  independent subscribers (no `ApproximateTimeSynchronizer`), so it does NOT
  match camera↔laser by `header.stamp`. Only co-liveness over a static target
  matters. `absolute` (= `started_utc + t_offset_s`) is kept for traceability
  vs the LiDAR clock; `now` also works.
- **`frames` source default** — PNGs carry an explicit per-frame `t_offset_s`;
  the mp4 timestamp would only be inferred from fps. `--source mp4` available.

## Deploy into the running container (no mount / image rebuild)

Container: `lvt2calib_gui`, workspace `/home/catkin_ws`.

```bash
# from the Windows host (this folder's parent = ...\ClaudeCode)
docker cp zed_frame_publisher lvt2calib_gui:/home/catkin_ws/src/

# inside the container
docker exec -it lvt2calib_gui bash
  cd /home/catkin_ws
  catkin build zed_frame_publisher     # or: catkin_make
  source devel/setup.bash
```

Bring a recorded session in via the already-mounted `/data/bags`
(`C:\Users\loren\Desktop\SLAM` on the host): copy the session folder there, e.g.
`/data/bags/zed_session/` (must contain `metadata.json`, `frames/`).

## Run

```bash
# 1) publish the ZED frames
rosrun zed_frame_publisher zed_frame_publisher.py \
    --session-dir /data/bags/zed_session --loop
# or:
roslaunch zed_frame_publisher zed_frame_publisher.launch \
    session_dir:=/data/bags/zed_session

# 2) LVT2Calib camera detection, pointed at our topic
roslaunch lvt2calib rgb_cam_pattern.launch \
    image_tp:=/zed_right/image_raw cam_info_filename:=zed_right_intrinsic.yaml
```

Put the ZED intrinsics YAML at
`/home/catkin_ws/src/lvt2calib/data/camera_info/zed_right_intrinsic.yaml`
(that is where LVT2Calib actually loads them from).

## Flags

| flag | default | meaning |
|---|---|---|
| `--session-dir` | (required) | zed_record.py session folder |
| `--source` | `frames` | `frames` (t_offset_s) or `mp4` (t from fps) |
| `--stamp-mode` | `absolute` | `absolute` = started_utc+t_offset_s, or `now` |
| `--frame-id` | `zed_right` | header.frame_id |
| `--image-topic` | `/zed_right/image_raw` | image topic (= cam_pattern `image_tp`) |
| `--rate-multiplier` | `1.0` | >1 faster, <1 slower, <=0 no sleep |
| `--loop` | off | republish until shutdown |
| `--camera-info-file` | (none) | optional CameraInfo YAML (not used by LVT2Calib) |

## Empty test (no target in scene)

The node should start, publish on `/zed_right/image_raw`, and `cam_pattern`
should subscribe and warn it finds no circles. Verify:

```bash
rostopic info /zed_right/image_raw    # Type: sensor_msgs/Image; Subscribers: /rgb/cam_pattern
rostopic hz   /zed_right/image_raw    # ~fps of the session
rosnode info  /zed_frame_publisher
```

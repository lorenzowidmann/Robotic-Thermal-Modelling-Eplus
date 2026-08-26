# flir_frame_publisher

ROS1 (Noetic) node that publishes FLIR thermal frames (radiometric JPEG /
**RJPG**) from a folder as `sensor_msgs/Image` (`bgr8`), so **LVT2Calib**'s
thermal `cam_pattern` node (`isRGB=false`) can consume them. Sibling of
`zed_frame_publisher`, adapted to RJPG input.

## Why these defaults (verified against lvt2calib source)

- **`sensor_msgs/Image` raw, `bgr8`** — `cam_pattern.cpp` subscribes with
  `image_transport` and does `cv_bridge::toCvShare(msg, "bgr8")`. For thermal
  (`is_rgb=false`) it uses the image directly (no gray conversion) for
  `findCirclesGrid` + `SimpleBlobDetector`. → publish bgr8 (3 channels).
- **Default topic `/thermal_cam/thermal_image`** — matches LVT2Calib's thermal
  default, so `thermal_cam_pattern.launch` needs no `image_tp` override.
- **`isDarkBoard`** decides blob polarity: `false` (default) = dark holes on a
  light board (`blobColor=0`); `true` = light holes on dark board
  (`blobColor=255`). Pick per how the holes look in the thermal image.
- **Intrinsics from FILE** — LVT2Calib reads `data/camera_info/<cam_info_filename>`
  (used for `cv::remap` undistort). Put your `thermal_intrinsic.yaml` there.
  The CameraInfo topic is NOT used; `--camera-info-file` here is optional.
- **`--stamp-mode now` default** — no `ApproximateTimeSynchronizer` in
  `pattern_collection_lc.cpp`, so stamps don't pair camera↔laser. FLIR EXIF
  times come from the camera RTC (not the rover clock), so they are NOT
  comparable to the LiDAR stamps — don't rely on `exif` for clock alignment.

## Image modes (`--image-mode`)

- **`embedded`** (default): `cv2.imread()` reads the FLIR palette-rendered JPEG
  embedded in the RJPG. No external tools.
- **`raw`**: extract the 16-bit `RawThermalImage` via **`exiftool`**, normalize
  min–max to 8-bit, optional `--colormap` (none/gray/jet/inferno/hot/magma).
  Cleaner contrast for blob detection. Needs `exiftool` (`apt-get install
  libimage-exiftool-perl`). If values look inverted/garbled, try `--raw-byteswap`.

## Deploy into the running container

```bash
# host Windows, from ...\ClaudeCode
docker cp flir_frame_publisher lvt2calib_gui:/home/catkin_ws/src/

# inside the container
docker exec -it lvt2calib_gui bash
  cd /home/catkin_ws && catkin build flir_frame_publisher && source devel/setup.bash
```

Bring the FLIR RJPG folder in via the mounted `/data/bags`
(`C:\Users\loren\Desktop\SLAM`): copy it there, e.g. `/data/bags/flir_session/`.

## Run + detection

```bash
# 1) publish thermal frames
rosrun flir_frame_publisher flir_frame_publisher.py \
    --image-dir /data/bags/flir_session --image-mode embedded --loop
#   raw mode:  --image-mode raw --colormap none   (needs exiftool)

# 2) LVT2Calib thermal detection (default topic already matches)
roslaunch lvt2calib thermal_cam_pattern.launch \
    cam_info_filename:=thermal_intrinsic.yaml
#   if holes appear bright on a dark board:  isDarkBoard:=true  (edit launch/arg)
```

Put the thermal intrinsics at
`/home/catkin_ws/src/lvt2calib/data/camera_info/thermal_intrinsic.yaml`
(you already have it in `Thesis/Calibration/`).

## Flags

| flag | default | meaning |
|---|---|---|
| `--image-dir` | (required) | folder of RJPG/JPG |
| `--pattern` | (all jpg/jpeg/rjpg) | optional glob, e.g. `FLIR*.jpg` |
| `--image-mode` | `embedded` | `embedded` or `raw` (exiftool) |
| `--colormap` | `none` | raw mode: none/gray/jet/inferno/hot/magma |
| `--raw-byteswap` | off | byte-swap 16-bit raw if garbled |
| `--exiftool` | `exiftool` | exiftool path (raw / exif modes) |
| `--fps` | `5.0` | playback rate (RJPG has no per-frame offset) |
| `--rate-multiplier` | `1.0` | >1 faster, <1 slower, <=0 no sleep |
| `--stamp-mode` | `now` | `now` / `sequential` / `exif` (camera RTC) |
| `--frame-id` | `flir_thermal` | header.frame_id |
| `--image-topic` | `/thermal_cam/thermal_image` | = cam_pattern `image_tp` |
| `--loop` | off | republish until shutdown |
| `--camera-info-file` | (none) | optional CameraInfo (not used by LVT2Calib) |

## Empty test

```bash
rostopic info /thermal_cam/thermal_image   # Type: sensor_msgs/Image; Subscribers: /thermal/cam_pattern
rostopic hz   /thermal_cam/thermal_image
rosnode info  /flir_frame_publisher
```
The node publishes, `cam_pattern` subscribes and warns it finds no circles
(no target). No crash = OK.

## What to confirm on real data
- **`embedded` vs `raw`**: try `embedded` first; if the holes don't show enough
  contrast for the blob detector, switch to `raw`.
- **`isDarkBoard`**: set per thermal hole appearance (dark vs bright holes).
- **`thermal_intrinsic.yaml`** resolution must match the published image size
  (embedded JPEG vs raw radiometric may differ in resolution!).

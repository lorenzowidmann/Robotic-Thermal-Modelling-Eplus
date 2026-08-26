#!/usr/bin/env python3
"""Publish ZED 2i frames recorded by SensorFusion/zed_record.py as ROS1
sensor_msgs/Image, for LVT2Calib's camera pattern node.

Source of truth is a zed_record.py session folder:

    <session>/
      frames/right_000000.png   <- PNG per frame (default source)
      session_right.mp4         <- fallback (--source mp4)
      metadata.json             <- schema "zed_record/v1"

Each PNG carries an explicit t_offset_s (seconds from session start,
time.monotonic()); the absolute capture instant is
    epoch_frame = started_utc + t_offset_s
with started_utc taken (microsecond precision) from the rover system clock,
the same clock the LiDAR stamps its ROS2 headers on.

LVT2Calib pairing note: pattern_collection_lc.cpp uses two INDEPENDENT
subscribers (cloud_laser / cloud_cam), NOT an ApproximateTimeSynchronizer, so
it does not match camera<->laser by header.stamp. For calibration only
co-liveness of the two streams over a static target matters; --stamp-mode is
therefore harmless either way. Default 'absolute' keeps stamps traceable
against the LiDAR clock for any offline analysis.

The image node in LVT2Calib (cam_pattern.cpp) subscribes via image_transport
to a raw sensor_msgs/Image and does cv_bridge::toCvShare(msg, "bgr8"); the PNGs
are already BGR (cv2.imwrite), so we publish bgr8 with no channel conversion.
LVT2Calib reads intrinsics from a FILE (data/camera_info/<name>), not from a
CameraInfo topic -- the optional --camera-info-file publisher here is only for
rviz/other consumers.
"""

import os
import sys
import json
import argparse

from datetime import datetime, timezone

import rospy
import cv2
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, CameraInfo

SCHEMA = "zed_record/v1"


def parse_iso_epoch(ts):
    """ISO-8601 (…Z, microsecond) -> UTC epoch seconds. fromisoformat handles
    the fractional seconds written by zed_record.utc_now_iso()."""
    return datetime.fromisoformat(ts.replace("Z", "+00:00")).timestamp()


def load_metadata(session_dir):
    meta_path = os.path.join(session_dir, "metadata.json")
    if not os.path.isfile(meta_path):
        raise SystemExit(
            "No metadata.json in %s -- expected a zed_record.py session folder."
            % session_dir
        )
    with open(meta_path, "r") as f:
        meta = json.load(f)
    schema = meta.get("schema")
    if schema != SCHEMA:
        raise SystemExit(
            "Unexpected metadata schema %r (expected %r): %s"
            % (schema, SCHEMA, meta_path)
        )
    return meta


def load_camera_info(path, frame_id):
    """Build a CameraInfo from a ROS camera_calibration-style YAML
    (image_width/height, camera_matrix, distortion_coefficients, ...).
    Optional: only used if --camera-info-file is given."""
    import yaml

    with open(path, "r") as f:
        data = yaml.safe_load(f)

    def mat_data(key, default=None):
        v = data.get(key)
        if isinstance(v, dict) and "data" in v:
            return [float(x) for x in v["data"]]
        if isinstance(v, list):
            return [float(x) for x in v]
        return default

    ci = CameraInfo()
    ci.header.frame_id = frame_id
    ci.width = int(data.get("image_width", 0))
    ci.height = int(data.get("image_height", 0))
    ci.distortion_model = data.get("distortion_model", "plumb_bob")
    ci.D = mat_data("distortion_coefficients", [])
    ci.K = mat_data("camera_matrix", [0.0] * 9)
    ci.R = mat_data("rectification_matrix", [1, 0, 0, 0, 1, 0, 0, 0, 1])
    ci.P = mat_data("projection_matrix", [0.0] * 12)
    return ci


def gen_frames(frames_dir, manifest):
    """Yield {img, t_offset_s} from the metadata frame manifest (PNG source)."""
    for fr in manifest:
        path = os.path.join(frames_dir, fr["file"])
        img = cv2.imread(path, cv2.IMREAD_COLOR)  # BGR
        if img is None:
            rospy.logwarn("Skipping unreadable frame: %s", path)
            continue
        yield {"img": img, "t_offset_s": float(fr["t_offset_s"])}


def gen_mp4(mp4_path, fps):
    """Yield {img, t_offset_s} from the mp4 (t inferred from fps -- less
    reliable than the PNG t_offset_s, hence not the default)."""
    cap = cv2.VideoCapture(mp4_path)
    if not cap.isOpened():
        raise SystemExit("Could not open mp4: %s" % mp4_path)
    fps = fps if fps and fps > 0 else 30.0
    i = 0
    try:
        while not rospy.is_shutdown():
            ok, frame = cap.read()  # BGR
            if not ok:
                break
            yield {"img": frame, "t_offset_s": i / fps}
            i += 1
    finally:
        cap.release()


def companion_camera_info_topic(image_topic):
    """ROS convention: camera_info is a sibling of the image topic."""
    return image_topic.rsplit("/", 1)[0] + "/camera_info"


def publish_pass(pub, ci_pub, ci_msg, frame_factory, base_epoch, args, bridge):
    """Publish one full pass over the frames, honoring inter-frame timing."""
    prev_off = None
    n = 0
    for item in frame_factory():
        if rospy.is_shutdown():
            return n
        off = item["t_offset_s"]
        if args.rate_multiplier > 0 and prev_off is not None:
            dt = (off - prev_off) / args.rate_multiplier
            if dt > 0:
                rospy.sleep(dt)
        prev_off = off

        if args.stamp_mode == "absolute":
            stamp = rospy.Time.from_sec(base_epoch + off)
        else:
            stamp = rospy.Time.now()

        msg = bridge.cv2_to_imgmsg(item["img"], encoding="bgr8")
        msg.header.stamp = stamp
        msg.header.frame_id = args.frame_id
        pub.publish(msg)

        if ci_pub is not None:
            ci_msg.header.stamp = stamp
            ci_pub.publish(ci_msg)

        n += 1
    return n


def main():
    rospy.init_node("zed_frame_publisher")
    # Strip ROS remapping args so argparse sees only real flags; each flag
    # falls back to its private (~) rosparam so roslaunch can drive it too.
    argv = rospy.myargv(argv=sys.argv)[1:]
    p = argparse.ArgumentParser(description="Publish zed_record.py frames as sensor_msgs/Image")
    p.add_argument("--session-dir", default=rospy.get_param("~session_dir", ""),
                   help="Path to a zed_record.py session folder (with metadata.json).")
    p.add_argument("--source", choices=("frames", "mp4"),
                   default=rospy.get_param("~source", "frames"),
                   help="frames (default, per-frame t_offset_s) or mp4 (t from fps).")
    p.add_argument("--stamp-mode", choices=("absolute", "now"),
                   default=rospy.get_param("~stamp_mode", "absolute"),
                   help="absolute = started_utc + t_offset_s (default); now = ros::Time::now().")
    p.add_argument("--frame-id", default=rospy.get_param("~frame_id", "zed_right"),
                   help="header.frame_id (default zed_right).")
    p.add_argument("--image-topic", default=rospy.get_param("~image_topic", "/zed_right/image_raw"),
                   help="Image topic to publish (feed this to cam_pattern image_tp).")
    p.add_argument("--rate-multiplier", type=float,
                   default=float(rospy.get_param("~rate_multiplier", 1.0)),
                   help=">1 faster, <1 slower, <=0 no inter-frame sleep.")
    p.add_argument("--loop", action="store_true",
                   default=bool(rospy.get_param("~loop", False)),
                   help="Republish in a loop until shutdown.")
    p.add_argument("--camera-info-file", default=rospy.get_param("~camera_info_file", ""),
                   help="Optional YAML intrinsics -> publish CameraInfo (NOT used by LVT2Calib).")
    args = p.parse_args(argv)

    if not args.session_dir:
        raise SystemExit("--session-dir (or ~session_dir) is required.")
    session_dir = os.path.abspath(os.path.expanduser(args.session_dir))
    if not os.path.isdir(session_dir):
        raise SystemExit("Session dir not found: %s" % session_dir)

    meta = load_metadata(session_dir)
    rec = meta.get("recording", {})
    base_epoch = parse_iso_epoch(meta["session"]["started_utc"])

    if args.source == "frames":
        manifest = meta.get("frames", [])
        if not manifest:
            raise SystemExit("metadata.json lists no frames (run without --no-frames?).")
        frames_dir = os.path.join(session_dir, rec.get("frames_dir") or "frames")

        def frame_factory():
            return gen_frames(frames_dir, manifest)
        n_expected = len(manifest)
    else:
        mp4_name = rec.get("mp4_path")
        if not mp4_name:
            raise SystemExit("metadata.json has no mp4_path (run without --no-mp4?).")
        mp4_path = os.path.join(session_dir, mp4_name)
        fps = float(meta.get("camera", {}).get("fps", 30) or 30)

        def frame_factory():
            return gen_mp4(mp4_path, fps)
        n_expected = "?"

    bridge = CvBridge()
    pub = rospy.Publisher(args.image_topic, Image, queue_size=10)

    ci_pub = None
    ci_msg = None
    if args.camera_info_file:
        ci_msg = load_camera_info(args.camera_info_file, args.frame_id)
        ci_topic = companion_camera_info_topic(args.image_topic)
        ci_pub = rospy.Publisher(ci_topic, CameraInfo, queue_size=10)
        rospy.loginfo("Publishing CameraInfo on %s (from %s)", ci_topic, args.camera_info_file)

    rospy.loginfo(
        "zed_frame_publisher: session=%s source=%s frames=%s topic=%s "
        "frame_id=%s stamp=%s rate=%.2f loop=%s",
        session_dir, args.source, n_expected, args.image_topic,
        args.frame_id, args.stamp_mode, args.rate_multiplier, args.loop,
    )
    # Give subscribers (cam_pattern) a moment to connect before the first frame.
    rospy.sleep(0.5)

    total = 0
    first = True
    while not rospy.is_shutdown():
        n = publish_pass(pub, ci_pub, ci_msg, frame_factory, base_epoch, args, bridge)
        total += n
        if first:
            rospy.loginfo("Published %d frame(s) in first pass.", n)
            first = False
        if not args.loop or rospy.is_shutdown():
            break

    rospy.loginfo("Done. %d frame(s) published total.", total)


if __name__ == "__main__":
    try:
        main()
    except rospy.ROSInterruptException:
        pass

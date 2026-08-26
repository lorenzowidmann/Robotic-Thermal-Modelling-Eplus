#!/usr/bin/env python3
"""Publish FLIR thermal frames (RJPG) from a folder as ROS1 sensor_msgs/Image,
for LVT2Calib's thermal camera pattern node (cam_pattern, isRGB=false).

Sibling of zed_frame_publisher, adapted to FLIR radiometric JPEG (RJPG). An
RJPG is a JPEG whose main stream is the FLIR palette-rendered thermal image,
with the raw 16-bit radiometric data in an EXIF tag (RawThermalImage).

Two image modes (--image-mode):
  * embedded (default): cv2.imread() reads the embedded colormapped JPEG.
    No external tools; works with cv2 alone.
  * raw: extract RawThermalImage via exiftool, normalize the 16-bit data to
    8-bit (optionally applyColorMap). Cleaner, controllable contrast for blob
    detection, but needs `exiftool` installed.

LVT2Calib (cam_pattern.cpp) subscribes to a raw sensor_msgs/Image and does
cv_bridge::toCvShare(msg, "bgr8"), then undistorts (cv::remap) and runs
findCirclesGrid + SimpleBlobDetector on it. So we publish bgr8 (3 channels)
even for thermal. Intrinsics are read by LVT2Calib from a FILE
(data/camera_info/<cam_info_filename>), NOT from a CameraInfo topic.

Timestamp note: FLIR EXIF times come from the camera's own RTC, NOT the rover
clock, so they are NOT comparable to the LiDAR ROS2 stamps (unlike the ZED's
started_utc). pattern_collection_lc.cpp does not sync camera<->laser by stamp
anyway, so --stamp-mode is harmless; default 'now'.
"""

import os
import sys
import glob
import argparse
import subprocess
from datetime import datetime

import numpy as np
import rospy
import cv2
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, CameraInfo

IMG_EXTS = (".jpg", ".jpeg", ".rjpg")

# name -> cv2 colormap (raw mode only). 'none' = grayscale replicated to BGR.
COLORMAPS = {
    "none": None,
    "gray": None,
    "jet": cv2.COLORMAP_JET,
    "inferno": cv2.COLORMAP_INFERNO,
    "hot": cv2.COLORMAP_HOT,
    "magma": cv2.COLORMAP_MAGMA,
}


def list_images(image_dir, pattern):
    if pattern:
        files = sorted(glob.glob(os.path.join(image_dir, pattern)))
    else:
        files = sorted(
            os.path.join(image_dir, f)
            for f in os.listdir(image_dir)
            if os.path.splitext(f)[1].lower() in IMG_EXTS
        )
    return files


def load_embedded(path):
    """Embedded FLIR palette JPEG -> BGR uint8 (cv2 native)."""
    return cv2.imread(path, cv2.IMREAD_COLOR)


def load_raw(path, exiftool, colormap, byteswap):
    """Extract the 16-bit RawThermalImage via exiftool, normalize to 8-bit,
    optionally colormap -> BGR uint8. Returns None on failure."""
    try:
        blob = subprocess.check_output(
            [exiftool, "-b", "-RawThermalImage", path], stderr=subprocess.DEVNULL
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        rospy.logwarn("exiftool RawThermalImage failed for %s: %s", path, exc)
        return None
    if not blob:
        return None
    raw = cv2.imdecode(np.frombuffer(blob, np.uint8), cv2.IMREAD_UNCHANGED)
    if raw is None:
        return None
    if raw.dtype == np.uint16 and byteswap:
        raw = raw.byteswap()
    norm = cv2.normalize(raw.astype(np.float32), None, 0, 255,
                         cv2.NORM_MINMAX).astype(np.uint8)
    cmap = COLORMAPS.get(colormap)
    if cmap is None:
        return cv2.cvtColor(norm, cv2.COLOR_GRAY2BGR)
    return cv2.applyColorMap(norm, cmap)


def exif_epoch(path, exiftool):
    """FLIR EXIF DateTimeOriginal(+SubSec) -> epoch seconds, or None.
    NB: camera RTC, not rover clock."""
    try:
        out = subprocess.check_output(
            [exiftool, "-s3", "-DateTimeOriginal", "-SubSecTimeOriginal", path],
            stderr=subprocess.DEVNULL,
        ).decode().splitlines()
    except (OSError, subprocess.CalledProcessError):
        return None
    if not out or not out[0].strip():
        return None
    try:
        dt = datetime.strptime(out[0].strip(), "%Y:%m:%d %H:%M:%S")
        epoch = dt.timestamp()
        if len(out) > 1 and out[1].strip().isdigit():
            epoch += float("0." + out[1].strip())
        return epoch
    except ValueError:
        return None


def load_camera_info(path, frame_id):
    """CameraInfo from a ROS camera_calibration-style YAML (optional)."""
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


def companion_camera_info_topic(image_topic):
    return image_topic.rsplit("/", 1)[0] + "/camera_info"


def main():
    rospy.init_node("flir_frame_publisher")
    argv = rospy.myargv(argv=sys.argv)[1:]
    p = argparse.ArgumentParser(description="Publish FLIR RJPG frames as sensor_msgs/Image")
    p.add_argument("--image-dir", default=rospy.get_param("~image_dir", ""),
                   help="Folder of FLIR RJPG/JPG frames.")
    p.add_argument("--pattern", default=rospy.get_param("~pattern", ""),
                   help="Optional glob (e.g. 'FLIR*.jpg'); default: all jpg/jpeg/rjpg.")
    p.add_argument("--image-mode", choices=("embedded", "raw"),
                   default=rospy.get_param("~image_mode", "embedded"),
                   help="embedded (cv2 palette JPEG, default) or raw (16-bit via exiftool).")
    p.add_argument("--colormap", default=rospy.get_param("~colormap", "none"),
                   help="raw mode colormap: none/gray/jet/inferno/hot/magma.")
    p.add_argument("--raw-byteswap", action="store_true",
                   default=bool(rospy.get_param("~raw_byteswap", False)),
                   help="Byte-swap 16-bit raw (some FLIR dumps are big-endian).")
    p.add_argument("--exiftool", default=rospy.get_param("~exiftool", "exiftool"),
                   help="exiftool executable (raw / exif stamp modes).")
    p.add_argument("--fps", type=float, default=float(rospy.get_param("~fps", 5.0)),
                   help="Playback frame rate (no per-frame offsets in RJPG).")
    p.add_argument("--rate-multiplier", type=float,
                   default=float(rospy.get_param("~rate_multiplier", 1.0)),
                   help=">1 faster, <1 slower, <=0 no inter-frame sleep.")
    p.add_argument("--stamp-mode", choices=("now", "sequential", "exif"),
                   default=rospy.get_param("~stamp_mode", "now"),
                   help="now (default) | sequential (start+i/fps) | exif (camera RTC).")
    p.add_argument("--frame-id", default=rospy.get_param("~frame_id", "flir_thermal"),
                   help="header.frame_id (default flir_thermal).")
    p.add_argument("--image-topic", default=rospy.get_param("~image_topic", "/thermal_cam/thermal_image"),
                   help="Image topic (matches LVT2Calib thermal default).")
    p.add_argument("--loop", action="store_true",
                   default=bool(rospy.get_param("~loop", False)),
                   help="Republish in a loop until shutdown.")
    p.add_argument("--camera-info-file", default=rospy.get_param("~camera_info_file", ""),
                   help="Optional YAML intrinsics -> CameraInfo (NOT used by LVT2Calib).")
    args = p.parse_args(argv)

    if not args.image_dir:
        raise SystemExit("--image-dir (or ~image_dir) is required.")
    image_dir = os.path.abspath(os.path.expanduser(args.image_dir))
    if not os.path.isdir(image_dir):
        raise SystemExit("Image dir not found: %s" % image_dir)

    files = list_images(image_dir, args.pattern)
    if not files:
        raise SystemExit("No RJPG/JPG frames in %s (pattern=%r)." % (image_dir, args.pattern))

    bridge = CvBridge()
    pub = rospy.Publisher(args.image_topic, Image, queue_size=10)

    ci_pub = None
    ci_msg = None
    if args.camera_info_file:
        ci_msg = load_camera_info(args.camera_info_file, args.frame_id)
        ci_topic = companion_camera_info_topic(args.image_topic)
        ci_pub = rospy.Publisher(ci_topic, CameraInfo, queue_size=10)
        rospy.loginfo("Publishing CameraInfo on %s (from %s)", ci_topic, args.camera_info_file)

    def load(path):
        if args.image_mode == "raw":
            return load_raw(path, args.exiftool, args.colormap, args.raw_byteswap)
        return load_embedded(path)

    rospy.loginfo(
        "flir_frame_publisher: dir=%s n=%d mode=%s topic=%s frame_id=%s "
        "stamp=%s fps=%.2f rate=%.2f loop=%s",
        image_dir, len(files), args.image_mode, args.image_topic,
        args.frame_id, args.stamp_mode, args.fps, args.rate_multiplier, args.loop,
    )
    rospy.sleep(0.5)  # let cam_pattern connect before the first frame

    dt = 0.0
    if args.fps > 0 and args.rate_multiplier > 0:
        dt = (1.0 / args.fps) / args.rate_multiplier
    seq_start = rospy.Time.now().to_sec()

    total = 0
    first = True
    while not rospy.is_shutdown():
        n = 0
        for i, path in enumerate(files):
            if rospy.is_shutdown():
                break
            img = load(path)
            if img is None:
                rospy.logwarn("Skipping unreadable frame: %s", path)
                continue

            if args.stamp_mode == "exif":
                e = exif_epoch(path, args.exiftool)
                stamp = rospy.Time.from_sec(e) if e else rospy.Time.now()
            elif args.stamp_mode == "sequential":
                stamp = rospy.Time.from_sec(seq_start + i * (dt if dt > 0 else 1.0 / max(args.fps, 1e-6)))
            else:
                stamp = rospy.Time.now()

            msg = bridge.cv2_to_imgmsg(img, encoding="bgr8")
            msg.header.stamp = stamp
            msg.header.frame_id = args.frame_id
            pub.publish(msg)
            if ci_pub is not None:
                ci_msg.header.stamp = stamp
                ci_pub.publish(ci_msg)

            n += 1
            if dt > 0:
                rospy.sleep(dt)

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

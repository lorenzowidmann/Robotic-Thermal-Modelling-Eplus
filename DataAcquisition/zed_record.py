"""Record-only capture utility for the ZED 2i (SVO2 + mp4 + still frames).

Two capture backends (see --backend):

  * sdk  — opens the ZED 2i via the ZED SDK (pyzed). Full stereo master:
           native SVO2 (.svo2) + mp4 + PNG frames. Requires an NVIDIA GPU
           with CUDA (the SDK is CUDA-only).
  * uvc  — opens the ZED 2i as a plain USB video (UVC) device via OpenCV.
           No SDK, no GPU: works on the rover (no NVIDIA card). RGB only —
           no SVO2 and no depth. The camera streams both eyes side-by-side
           in one frame; we split out the chosen --eye. mp4 + PNG frames.

Default backend is 'auto': try the SDK, and if pyzed is not importable fall
back to UVC automatically. Outputs land in one timestamped session folder:

  1. (sdk only) a native SVO2 recording (.svo2) — lossless container master,
     H264 by default (--svo-compression);
  2. a plain, viewable mp4 built from one RGB eye (--eye) via OpenCV VideoWriter;
  3. still frames dumped every --frame-interval seconds as PNGs into frames/ —
     this is what EmissivityCalculation/ consumes as input (it reads individual
     RGB image files via emissivity.sources.ImageSource). The RIGHT eye is used
     by default (--eye) because that is the eye CLIP classification runs on in
     EmissivityCalculation (see emissivity.sources: both --zed-uvc and --shared
     pass eye="right"); the mp4 uses the same eye so preview and frames match.

This is a pure recording tool: no emissivity estimation, CLIP inference, or
radiometric math lives here. Emissivity stays in EmissivityCalculation/ and
consumes frames/ (or the SVO) as a separate post-processing step.

Records until Ctrl+C or --duration elapses. Session metadata (resolution, fps,
serial number, compression, start/stop timestamps, frame manifest) is written
to metadata.json alongside the outputs.

Usage:
    py zed_record.py                                  # record until Ctrl+C
    py zed_record.py --duration 60                    # record 60 s then stop
    py zed_record.py --resolution HD720 --fps 60
    py zed_record.py --svo-compression lossless       # sdk backend only
    py zed_record.py --frame-interval 2.0             # a PNG every 2 s
    py zed_record.py --no-mp4 --no-frames             # SVO only (sdk)
    py zed_record.py --backend uvc                    # force UVC (no GPU/SDK)
    py zed_record.py --backend uvc --camera-index 2   # pick the ZED device

TODO(emissivity): EmissivityCalculation/ could consume a finished session
directly (point its --image loop at <session>/frames/*.png, or extract frames
from <session>/session.svo2). Left as a follow-up; not built here.
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "recordings"

# Flag string -> ZED SDK enum member name (resolved lazily against the imported
# sl module so this file still parses/imports without the SDK installed).
RESOLUTIONS = {
    "HD2K": "HD2K",
    "HD1080": "HD1080",
    "HD720": "HD720",
    "VGA": "VGA",
}
SVO_COMPRESSIONS = {
    "h264": "H264",
    "h265": "H265",
    "lossless": "LOSSLESS",
}

# UVC (no-SDK) side-by-side stereo frame geometry per resolution:
# (full_width = 2 x eye_width, full_height, max_fps). The ZED streams both eyes
# in one MJPG frame; eye_width = full_width // 2.
UVC_RESOLUTIONS = {
    "HD2K": (4416, 1242, 15),
    "HD1080": (3840, 1080, 30),
    "HD720": (2560, 720, 60),
    "VGA": (1344, 376, 100),
}


def parse_args():
    p = argparse.ArgumentParser(description="ZED 2i recorder (SVO2 + mp4 + frames)")
    p.add_argument(
        "--output-dir", default=str(DEFAULT_OUTPUT_DIR), metavar="DIR",
        help="Parent directory for session folders (default: recordings/ next "
        "to this script). Each run creates a timestamped sub-folder holding "
        "session.svo2, session_left.mp4, frames/, and metadata.json.",
    )
    p.add_argument(
        "--backend", choices=("auto", "sdk", "uvc"), default="auto",
        help="Capture backend: 'sdk' (ZED SDK/pyzed, needs NVIDIA CUDA, gives "
        "SVO2+depth), 'uvc' (plain USB webcam via OpenCV, no GPU/SDK, RGB only, "
        "no SVO), or 'auto' (default: try sdk, fall back to uvc if pyzed is "
        "missing).",
    )
    p.add_argument(
        "--camera-index", type=int, default=0, metavar="N",
        help="UVC backend only: OpenCV VideoCapture device index for the ZED "
        "(default 0). Bump this if another camera grabs index 0.",
    )
    p.add_argument(
        "--duration", type=float, default=None, metavar="SEC",
        help="Stop automatically after this many seconds (default: record "
        "until Ctrl+C).",
    )
    p.add_argument(
        "--resolution", choices=tuple(RESOLUTIONS), default="HD1080",
        help="Camera resolution (default HD1080). Higher resolutions cap the "
        "max fps: HD2K/HD1080 <= 30, HD720 <= 60, VGA <= 100.",
    )
    p.add_argument(
        "--fps", type=int, default=30, metavar="N",
        help="Capture frame rate (default 30). The SDK clamps to the nearest "
        "value the chosen --resolution supports.",
    )
    p.add_argument(
        "--svo-compression", choices=tuple(SVO_COMPRESSIONS), default="h264",
        help="SVO2 compression mode (sdk backend only): 'h264' (default, "
        "GPU-encoded, small), 'h265' (smaller, more GPU load), or 'lossless' "
        "(no quality loss, large files).",
    )
    p.add_argument(
        "--frame-interval", type=float, default=0.0, metavar="SEC",
        help="Seconds between dumped PNGs, for EmissivityCalculation to read. "
        "Default 0.0 = dump EVERY captured frame, so at --fps 30 you get 30 "
        "PNG/s (matching the mp4). Set e.g. 1.0 to throttle to one PNG/s. The "
        "full stereo stream still lives in the SVO/mp4.",
    )
    p.add_argument(
        "--eye", choices=("left", "right"), default="right",
        help="Which eye to export for the mp4 and PNG frames (default right): "
        "EmissivityCalculation's CLIP classification runs on the right eye "
        "(--zed-uvc/--shared pass eye='right'), so right keeps the recorded "
        "frames consistent with the classifier's input. The SVO always keeps "
        "both eyes regardless.",
    )
    p.add_argument(
        "--no-mp4", action="store_true",
        help="Skip the viewable mp4 export (SVO + frames only).",
    )
    p.add_argument(
        "--no-frames", action="store_true",
        help="Skip the per-interval PNG frame dump (SVO + mp4 only).",
    )
    p.add_argument(
        "--session-name", default=None, metavar="NAME",
        help="Session sub-folder name (default: UTC timestamp "
        "YYYYmmdd_HHMMSS).",
    )
    return p.parse_args()


def utc_now_iso() -> str:
    """ISO-8601 UTC timestamp with microsecond resolution
    (e.g. 2026-07-28T14:03:11.482913Z). Sub-second precision matters for
    clock-sync verification against the LiDAR's nanosecond ROS2 header stamps;
    rounding to whole seconds here was the dominant noise source in that diff."""
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def build_metadata(args, session_dir, camera_info, outputs, started_iso,
                   stopped_iso, duration_s, frames):
    """Provenance-style session sidecar, matching this codebase's flat
    json.dumps(doc, indent=2) style (cf. octree.smoothing.to_openstudio_json).
    outputs["svo"] is None on the UVC backend (no SVO master)."""
    svo = outputs["svo"]
    return {
        "schema": "zed_record/v1",
        "generated_by": "zed_record.py",
        "backend": outputs["backend"],
        "camera": camera_info,
        "recording": {
            "svo_path": None if svo is None else svo.name,
            "svo_compression": None if svo is None else args.svo_compression,
            "mp4_path": None if outputs["mp4"] is None else outputs["mp4"].name,
            "frames_dir": None if outputs["frames_dir"] is None else outputs["frames_dir"].name,
            "export_eye": args.eye,
            "frame_format": "png",
            "frame_interval_s": args.frame_interval,
            "n_frames": len(frames),
        },
        "session": {
            "dir": session_dir.name,
            "started_utc": started_iso,
            "stopped_utc": stopped_iso,
            "duration_s": None if duration_s is None else round(duration_s, 3),
            "stop_reason": outputs["stop_reason"],
        },
        # Frame manifest: filename + capture offset (s) from session start, so
        # a post-processing step can align each PNG to the SVO timeline.
        "frames": frames,
    }


def write_metadata(path, doc):
    path.write_text(json.dumps(doc, indent=2), encoding="utf-8")


def prepare_session(args):
    """Create the timestamped session folder and resolve output paths shared by
    both backends. Returns (session_dir, mp4_path, frames_dir, metadata_path)."""
    session_name = args.session_name or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    session_dir = Path(args.output_dir) / session_name
    session_dir.mkdir(parents=True, exist_ok=True)
    mp4_path = None if args.no_mp4 else session_dir / f"session_{args.eye}.mp4"
    frames_dir = None if args.no_frames else session_dir / "frames"
    if frames_dir is not None:
        frames_dir.mkdir(exist_ok=True)
    metadata_path = session_dir / "metadata.json"
    return session_dir, mp4_path, frames_dir, metadata_path


def record_sdk(args, sl):
    """SDK backend: SVO2 master + mp4 + PNG frames. Requires NVIDIA CUDA."""
    import time

    # --- open the camera -----------------------------------------------------
    zed = sl.Camera()
    init = sl.InitParameters()
    init.camera_resolution = getattr(sl.RESOLUTION, RESOLUTIONS[args.resolution])
    init.camera_fps = args.fps
    init.depth_mode = sl.DEPTH_MODE.NONE  # recording stores raw stereo; skip depth compute

    status = zed.open(init)
    if status != sl.ERROR_CODE.SUCCESS:
        # Graceful, single-line message instead of a raw stack trace for the
        # common "camera missing / already in use" cases.
        hints = {
            "CAMERA_NOT_DETECTED": "No ZED camera detected — check the USB connection.",
            "CAMERA_NOT_INITIALIZED": "ZED failed to initialize — replug and retry.",
            "CAMERA_ALREADY_IN_USE": "ZED is already in use by another process "
            "(close LivoxViewer2/other viewers or camera_server.py).",
        }
        msg = hints.get(str(status), f"Could not open ZED camera: {status}")
        print(msg, file=sys.stderr)
        return 1

    # --- session folder + output paths --------------------------------------
    session_dir, mp4_path, frames_dir, metadata_path = prepare_session(args)
    svo_path = session_dir / "session.svo2"

    # --- camera info for the sidecar ----------------------------------------
    info = zed.get_camera_information()
    camera_info = {
        "resolution": args.resolution,
        "fps": args.fps,
        "serial_number": getattr(info, "serial_number", None),
        "model": str(getattr(info, "camera_model", "")) or None,
    }

    # --- enable SVO2 recording ----------------------------------------------
    rec_params = sl.RecordingParameters(
        str(svo_path),
        getattr(sl.SVO_COMPRESSION_MODE, SVO_COMPRESSIONS[args.svo_compression]),
    )
    if zed.enable_recording(rec_params) != sl.ERROR_CODE.SUCCESS:
        print(f"Could not start SVO recording to {svo_path}", file=sys.stderr)
        zed.close()
        return 1

    # --- which eye feeds the mp4 + PNG exports (SVO keeps both) -------------
    eye_view = sl.VIEW.LEFT if args.eye == "left" else sl.VIEW.RIGHT
    eye_mat = sl.Mat()

    # --- optional mp4 writer (built from the chosen RGB eye) ----------------
    writer = None
    if mp4_path is not None:
        import cv2

        cam_cfg = getattr(info, "camera_configuration", None)
        res = getattr(cam_cfg, "resolution", None) if cam_cfg is not None else None
        width = getattr(res, "width", 1920) if res is not None else 1920
        height = getattr(res, "height", 1080) if res is not None else 1080
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(mp4_path), fourcc, float(args.fps), (width, height))

    started_iso = utc_now_iso()
    started_monotonic = time.monotonic()
    frames = []
    next_frame_at = 0.0
    frame_idx = 0
    runtime = sl.RuntimeParameters()

    def outputs(stop_reason):
        return {"svo": svo_path, "mp4": mp4_path, "frames_dir": frames_dir,
                "stop_reason": stop_reason, "backend": "sdk"}

    # Write a partial sidecar up front so a crash still leaves provenance.
    write_metadata(
        metadata_path,
        build_metadata(args, session_dir, camera_info, outputs(None),
                       started_iso, None, None, frames),
    )

    print(
        f"Recording (sdk) -> {session_dir}\n"
        f"  svo: {svo_path.name} ({args.svo_compression})"
        + ("" if writer is None else f"  |  mp4: {mp4_path.name}")
        + ("" if frames_dir is None else f"  |  frames: every {args.frame_interval}s")
        + "\nPress Ctrl+C to stop"
        + ("" if args.duration is None else f" (auto-stops after {args.duration}s).")
    )

    stop_reason = "interrupted"
    try:
        while True:
            if zed.grab(runtime) != sl.ERROR_CODE.SUCCESS:
                continue  # dropped frame; SVO keeps recording on the next grab
            elapsed = time.monotonic() - started_monotonic

            need_mp4 = writer is not None
            need_png = frames_dir is not None and elapsed >= next_frame_at
            if need_mp4 or need_png:
                import cv2

                zed.retrieve_image(eye_mat, eye_view)
                bgr = eye_mat.get_data()[:, :, :3]  # BGRA -> BGR (cv2 native)
                if need_mp4:
                    writer.write(bgr)
                if need_png:
                    fname = f"{args.eye}_{frame_idx:06d}.png"
                    cv2.imwrite(str(frames_dir / fname), bgr)
                    frames.append({"file": fname, "t_offset_s": round(elapsed, 3)})
                    frame_idx += 1
                    next_frame_at += args.frame_interval

            if args.duration is not None and elapsed >= args.duration:
                stop_reason = "duration"
                break
    except KeyboardInterrupt:
        stop_reason = "interrupted"
        print("\nStopping (Ctrl+C) ...")
    finally:
        zed.disable_recording()
        zed.close()
        if writer is not None:
            writer.release()

    stopped_iso = utc_now_iso()
    duration_s = time.monotonic() - started_monotonic
    write_metadata(
        metadata_path,
        build_metadata(args, session_dir, camera_info, outputs(stop_reason),
                       started_iso, stopped_iso, duration_s, frames),
    )

    print(
        f"Done ({stop_reason}, {duration_s:.1f}s). "
        f"{len(frames)} frame(s). Wrote {metadata_path}"
    )
    return 0


def record_uvc(args):
    """UVC backend: open the ZED as a plain USB webcam via OpenCV — no SDK, no
    NVIDIA GPU. RGB only (no SVO/depth). The ZED streams both eyes side-by-side
    in one frame; we split out the chosen --eye for the mp4 + PNG frames."""
    import time

    import cv2

    full_w, full_h, fps_cap = UVC_RESOLUTIONS[args.resolution]
    fps = min(args.fps, fps_cap)

    # --- open the camera as a UVC device ------------------------------------
    cap = cv2.VideoCapture(args.camera_index, cv2.CAP_ANY)
    if not cap.isOpened():
        print(
            f"Could not open the ZED as a UVC device (index {args.camera_index}). "
            "Check the USB connection and try a different --camera-index.",
            file=sys.stderr,
        )
        return 1
    # MJPG is what the ZED exposes for the full side-by-side resolutions.
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, full_w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, full_h)
    cap.set(cv2.CAP_PROP_FPS, fps)

    # Trust what the driver actually gave us (it may clamp the request).
    act_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or full_w
    act_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or full_h
    eye_w = act_w // 2

    # --- session folder + output paths --------------------------------------
    session_dir, mp4_path, frames_dir, metadata_path = prepare_session(args)

    camera_info = {
        "resolution": args.resolution,
        "fps": fps,
        "serial_number": None,
        "model": "ZED 2i (UVC)",
    }

    # --- which half of the side-by-side frame is the chosen eye -------------
    # ZED UVC layout: left eye in the left half, right eye in the right half.
    def crop_eye(frame):
        return frame[:, :eye_w] if args.eye == "left" else frame[:, eye_w:]

    # --- optional mp4 writer (single eye) -----------------------------------
    writer = None
    if mp4_path is not None:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(mp4_path), fourcc, float(fps), (eye_w, act_h))

    started_iso = utc_now_iso()
    started_monotonic = time.monotonic()
    frames = []
    next_frame_at = 0.0
    frame_idx = 0

    def outputs(stop_reason):
        return {"svo": None, "mp4": mp4_path, "frames_dir": frames_dir,
                "stop_reason": stop_reason, "backend": "uvc"}

    # Write a partial sidecar up front so a crash still leaves provenance.
    write_metadata(
        metadata_path,
        build_metadata(args, session_dir, camera_info, outputs(None),
                       started_iso, None, None, frames),
    )

    print(
        f"Recording (uvc, no SDK/GPU) -> {session_dir}\n"
        f"  eye: {args.eye} {eye_w}x{act_h} @ {fps}fps  (no SVO on this backend)"
        + ("" if writer is None else f"  |  mp4: {mp4_path.name}")
        + ("" if frames_dir is None else f"  |  frames: every {args.frame_interval}s")
        + "\nPress Ctrl+C to stop"
        + ("" if args.duration is None else f" (auto-stops after {args.duration}s).")
    )

    stop_reason = "interrupted"
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                continue  # dropped frame; keep going
            elapsed = time.monotonic() - started_monotonic

            need_mp4 = writer is not None
            need_png = frames_dir is not None and elapsed >= next_frame_at
            if need_mp4 or need_png:
                eye = crop_eye(frame)
                if need_mp4:
                    writer.write(eye)
                if need_png:
                    fname = f"{args.eye}_{frame_idx:06d}.png"
                    cv2.imwrite(str(frames_dir / fname), eye)
                    frames.append({"file": fname, "t_offset_s": round(elapsed, 3)})
                    frame_idx += 1
                    next_frame_at += args.frame_interval

            if args.duration is not None and elapsed >= args.duration:
                stop_reason = "duration"
                break
    except KeyboardInterrupt:
        stop_reason = "interrupted"
        print("\nStopping (Ctrl+C) ...")
    finally:
        cap.release()
        if writer is not None:
            writer.release()

    stopped_iso = utc_now_iso()
    duration_s = time.monotonic() - started_monotonic
    write_metadata(
        metadata_path,
        build_metadata(args, session_dir, camera_info, outputs(stop_reason),
                       started_iso, stopped_iso, duration_s, frames),
    )

    print(
        f"Done ({stop_reason}, {duration_s:.1f}s). "
        f"{len(frames)} frame(s). Wrote {metadata_path}"
    )
    # Print the exit wall-clock stamp (same value as metadata's stopped_utc) so
    # it can be diffed against an external clock (`date`, LiDAR ROS2 clock)
    # taken right after the script exits — camera-open/negotiation latency is
    # already behind us here, so this is a clean clock comparison point.
    print(f"stopped_utc {stopped_iso}")
    return 0


def main():
    args = parse_args()

    if args.backend == "uvc":
        return record_uvc(args)

    try:
        import pyzed.sl as sl
    except ImportError:
        if args.backend == "sdk":
            print(
                "ZED SDK not installed. Install the ZED SDK from "
                "https://www.stereolabs.com/developers/release/ and then the "
                "pyzed Python API (run the SDK's get_python_api.py). Requires an "
                "NVIDIA GPU with CUDA. On a machine with no NVIDIA GPU, run with "
                "--backend uvc instead (RGB frames only, no SVO/depth).",
                file=sys.stderr,
            )
            return 1
        # auto: no SDK -> fall back to the GPU-free UVC path (the rover case).
        print(
            "ZED SDK (pyzed) not available; falling back to UVC capture "
            "(RGB frames only, no SVO/depth — no NVIDIA GPU needed). "
            "Use --backend sdk to require the SDK, or --backend uvc to silence "
            "this notice.",
            file=sys.stderr,
        )
        return record_uvc(args)

    return record_sdk(args, sl)


if __name__ == "__main__":
    sys.exit(main())

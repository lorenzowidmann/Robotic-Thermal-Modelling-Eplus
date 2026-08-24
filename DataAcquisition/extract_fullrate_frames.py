#!/usr/bin/env python3
"""
extract_fullrate_frames.py

Extract every frame of a ZED session's exported MP4 (session_right.mp4) at the
video's real measured fps, into a fullrate/frames/ subfolder alongside the
zed_record.py session's existing subsampled frames/ (left untouched). Writes
a matching fullrate/metadata.json (same zed_record/v1 schema) with per-frame
t_offset_s based on the measured fps.

The nominal fps in metadata.json ("camera": {"fps": 30}) is what was
requested from the UVC camera at record time, not necessarily what was
achieved -- the OpenCV/UVC capture loop (no SDK/GPU on the rover) can have
jitter. Real average fps is measured from the decoded frame count instead of
assumed, using metadata.json's session.duration_s (from the recorder's own
start/stop timestamps) as the more reliable duration reference, rather than
the mp4 container's own (often rounded) duration.

Usage:
    py extract_fullrate_frames.py --session-dir <ZED session dir>
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path


def ffprobe_frame_count(video: Path) -> int:
    """Exact decoded frame count (ffprobe -count_frames), not the container's
    (sometimes inaccurate) nb_frames tag."""
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
         "-show_entries", "stream=nb_read_frames", "-of", "csv=p=0", str(video)],
        capture_output=True, text=True, check=True,
    )
    return int(out.stdout.strip())


def extract_frames(video: Path, out_dir: Path) -> int:
    import cv2

    out_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video))
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        cv2.imwrite(str(out_dir / f"right_{i:06d}.png"), frame)
        i += 1
    cap.release()
    return i


def main():
    p = argparse.ArgumentParser(
        description="Extract every session_right.mp4 frame at measured real fps into fullrate/")
    p.add_argument(
        "--session-dir", required=True, metavar="DIR",
        help="ZED session folder from zed_record.py (holds metadata.json + session_right.mp4).",
    )
    args = p.parse_args()

    session_dir = Path(args.session_dir)
    meta_path = session_dir / "metadata.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))

    video = session_dir / meta["recording"]["mp4_path"]
    duration_s = meta["session"]["duration_s"]

    n_probed = ffprobe_frame_count(video)
    fps = (n_probed - 1) / duration_s
    print(f"ffprobe decoded frame count = {n_probed}")
    print(f"session.duration_s (metadata.json)  = {duration_s}")
    print(f"fps_reale (measured)                = {fps:.6f}")

    out_root = session_dir / "fullrate"
    frames_dir = out_root / "frames"
    n_extracted = extract_frames(video, frames_dir)
    print(f"extracted {n_extracted} PNG frame(s) -> {frames_dir}")

    n_frames = n_extracted
    if n_extracted != n_probed:
        print(
            f"NOTE: ffprobe counted {n_probed} but OpenCV extracted {n_extracted} "
            "-- using the extracted count for fps/timestamps.", file=sys.stderr,
        )
        fps = (n_frames - 1) / duration_s

    frames = [
        {"file": f"right_{i:06d}.png", "t_offset_s": round(i / fps, 3)}
        for i in range(n_frames)
    ]

    last_offset = frames[-1]["t_offset_s"]
    if abs(last_offset - duration_s) > 3.0:
        sys.exit(
            f"Sanity check failed: last t_offset_s={last_offset} vs "
            f"session.duration_s={duration_s} (diff "
            f"{abs(last_offset - duration_s):.3f}s > 3s). Not writing metadata.json."
        )

    out_meta = {
        "schema": "zed_record/v1",
        "generated_by": "extract_fullrate_frames.py",
        "backend": meta.get("backend"),
        "camera": meta.get("camera"),
        "recording": {
            "svo_path": None,
            "svo_compression": None,
            "mp4_path": meta["recording"]["mp4_path"],
            "frames_dir": "frames",
            "export_eye": meta["recording"].get("export_eye"),
            "frame_format": "png",
            "frame_interval_s": round(1.0 / fps, 6),
            "n_frames": n_frames,
        },
        "session": meta["session"],
        "frames": frames,
    }
    out_path = out_root / "metadata.json"
    out_path.write_text(json.dumps(out_meta, indent=2), encoding="utf-8")
    print(f"wrote {out_path}")
    print(f"first t_offset_s = {frames[0]['t_offset_s']}, last t_offset_s = {last_offset}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

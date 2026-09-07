#!/usr/bin/env python3
"""
retime_fullrate_frames.py

Replace the *assumed uniform* timestamps written by extract_fullrate_frames.py
with the ZED's *measured* capture times.

Why this is needed
------------------
zed_record.py writes two things from the same capture loop:

  * session_right.mp4 -- EVERY grabbed frame (mp4 frame index == grab index);
  * frames/right_NNNNNN.png -- one frame every --frame-interval seconds, each
    recorded in metadata.json with its REAL `t_offset_s` (time.monotonic since
    the session start).

extract_fullrate_frames.py then re-extracts the mp4 into fullrate/frames/ but
throws the measured times away: it re-times every frame on a uniform grid,
`t_offset_s = i / fps` with `fps = (n-1) / session.duration_s`. That grid is
wrong twice over:

  * the anchor -- frame 0 is pinned to session.started_utc, but the first grab
    only lands after the UVC pipeline has warmed up (~0.27 s on session 9);
  * the rate -- `duration_s` is the recorder's start->stop wall clock, which
    also covers camera open/close, so the derived fps is biased (27.155 vs a
    measured 27.276 on session 9), and the error accumulates linearly.

On top of that the real grab loop jitters, which no single fps can express.
The result is a ZED clock that wanders by several hundred ms against the LiDAR
clock -- i.e. the projected cloud lands on the wrong part of the RGB frame, and
by a different amount at the start and at the end of the session.

What this does
--------------
Every subsampled PNG is a frame that is also in the mp4, and it carries a real
timestamp. Matching each PNG back to its mp4 frame index therefore recovers
(index, true time) anchors -- roughly one every 0.4 s -- and the timestamp of
every fullrate frame follows by interpolation between them.

Matching is normalized cross-correlation on a downscaled grayscale descriptor,
restricted to a window around the uniform-grid prediction. Anchors whose peak is
not locally unique (static scene: many identical frames) are dropped, not
guessed.

Residual limitation: `t_offset_s` is stamped after `cap.read()` returns, so it
is a delivery time, not an exposure time. The USB/decode latency it includes is
constant and cannot be recovered from the recording -- it stays as a constant
bias, which the FLIR<->ZED event sync absorbs.

Usage:
    py retime_fullrate_frames.py --session-dir <ZED session dir>            # dry run
    py retime_fullrate_frames.py --session-dir <ZED session dir> --apply
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

DESCR_SIZE = (96, 54)   # (w, h) of the matching descriptor
TIE_TOL = 1e-3          # NCC margin below the peak still counted as a tie
MAX_TIE_SPREAD = 4      # frames; wider tie cluster -> ambiguous anchor, dropped


def descriptor(bgr) -> np.ndarray:
    """Downscaled, zero-mean, unit-norm grayscale vector -> NCC by dot product."""
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    g = cv2.resize(g, DESCR_SIZE, interpolation=cv2.INTER_AREA).astype(np.float32).ravel()
    g -= g.mean()
    n = np.linalg.norm(g)
    return g / n if n > 0 else g


def video_descriptors(video: Path) -> np.ndarray:
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        sys.exit(f"Cannot open {video}")
    out = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        out.append(descriptor(frame))
    cap.release()
    if not out:
        sys.exit(f"No frames decoded from {video}")
    return np.stack(out)


def match_anchors(vid: np.ndarray, frames_dir: Path, anchors: list[dict],
                  interval: float, half_window: int) -> tuple[list[tuple[int, float]], list[dict]]:
    """(mp4 index, true t_offset_s) for every unambiguously matched PNG."""
    kept, report = [], []
    n = vid.shape[0]
    for a in anchors:
        path = frames_dir / a["file"]
        img = cv2.imread(str(path))
        if img is None:
            report.append({"file": a["file"], "status": "unreadable"})
            continue
        t = float(a["t_offset_s"])
        guess = int(round(t / interval))
        lo = max(0, guess - half_window)
        hi = min(n, guess + half_window + 1)
        sims = vid[lo:hi] @ descriptor(img)
        best = int(np.argmax(sims))
        ties = np.flatnonzero(sims >= sims[best] - TIE_TOL)
        spread = int(max(abs(ties[0] - best), abs(ties[-1] - best)))
        idx = lo + best
        rec = {"file": a["file"], "t_offset_s": t, "index": idx,
               "ncc": float(sims[best]), "tie_spread": spread}
        if spread > MAX_TIE_SPREAD:
            rec["status"] = "ambiguous"
        elif kept and idx <= kept[-1][0]:
            rec["status"] = "non-monotonic"
        else:
            rec["status"] = "ok"
            kept.append((idx, t))
        report.append(rec)
    return kept, report


def interpolate(n_frames: int, kept: list[tuple[int, float]]) -> np.ndarray:
    """Per-frame times: linear between anchors, local-rate extrapolation outside."""
    ks = np.array([k for k, _ in kept], float)
    ts = np.array([t for _, t in kept], float)
    rate = np.polyfit(ks, ts, 1)[0]          # s per frame, for the tails
    idx = np.arange(n_frames, dtype=float)
    out = np.interp(idx, ks, ts)
    head, tail = idx < ks[0], idx > ks[-1]
    out[head] = ts[0] + (idx[head] - ks[0]) * rate
    out[tail] = ts[-1] + (idx[tail] - ks[-1]) * rate
    return out


def main() -> int:
    p = argparse.ArgumentParser(
        description="Re-timestamp fullrate/ frames from the recorder's measured times")
    p.add_argument("--session-dir", required=True, metavar="DIR",
                   help="ZED session folder (metadata.json + session_right.mp4 + fullrate/).")
    p.add_argument("--apply", action="store_true",
                   help="Write fullrate/metadata.json (backing the old one up). "
                        "Without it, nothing is written.")
    p.add_argument("--search-window", type=int, default=90, metavar="N",
                   help="Half-width, in mp4 frames, of the search around the "
                        "uniform-grid guess (default 90 ~ 3.3 s).")
    args = p.parse_args()

    session = Path(args.session_dir)
    rec_meta = json.loads((session / "metadata.json").read_text(encoding="utf-8"))
    full_dir = session / "fullrate"
    full_path = full_dir / "metadata.json"
    full_meta = json.loads(full_path.read_text(encoding="utf-8"))

    interval = float(full_meta["recording"]["frame_interval_s"])
    frames_dir = session / (rec_meta["recording"].get("frames_dir") or "frames")
    anchors = rec_meta.get("frames", [])
    if not anchors:
        sys.exit(f"{session/'metadata.json'} lists no frames -- nothing to anchor on.")

    video = session / rec_meta["recording"]["mp4_path"]
    print(f"decoding {video.name} ...")
    vid = video_descriptors(video)
    n_frames = int(full_meta["recording"]["n_frames"])
    if vid.shape[0] != n_frames:
        print(f"NOTE: mp4 has {vid.shape[0]} frames, fullrate/metadata.json says "
              f"{n_frames}; using {min(vid.shape[0], n_frames)}.", file=sys.stderr)
        n_frames = min(vid.shape[0], n_frames)

    kept, report = match_anchors(vid, frames_dir, anchors, interval, args.search_window)
    dropped = [r for r in report if r["status"] != "ok"]
    print(f"anchors: {len(kept)}/{len(anchors)} matched"
          + (f", {len(dropped)} dropped ("
             + ", ".join(sorted({r['status'] for r in dropped})) + ")" if dropped else ""))
    if len(kept) < 10:
        sys.exit("Too few unambiguous anchors to re-time this session.")

    t_new = interpolate(n_frames, kept)
    t_old = np.arange(n_frames) * interval
    delta = t_new - t_old

    ks = np.array([k for k, _ in kept], float)
    ts = np.array([t for _, t in kept], float)
    slope, icept = np.polyfit(ks, ts, 1)
    resid = ts - (slope * ks + icept)
    print(f"measured rate  {1/slope:.4f} fps (uniform grid assumed {1/interval:.4f})")
    print(f"anchor offset  {icept:+.3f} s at frame 0")
    print(f"jitter around a constant rate: rms {resid.std()*1000:.0f} ms, "
          f"max {np.abs(resid).max()*1000:.0f} ms")
    print(f"correction applied to fullrate timestamps: "
          f"{delta.min():+.3f} .. {delta.max():+.3f} s "
          f"(first {delta[0]:+.3f}, last {delta[-1]:+.3f})")

    if not args.apply:
        print("\ndry run -- pass --apply to write fullrate/metadata.json")
        return 0

    backup = full_dir / "metadata.uniform.json"
    if not backup.exists():
        backup.write_text(json.dumps(full_meta, indent=2), encoding="utf-8")
        print(f"backed up original -> {backup}")

    full_meta["frames"] = [{"file": f"right_{i:06d}.png", "t_offset_s": round(float(t_new[i]), 3)}
                           for i in range(n_frames)]
    full_meta["recording"]["n_frames"] = n_frames
    full_meta["recording"]["frame_interval_s"] = round(float(slope), 6)
    full_meta["generated_by"] = "retime_fullrate_frames.py"
    full_meta["retiming"] = {
        "source": "measured t_offset_s of the recorder's subsampled frames, "
                  "matched to mp4 frame indices by NCC",
        "previous_generated_by": "extract_fullrate_frames.py",
        "previous_frame_interval_s": interval,
        "backup": backup.name,
        "n_anchors_used": len(kept),
        "n_anchors_total": len(anchors),
        "measured_fps": round(1 / slope, 4),
        "anchor_offset_s": round(float(icept), 4),
        "jitter_rms_s": round(float(resid.std()), 4),
        "correction_min_s": round(float(delta.min()), 4),
        "correction_max_s": round(float(delta.max()), 4),
    }
    full_path.write_text(json.dumps(full_meta, indent=2), encoding="utf-8")
    print(f"wrote {full_path}")
    print("NOTE: the ZED clock moved -- regenerate sync_manifest.json "
          "(TimeSyncCheck/sync_manifest.py) so the FLIR<->ZED event offset is "
          "recomputed on the corrected timestamps.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

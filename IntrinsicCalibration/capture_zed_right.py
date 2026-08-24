"""Grab ZED 2i right-eye frames on a keypress, for intrinsic calibration.

Step one of calibrating: collect the checkerboard images. Live preview, one
keypress per pose, straight to disk (default: ZedCaptures/ alongside this
script, created automatically). The calibration itself runs separately in
MATLAB (IntCalibration_ZED.m) -- point its folderPath at ZedCaptures/ (or
wherever you moved the images) and set squareSize to match your printed
board.

Over USB the ZED 2i is one wide webcam whose frame is the left+right pair
concatenated side by side (unrectified). Only the right half is written --
that is the eye the pipeline uses.

Keys:
    SPACE / ENTER   save the current frame
    u               delete the last saved frame
    q / ESC         quit

Usage:
    py capture_zed_right.py                              # largest mode (2208x1242 per eye)
    py capture_zed_right.py --checkerboard-size 9 6      # live board-detected indicator
    py capture_zed_right.py --resolution 2560x720        # 1280x720 per eye, if USB can't hold HD2K
    py capture_zed_right.py --resolution driver          # leave the camera's own mode alone
    py capture_zed_right.py --out-dir some/other/folder
    py capture_zed_right.py --camera-index 1             # if the ZED isn't device 0
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

# Default alongside this script, so it resolves correctly on both the Windows
# box and the rover's Ubuntu checkout without editing anything.
DEFAULT_OUT_DIR = Path(__file__).resolve().parent / "ZedCaptures"

# ZED 2i UVC side-by-side modes (full stereo frame). Per-eye width is half.
# The camera often opens at the smallest one, and calibrating a 672x376 eye is
# needlessly imprecise -- prefer 2560x720 (1280x720 per eye) when the USB link
# sustains it.
RESOLUTIONS = {
    "1344x376": (1344, 376),
    "2560x720": (2560, 720),
    "3840x1080": (3840, 1080),
    "4416x1242": (4416, 1242),
}

# Default to the largest mode, not the driver's pick: left alone the ZED
# usually opens at the smallest (672x376 per eye), where a checkerboard square
# is ~15 px and corner localisation is at its noisiest. Frame rate is
# irrelevant here -- the board is static, and 15 fps is plenty to compose a
# pose. If the USB link can't sustain it the driver keeps its own mode and
# open_camera reports the mismatch rather than pretending.
DEFAULT_RESOLUTION = "4416x1242"

_SAVE_KEYS = (32, 13, 10)  # SPACE, CR, LF
_QUIT_KEYS = (ord("q"), 27)  # q, ESC


def open_camera(index: int, resolution: str | None):
    """Open the ZED as a plain UVC device, optionally forcing a stereo mode.

    No ZED SDK: this is cv2.VideoCapture on the raw side-by-side stream. The
    requested mode is only a request -- the driver silently keeps its current
    mode if the USB link can't sustain it, so the applied size is read back and
    returned rather than assumed.
    """
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        raise RuntimeError(
            f"could not open camera index {index} "
            "(try --camera-index 1/2; on Ubuntu check `ls /dev/video*` and that "
            "you are in the `video` group)"
        )
    if resolution:
        w, h = RESOLUTIONS[resolution]
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)

    applied = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    return cap, applied


def right_eye(frame: np.ndarray) -> np.ndarray:
    """Right half of a side-by-side stereo frame."""
    _left, right = np.split(frame, 2, axis=1)
    return np.ascontiguousarray(right)


def next_index(out_dir: Path) -> int:
    """Continue numbering after whatever is already in `out_dir`.

    Capture sessions get interrupted (battery, USB re-enumeration); resuming
    should extend the set, not overwrite frame 001 of the previous run.
    """
    existing = sorted(out_dir.glob("right_*.png"))
    if not existing:
        return 1
    return max(int(p.stem.split("_")[-1]) for p in existing) + 1


def capture_loop(cap, out_dir: Path, pattern: tuple[int, int] | None, detect_every: int) -> int:
    """Preview, save on keypress, return how many frames were written.

    When `pattern` is given, a cheap FAST_CHECK detection runs every
    `detect_every` frames purely as a green/red "board visible" indicator --
    it saves the raw frame regardless. The real, sub-pixel detection happens
    later in MATLAB (IntCalibration_ZED.m), so a missed indicator never costs
    accuracy.
    """
    window = "ZED right eye  |  SPACE save  |  u undo  |  q quit"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)

    saved: list[Path] = []
    index = next_index(out_dir)
    found = None
    frame_no = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            print("  [warn] dropped frame", file=sys.stderr)
            if cv2.waitKey(1) & 0xFF in _QUIT_KEYS:
                break
            continue

        eye = right_eye(frame)
        frame_no += 1

        if pattern is not None and frame_no % max(1, detect_every) == 0:
            gray = cv2.cvtColor(eye, cv2.COLOR_BGR2GRAY)
            found, _ = cv2.findChessboardCorners(
                gray, pattern,
                flags=cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK,
            )

        preview = eye.copy()
        status = f"saved: {len(saved)}   {eye.shape[1]}x{eye.shape[0]}"
        colour = (255, 255, 255)
        if pattern is not None:
            status += f"   board: {'YES' if found else 'no'}"
            colour = (0, 255, 0) if found else (0, 0, 255)
        cv2.putText(preview, status, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, colour, 2)
        cv2.imshow(window, preview)

        key = cv2.waitKey(1) & 0xFF
        if key in _QUIT_KEYS:
            break
        if key in _SAVE_KEYS:
            path = out_dir / f"right_{index:03d}.png"
            cv2.imwrite(str(path), eye)
            saved.append(path)
            index += 1
            print(f"  saved {path.name}  ({len(saved)} this session)")
        elif key == ord("u") and saved:
            gone = saved.pop()
            gone.unlink(missing_ok=True)
            index -= 1
            print(f"  deleted {gone.name}  ({len(saved)} left this session)")

    cv2.destroyWindow(window)
    return len(saved)


def parse_args():
    p = argparse.ArgumentParser(
        description="Capture ZED 2i right-eye frames for intrinsic calibration"
    )
    p.add_argument(
        "--out-dir", default=str(DEFAULT_OUT_DIR), metavar="DIR",
        help=f"Where to write the frames (default {DEFAULT_OUT_DIR})",
    )
    p.add_argument(
        "--camera-index", type=int, default=0, metavar="N",
        help="UVC device index (default 0)",
    )
    p.add_argument(
        "--resolution", choices=sorted(RESOLUTIONS) + ["driver"], default=DEFAULT_RESOLUTION,
        help=f"ZED side-by-side stereo mode; per-eye width is half of it "
             f"(default {DEFAULT_RESOLUTION}, the largest). 'driver' leaves the "
             "mode alone, whatever the camera opens with.",
    )
    p.add_argument(
        "--checkerboard-size", type=int, nargs=2, default=None, metavar=("COLS", "ROWS"),
        help="INNER corner count -- only drives the live 'board visible' "
             "indicator, frames are saved either way",
    )
    p.add_argument(
        "--detect-every", type=int, default=5, metavar="N",
        help="Run the preview detection every N frames (default 5)",
    )
    return p.parse_args()


def main():
    args = parse_args()
    pattern = tuple(args.checkerboard_size) if args.checkerboard_size else None
    if pattern is not None and min(pattern) < 2:
        raise SystemExit(f"--checkerboard-size must be >= 2 in both axes (got {pattern!r})")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    requested = None if args.resolution == "driver" else args.resolution
    try:
        cap, applied = open_camera(args.camera_index, requested)
    except RuntimeError as exc:
        raise SystemExit(f"error: {exc}") from None

    print(f"Camera {args.camera_index} open at {applied[0]}x{applied[1]} stereo "
          f"-> {applied[0] // 2}x{applied[1]} per eye")
    if requested and applied != RESOLUTIONS[requested]:
        print(f"  note: asked for {requested}, driver kept {applied[0]}x{applied[1]} "
              "-- USB link may not sustain that mode; try a smaller --resolution")
    print("  calibrate at the mode you will RUN at: ZED modes are different "
          "sensor crops/binnings, so K does not rescale exactly between them")
    print(f"Writing to {out_dir}")
    print("SPACE = save | u = delete last | q or ESC = quit\n")

    try:
        n = capture_loop(cap, out_dir, pattern, args.detect_every)
    except KeyboardInterrupt:
        n = 0
    finally:
        cap.release()

    total = len(list(out_dir.glob("right_*.png")))
    print(f"\nSaved {n} frame(s) this session, {total} total in {out_dir}")
    if total < 10:
        print("  note: aim for 12-20 poses -- large tilts about both board axes, "
              "varied distances, board filling different parts of the frame")
    return 0


if __name__ == "__main__":
    sys.exit(main())

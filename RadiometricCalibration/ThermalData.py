"""Read apparent-temperature data out of recorded FLIR radiometric JPEGs.

The thermal camera (USB mass-storage / Linux gadget device) drops one
radiometric `*_R.jpg` per second into a session folder while running. This
reads those files back with `flyr` and returns the embedded per-pixel
temperature map (deg C) -- no radiometric correction applied here, that's
`main.py`'s job once a map from here is fed into it as `--thermal`.

Usage:
    py ThermalData.py --folder "C:\\Users\\loren\\Desktop\\FlyrCamera\\20250823_211855"
    py ThermalData.py --folder <session_dir> --frame 3 --out apparent.npy --show
"""

import argparse
import sys
from pathlib import Path

import flyr
import numpy as np


def list_session_frames(folder: str | Path) -> list[Path]:
    """Radiometric JPEGs in a session folder, in capture order."""
    folder = Path(folder)
    if not folder.is_dir():
        raise FileNotFoundError(f"Session folder not found: {folder}")
    frames = sorted(folder.glob("*_R.jpg"))
    if not frames:
        raise FileNotFoundError(f"No *_R.jpg radiometric files in {folder}")
    return frames


def read_temperature(path: str | Path) -> np.ndarray:
    """Apparent temperature map (deg C, 2-D float array) from one radiometric JPEG."""
    return flyr.unpack(str(path)).celsius


def read_session(folder: str | Path) -> list[np.ndarray]:
    """Apparent temperature maps for every frame in a session folder, in order."""
    return [read_temperature(f) for f in list_session_frames(folder)]


def consensus_temperature(maps: list[np.ndarray], tol: float) -> np.ndarray:
    """Per-pixel median over `maps`, rejecting (NaN) pixels that disagree by > tol.

    Meant for a short window of consecutive frames of a roughly static scene
    (or stationary rover): a real surface temperature stays put frame to
    frame, while a motion-blur/edge-lag transient (e.g. a moving object's
    edge sweeping a pixel) shows up as an outlier and gets rejected instead
    of silently reported.
    """
    stack = np.stack(maps, axis=0)
    spread = stack.max(axis=0) - stack.min(axis=0)
    consensus = np.median(stack, axis=0)
    consensus[spread > tol] = np.nan
    return consensus


def parse_args():
    p = argparse.ArgumentParser(
        description="Read apparent temperature from recorded thermal camera frames"
    )
    p.add_argument("--folder", required=True, help="Session folder of *_R.jpg files")
    p.add_argument(
        "--frame",
        type=int,
        default=None,
        help="Index of a single frame to read (default: report stats for all frames)",
    )
    p.add_argument("--out", help="Save the selected frame's temperature map (.npy/.csv)")
    p.add_argument(
        "--show",
        action="store_true",
        help="Display the selected frame; hover over it to read temperature per pixel",
    )
    p.add_argument(
        "--xy",
        type=int,
        nargs=2,
        metavar=("X", "Y"),
        default=None,
        help="Print the temperature at pixel (X, Y) of the selected frame",
    )
    p.add_argument(
        "--consensus",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Use a rolling window of N frames ending at --frame instead of a "
            "single frame: per-pixel median, rejecting (NaN) pixels where the "
            "window's max-min spread exceeds --tol. Requires --frame."
        ),
    )
    p.add_argument(
        "--tol",
        type=float,
        default=1.0,
        help="Max deg C spread allowed within the --consensus window (default: 1.0)",
    )
    return p.parse_args()


def main():
    args = parse_args()
    if args.consensus is not None and args.frame is None:
        sys.exit("--consensus requires --frame (the window's last frame)")

    frames = list_session_frames(args.folder)
    print(f"{len(frames)} frame(s) in {args.folder}")

    if args.frame is not None:
        path = frames[args.frame]
        title = path.name

        if args.consensus is not None:
            start = max(0, args.frame - args.consensus + 1)
            window = frames[start : args.frame + 1]
            temp = consensus_temperature(
                [read_temperature(f) for f in window], args.tol
            )
            rejected = np.isnan(temp).mean() * 100
            print(
                f"[{args.frame}] consensus over {len(window)} frame(s) "
                f"({window[0].name} .. {window[-1].name}), tol={args.tol} deg C: "
                f"{rejected:.1f}% pixels rejected as transient"
            )
            title = f"{title}  (consensus x{len(window)}, tol={args.tol})"
        else:
            temp = read_temperature(path)

        print(
            f"[{args.frame}] {path.name}: min={np.nanmin(temp):.2f}"
            f"  max={np.nanmax(temp):.2f}  mean={np.nanmean(temp):.2f}"
            f"  shape={temp.shape}"
        )
        if args.xy is not None:
            x, y = args.xy
            val = temp[y, x]
            print(f"T({x}, {y}) = {'rejected (transient)' if np.isnan(val) else f'{val:.2f} deg C'}")
        if args.out:
            from radiometric.io_maps import save_map

            save_map(args.out, temp)
            print(f"Saved to {args.out}")
        if args.show:
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots()
            ax.imshow(temp, cmap="inferno")
            fig.colorbar(ax.images[0], ax=ax, label="deg C")
            ax.set_title(f"{title}  (hover to read temperature)")

            h, w = temp.shape

            def format_coord(x, y):
                xi, yi = int(round(x)), int(round(y))
                if 0 <= xi < w and 0 <= yi < h:
                    val = temp[yi, xi]
                    val_str = "rejected" if np.isnan(val) else f"{val:.2f} deg C"
                    return f"x={xi} y={yi}  T={val_str}"
                return f"x={x:.0f} y={y:.0f}"

            ax.format_coord = format_coord
            plt.show()
        return

    for i, path in enumerate(frames):
        temp = read_temperature(path)
        print(
            f"[{i}] {path.name}: min={temp.min():.2f}  max={temp.max():.2f}"
            f"  mean={temp.mean():.2f}"
        )


if __name__ == "__main__":
    sys.exit(main())

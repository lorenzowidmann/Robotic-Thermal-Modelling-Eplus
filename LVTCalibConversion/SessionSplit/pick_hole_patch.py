"""
Interactive helper: click on one representative frame of a pose to define a
circular patch (centre, radius, fill colour), for patch_hole_pose.py.

Why: a physical hole partially occluded by a background object (seen through
it, e.g. a door handle) breaks the RGB blob detector's circularity test on
every frame of the pose -- but the board is static during a pose, so centre
and radius are the same in every frame. Pick them once here.

Runs on the Windows host (plain opencv-python window, no VcXsrv/Docker
needed).

Usage:
    py pick_hole_patch.py --frame <path_to_one_frame.png>

Click sequence (left click):
    1. Hole centre
    2. A point on the hole's boundary (defines the radius)
    3. A point in the CLEAN part of the same hole, away from the obstruction
       (defines the fill colour -- sampled as the median of a 5x5 patch there)

Press 'r' to restart the 3 clicks, 'q' or ESC to quit without printing.
After the 3rd click, a preview of the filled circle is shown; press any key
to accept and print the parameters, or 'r' to redo.
"""

import argparse
import sys

import cv2
import numpy as np


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--frame", required=True, help="One representative frame of the pose")
    ap.add_argument("--sample-size", type=int, default=5,
                    help="Side of the square patch sampled for colour (default 5)")
    args = ap.parse_args()

    img = cv2.imread(args.frame, cv2.IMREAD_COLOR)
    if img is None:
        sys.exit(f"Could not read: {args.frame}")

    win = "click: center, edge, clean-sample point  (r=restart, q=quit)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, min(1280, img.shape[1]), min(800, img.shape[0]))

    points = []

    def on_mouse(event, x, y, flags, userdata):
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < 3:
            points.append((x, y))

    cv2.setMouseCallback(win, on_mouse)

    while True:
        vis = img.copy()
        labels = ["centre", "edge", "sample"]
        for i, (x, y) in enumerate(points):
            cv2.drawMarker(vis, (x, y), (0, 165, 255), cv2.MARKER_CROSS, 14, 2)
            cv2.putText(vis, labels[i], (x + 8, y - 8), cv2.FONT_HERSHEY_SIMPLEX,
                       0.5, (0, 165, 255), 1, cv2.LINE_AA)
        if len(points) >= 2:
            cx, cy = points[0]
            ex, ey = points[1]
            r = int(round(((ex - cx) ** 2 + (ey - cy) ** 2) ** 0.5))
            cv2.circle(vis, (cx, cy), r, (0, 200, 0), 1)

        cv2.imshow(win, vis)
        key = cv2.waitKey(20) & 0xFF
        if key in (ord('q'), 27):
            cv2.destroyAllWindows()
            return
        if key == ord('r'):
            points.clear()
            continue

        if len(points) == 3:
            (cx, cy), (ex, ey), (sx, sy) = points
            radius = int(round(((ex - cx) ** 2 + (ey - cy) ** 2) ** 0.5))
            half = args.sample_size // 2
            patch = img[max(0, sy - half):sy + half + 1, max(0, sx - half):sx + half + 1]
            color = np.median(patch.reshape(-1, 3), axis=0).astype(int)  # BGR

            preview = img.copy()
            cv2.circle(preview, (cx, cy), radius, tuple(int(c) for c in color), -1)
            cv2.putText(preview, "accept: any key   redo: r", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
            cv2.imshow(win, preview)
            key2 = cv2.waitKey(0) & 0xFF
            if key2 == ord('r'):
                points.clear()
                continue

            b, g, r_ = int(color[0]), int(color[1]), int(color[2])
            print(f"\n--cx {cx} --cy {cy} --radius {radius} --color {r_},{g},{b}")
            cv2.destroyAllWindows()
            return


if __name__ == "__main__":
    main()

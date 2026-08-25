# IntrinsicCalibration

Camera intrinsics (focal length, principal point, distortion) for the ZED 2i
and the FLIR. Two-step: **capture in Python, calibrate in MATLAB.**

| file | what it does |
|---|---|
| `capture_zed_right.py` | grabs ZED 2i right-eye frames on a keypress, with live preview. One keypress per checkerboard pose, written straight to `ZedCaptures/`. |
| `IntCalibration_ZED.m` | MATLAB: runs the actual ZED calibration over the captured images. |
| `IntCalibration_Flir.m` | MATLAB: same for the FLIR. |
| `ZedCaptures/` | the captured checkerboard images (not tracked). |

## Usage

```powershell
py capture_zed_right.py                            # largest mode (2208x1242 per eye)
py capture_zed_right.py --checkerboard-size 9 6    # live board-detected indicator
py capture_zed_right.py --resolution 2560x720      # 1280x720 per eye, if USB can't hold HD2K
```

Keys: `SPACE`/`ENTER` save the current frame · `u` delete the last saved · `q`/`ESC` quit.

Then in MATLAB, point `IntCalibration_ZED.m`'s `folderPath` at `ZedCaptures/`
and set `squareSize` to match the printed board.

Over USB the ZED 2i appears as one wide webcam whose frame is the left+right
pair side by side (unrectified). Only the **right half** is written — that is
the eye the rest of the pipeline uses. No ZED SDK, no `pyzed`, no GPU needed.

These are the **intrinsics** only. The LiDAR↔camera **extrinsics** in the same
file come from a separate [LVT2Calib](https://github.com/Clothooo/lvt2calib)
board session — see `../LVTCalibBoardGenearation/`.

Results feed `../SensorFusionLoader/rig_calibration.yaml`, which is where every
downstream module reads intrinsics from. Both `flir:` and `zed:` blocks there
are MATLAB (`estimateCameraParameters`, no-skew) results, as of 2026-08-02
(ZED) / 2026-07-31 (FLIR) — not the OpenCV scripts under `../Thesis/Calibration/`,
which are kept only for cross-check/traceability.

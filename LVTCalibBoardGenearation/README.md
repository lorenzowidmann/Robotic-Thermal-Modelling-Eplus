# LVTCalibBoardGenearation

Generates the PCD template files that **LVT2Calib**
(https://github.com/Clothooo/lvt2calib) matches against, using the **real** geometry of the calibration board that was
built for this thesis rather than the repo's stock one.

Board: 100 × 70 cm, four circular holes of diameter 13 cm (radius 0.065 m),
centred at (±0.15, ±0.15) m from the board centre.

## Usage

```powershell
py generate_board_template.py --outdir <output_folder>
```

No dependencies — Python standard library only.

Writes two files, using the names LVT2Calib expects:

| file | what it is |
|---|---|
| `four_circle_boundary.pcd` | outlines only (outer rectangle + 4 circles). This is what `model_path` points at in `livox_pattern.launch`, and what the PCA + ICP registration in `isCalibBoard()` uses. |
| `four_circle_dense.pcd` | the filled board surface with the holes cut out — for visualisation and other comparisons. |

Then copy both into `lvt2calib/data/template_pcl/`, overwriting the originals
(back them up first).

The run also prints the matching values for `config/lidar_pattern_param.yaml`
(`circle_radius`, `centroid_dis_min`, `centroid_dis_max`) — keep those in sync
with the board, or detection silently fails.

**If you cut a new board, measure the holes with callipers and pass the real
number** via `--hole-diameter`; the nominal 13 cm is the design value, not the
cut one.

**Schema note:** the templates are written with the 5-field
`x y z intensity range` schema, matching the repo's original PCD. An earlier
version wrote only 4 fields, which `pcl::io::loadPCDFile` accepts silently while
mis-aligning the data — no error, wrong result.

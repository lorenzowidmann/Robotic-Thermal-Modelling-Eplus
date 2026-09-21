# MATLAB_NIRTDReading

Pt100 (RTD, four-wire) ground-truth temperature acquisition via a National
Instruments USB-9162 DAQ, used to validate `RadiometricCalibration`'s FLIR
correction against a physical contact reading.

| Script | What it does |
|---|---|
| `USB9126_Pt100Reading.m` | Live-plots and records one Pt100 acquisition (2 Hz, `dq.addinput("Dev2", "ai0", "RTD")`) to a `.mat` file (`data_vect` only, no timestamps). |
| `read_plot_acquisitions.m` | Loads every `.mat` in `acqFolder`, rebuilds the time axis from the known acquisition rate, and exports one plot per file to `output/`. |

## Usage

1. Set `saveName` in `USB9126_Pt100Reading.m`, run it, let it acquire, save
   the resulting `data_vect` to `Acquisitions/<saveName>.mat`.
2. Point `read_plot_acquisitions.m`'s `acqFolder` at that folder and run it —
   plots land in `output/` (gitignored, contents are per-run data).

No `requirements.txt` — MATLAB only, uses the Data Acquisition Toolbox.

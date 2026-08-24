"""Parse an ARPAV (Veneto regional weather service) hourly station-data page
(saved as .htm, e.g. "Dati orari" for a station) into a timestamp,ghi CSV
usable by sun_incidence.py's --irradiance-csv.

ARPAV's hourly tables label every row "ora solare" -- in Italian
meteorological usage this means fixed CET (UTC+1, winter/standard time),
NOT true apparent solar time and NOT whatever civil time (CET/CEST) was
actually in effect that day. It's simply how these stations avoid DST
ambiguity in their raw feed. This script localizes each row as UTC+1 fixed
offset, then converts to real Europe/Rome civil time (which becomes CEST/
UTC+2 for summer rows) -- get this wrong and every summer reading is off by
exactly 1 hour.

Column used: "Radiazione globale (W/m2)" -- the table's own hourly mean GHI.
Rows are stdlib-regex-parsed (no lxml/bs4 dependency) since ARPAV's table
markup is simple and consistent: <th ...>DD/MM/YYYY HH</th><td>...</td>...

Usage:
    python parse_arpav.py --in Legnaro.htm --out legnaro_ghi.csv

Venv: any with pandas (this repo's C:\\venvs\\planefit works).
"""
import argparse
import re
from pathlib import Path

import pandas as pd

ROW_RE = re.compile(
    r'<th[^>]*>(\d{2}/\d{2}/\d{4})\s+(\d{2})</th>'  # date, hour ("ora solare" = fixed UTC+1)
    r'(?:<td[^>]*>.*?</td>)'      # temp aria
    r'(?:<td[^>]*>.*?</td>)'      # pioggia
    r'(?:<td[^>]*>.*?</td>)'      # umidita min
    r'(?:<td[^>]*>.*?</td>)'      # umidita max
    r'<td[^>]*>([\d.]+)</td>',    # radiazione globale, W/m2
    re.DOTALL,
)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="inp", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=None, help="default: <in>_ghi.csv")
    args = ap.parse_args()

    html = args.inp.read_text(encoding="utf-8")
    rows = ROW_RE.findall(html)
    if not rows:
        raise SystemExit(f"no data rows matched in {args.inp} -- table markup may have changed")

    records = []
    for date_str, hour_str, ghi_str in rows:
        # "ora solare" = fixed UTC+1 (CET), not true solar time, not the civil
        # tz actually in effect -- localize as a fixed offset first, THEN
        # convert to real Europe/Rome civil time (shifts to CEST in summer).
        naive = pd.to_datetime(f"{date_str} {hour_str}:00", format="%d/%m/%Y %H:%M")
        ts_fixed = naive.tz_localize("Etc/GMT-1")
        ts_civil = ts_fixed.tz_convert("Europe/Rome")
        records.append({"timestamp": ts_civil, "ghi": float(ghi_str)})

    df = pd.DataFrame(records).sort_values("timestamp")
    out = args.out or args.inp.with_name(args.inp.stem + "_ghi.csv")
    df.to_csv(out, index=False)
    print(f"{len(df)} hourly row(s), {df['timestamp'].min()} .. {df['timestamp'].max()} (Europe/Rome civil time)")
    print(f"GHI: min {df['ghi'].min():.0f}, max {df['ghi'].max():.0f} W/m2")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()

"""
Stage 1 - Step 1 and 2: Load, clean and build household profiles.
Low Carbon London smart meter data (TU Delft .csv, Kaggle .csv, or the Monash .ts/.tsf file).

Place the downloaded file in the data/ folder, then run from the repo root:
    python scripts/clean_london_data.py

Outputs (in results/):
  hourly_kw.csv            hourly average kW, one column per kept household
  profiles_kw.csv          households x 24 hours, average kW  (use for centroids in kW)
  profiles_normalized.csv  households x 24 hours, each row divided by its own mean
                           (use for K-means if you want to cluster by SHAPE)
  cleaning_summary.txt     what was kept and dropped (copy into your report)

Needs: pip install -r requirements.txt
"""

from pathlib import Path
import numpy as np
import pandas as pd

# ------------------------- SETTINGS (edit these) -------------------------
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "results"
INPUT_FILE = None           # None = use the biggest .csv or .tsf file in data/. Or set e.g. DATA_DIR / "myfile.csv"
N_HOUSEHOLDS = 300          # keep 200 to 500
MAX_MISSING_FRACTION = 0.05 # drop households with more than 5% missing readings
MAX_GAP_HALFHOURS = 4       # fill gaps up to 4 half-hours (2 hours) by interpolation
RANDOM_SEED = 42
DAYFIRST = False            # set True if dates look like 31/12/2013
# Settings for the .ts / .tsf files (5,560 households, Nov 2011 to Feb 2014, gaps filled by carrying the
# last reading forward). The file is huge, so it is read line by line and only households that cover the
# whole analysis window are kept (this matches the TU Delft file, which is 2013 only).
ANALYSIS_START = "2013-01-01 00:00"
ANALYSIS_END = "2013-12-31 23:30"
STUCK_RUN_HALFHOURS = 12    # 6 hours or more of the exact same reading = filled-in gap, not real usage
# --------------------------------------------------------------------------


def find_input():
    if INPUT_FILE:
        return Path(INPUT_FILE)
    files = [f for f in DATA_DIR.iterdir() if f.suffix.lower() in (".csv", ".tsf", ".ts")] if DATA_DIR.exists() else []
    files = sorted(files, key=lambda f: f.stat().st_size, reverse=True)
    if not files:
        raise SystemExit(f"No .csv, .ts or .tsf file found in {DATA_DIR}. Download the data and save it there "
                         "(unzip it first if it is a .zip).")
    return files[0]


def load_tsf(path):
    """Read a .ts / .tsf file (Monash format): lines of  series_name:start_time:v1,v2,v3,...
    Reads line by line (the file can be 1 GB). Keeps only households that cover the whole analysis
    window and whose readings are not mostly filled-in (stuck) values. Returns (table, info lines).
    The table is time x household in kWh per half hour, for a random, repeatable sample of households."""
    start = pd.Timestamp(ANALYSIS_START)
    n_win = int((pd.Timestamp(ANALYSIS_END) - start) / pd.Timedelta("30min")) + 1
    index = pd.date_range(start, periods=n_win, freq="30min")
    cap = int(N_HOUSEHOLDS * 1.2)       # keep a few extra, some drop out later
    rng = np.random.default_rng(RANDOM_SEED)

    freq = None
    in_data = False
    total = no_cover = stuck = passed = 0
    reservoir = []                      # (name, values) kept as a uniform random sample
    with open(path, encoding="cp1252", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if not in_data:
                if line.lower().startswith("@frequency"):
                    freq = line.split(None, 1)[1].strip().lower()
                    if not ("half" in freq or "30" in freq):
                        raise SystemExit(f"This file has frequency '{freq}'. The script expects half-hourly data.")
                if line.lower().startswith("@data"):
                    in_data = True
                continue

            total += 1
            parts = line.split(":")
            name, values = parts[0], parts[-1]
            t0 = ":".join(parts[1:-1]).strip()
            try:
                t0 = pd.to_datetime(t0, format="%Y-%m-%d %H-%M-%S")
            except ValueError:
                t0 = pd.to_datetime(t0)
            offset = int(round((start - t0.floor("30min")) / pd.Timedelta("30min")))
            n_vals = values.count(",") + 1
            if offset < 0 or offset + n_win > n_vals:
                no_cover += 1
                continue

            toks = values.split(",")[offset:offset + n_win]
            try:
                arr = np.array(toks, dtype=np.float32)
            except ValueError:
                arr = pd.to_numeric(pd.Series(toks), errors="coerce").values.astype(np.float32)

            # share of readings that sit in long runs of the exact same value (= carried-forward gaps)
            change = np.flatnonzero(np.diff(arr) != 0)
            runs = np.diff(np.r_[-1, change, len(arr) - 1])
            stuck_share = runs[runs >= STUCK_RUN_HALFHOURS].sum() / len(arr)
            if np.isnan(arr).mean() > MAX_MISSING_FRACTION or stuck_share > MAX_MISSING_FRACTION:
                stuck += 1
                continue

            passed += 1
            if len(reservoir) < cap:
                reservoir.append((name, arr))
            else:
                j = int(rng.integers(0, passed))
                if j < cap:
                    reservoir[j] = (name, arr)

    info = [
        f"Households in the file: {total}",
        f"Dropped, do not cover {ANALYSIS_START[:10]} to {ANALYSIS_END[:10]}: {no_cover}",
        f"Dropped, more than {MAX_MISSING_FRACTION:.0%} filled-in or missing readings: {stuck}",
        f"Households passing both checks: {passed}",
        f"Random sample taken for the next steps: {len(reservoir)}",
    ]
    if not reservoir:
        raise SystemExit("No household passed the checks. Check ANALYSIS_START and ANALYSIS_END.")
    table = pd.DataFrame({name: arr for name, arr in reservoir}, index=index)
    return table, info


def load_data(path):
    """Return a wide table: datetime index, one column per household, values in kWh per half hour.
    Handles two layouts:
      wide: first column = datetime, other columns = households (TU Delft style)
      long: columns like LCLid, tstp, energy(kWh/hh) (original London style)
    """
    print(f"Loading {path} ...")
    if Path(path).suffix.lower() in (".tsf", ".ts"):
        wide, info = load_tsf(path)
        wide.index.name = "time"
        wide.columns = wide.columns.astype(str)
        return wide, info
    df = pd.read_csv(path, low_memory=False)
    lower = {c.lower(): c for c in df.columns}

    id_col = next((lower[c] for c in ("lclid", "household", "household_id", "id") if c in lower), None)
    val_col = next((c for c in df.columns if "energy" in c.lower() or "kwh" in c.lower()), None)
    time_col = next((lower[c] for c in ("tstp", "datetime", "timestamp", "time", "date") if c in lower), None)

    if id_col and val_col and time_col:
        print("Detected LONG format (household id, time, energy).")
        df[time_col] = pd.to_datetime(df[time_col], errors="coerce", dayfirst=DAYFIRST)
        df[val_col] = pd.to_numeric(df[val_col], errors="coerce")
        df = df.dropna(subset=[time_col])
        wide = df.pivot_table(index=time_col, columns=id_col, values=val_col, aggfunc="mean")
    else:
        print("Detected WIDE format (first column = time, other columns = households).")
        time_col = time_col or df.columns[0]
        df[time_col] = pd.to_datetime(df[time_col], errors="coerce", dayfirst=DAYFIRST)
        df = df.dropna(subset=[time_col]).set_index(time_col)
        wide = df.apply(pd.to_numeric, errors="coerce")

    wide.index.name = "time"
    wide.columns = wide.columns.astype(str)
    wide = wide.sort_index()
    wide = wide[~wide.index.duplicated(keep="first")]
    wide = wide.dropna(axis=1, how="all")
    # make every half hour explicit so missing readings show up as NaN
    wide = wide.resample("30min").mean()
    return wide, []


def clean(wide_kwh):
    log = []
    log.append(f"Households loaded: {wide_kwh.shape[1]}")
    log.append(f"Period: {wide_kwh.index.min()} to {wide_kwh.index.max()}")
    log.append(f"Half-hour rows: {len(wide_kwh)}")

    # kWh per half hour -> average kW
    kw = wide_kwh * 2

    # negative readings are errors
    n_neg = int((kw < 0).sum().sum())
    kw = kw.mask(kw < 0)
    log.append(f"Negative readings set to missing: {n_neg}")

    # drop households with too many missing readings
    missing = kw.isna().mean()
    keep = missing[missing <= MAX_MISSING_FRACTION].index
    log.append(f"Dropped for >{MAX_MISSING_FRACTION:.0%} missing: {kw.shape[1] - len(keep)}")
    kw = kw[keep]

    # fill short gaps only
    kw = kw.interpolate(limit=MAX_GAP_HALFHOURS, limit_area="inside")

    # drop households that are (almost) always zero or constant
    flat = kw.std() < 1e-6
    log.append(f"Dropped for flat or zero usage: {int(flat.sum())}")
    kw = kw.loc[:, ~flat]

    log.append(f"Households after cleaning: {kw.shape[1]}")
    return kw, log


def build_profiles(kw):
    """Hourly averages, then each household's average load for each of the 24 hours."""
    hourly = kw.resample("1h").mean()
    profiles = hourly.groupby(hourly.index.hour).mean().T   # households x 24
    profiles.columns = list(range(24))
    profiles.index.name = "household_id"
    profiles = profiles.dropna(how="any")
    profiles = profiles[profiles.mean(axis=1) > 0]
    return hourly, profiles


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)

    wide, info = load_data(find_input())
    kw, log = clean(wide)
    log = info + log
    hourly, profiles = build_profiles(kw)

    # keep 200 to 500 households (random but repeatable)
    if len(profiles) > N_HOUSEHOLDS:
        profiles = profiles.sample(n=N_HOUSEHOLDS, random_state=RANDOM_SEED).sort_index()
    hourly = hourly[profiles.index]
    log.append(f"Households kept for clustering: {len(profiles)}")

    # normalize: divide each household by its own average, so clustering compares SHAPE
    normalized = profiles.div(profiles.mean(axis=1), axis=0)

    hourly.to_csv(OUTPUT_DIR / "hourly_kw.csv")
    profiles.to_csv(OUTPUT_DIR / "profiles_kw.csv")
    normalized.to_csv(OUTPUT_DIR / "profiles_normalized.csv")
    (OUTPUT_DIR / "cleaning_summary.txt").write_text("\n".join(log))

    print("\n".join(log))
    print(f"\nSaved to {OUTPUT_DIR.resolve()}")
    print("Peak hour of the average household:", int(profiles.mean().idxmax()), "h")


if __name__ == "__main__":
    main()
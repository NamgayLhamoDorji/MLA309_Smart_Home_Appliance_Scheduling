"""
Reads MEMBER 1's result files (cluster_centroids.csv, optionally household files)
so you can run stages 2 and 3 without the raw London data.
"""
import re
import numpy as np, pandas as pd

HOURS = [f"h{h:02d}" for h in range(24)]

def _pick(cols, names):
    low = {c.lower().strip(): c for c in cols}
    for n in names:
        if n in low: return low[n]
    return None

def _hour_columns(df):
    found = {}
    for c in df.columns:
        m = re.fullmatch(r"(?:h|hr|hour|hour_)?\s*(\d{1,2})(?::00)?", str(c).strip().lower())
        if m and 0 <= int(m.group(1)) <= 23: found[int(m.group(1))] = c
    if len(found) == 24: return [found[h] for h in range(24)]
    num = [c for c in df.select_dtypes("number").columns
           if str(c).lower() not in ("cluster", "cluster_id", "label", "size", "n", "count", "n_households")]
    if len(num) == 24: return num
    raise SystemExit(f"Could not find 24 hourly columns in the centroids file. Columns are: {list(df.columns)}\n"
                     "Rename them to h00..h23 (or 0..23) and run again.")

def load_centroids(path):
    df = pd.read_csv(path)
    hcols = _hour_columns(df)
    id_c = _pick(df.columns, ["cluster", "cluster_id", "label", "cluster_label"])
    nm_c = _pick(df.columns, ["name", "cluster_name", "group", "group_name", "description"])
    sz_c = _pick(df.columns, ["size", "n", "count", "n_households", "households", "cluster_size"])
    ids = df[id_c].astype(int).values if id_c else np.arange(len(df))
    cent = pd.DataFrame(df[hcols].values.astype(float), columns=HOURS, index=pd.Index(ids, name="cluster"))
    cent.insert(0, "name", df[nm_c].astype(str).values if nm_c else [f"Cluster {i}" for i in ids])
    cent.insert(1, "size", df[sz_c].astype(int).values if sz_c else 50)
    v = cent[HOURS].values
    print(f"[adapter] {len(cent)} clusters; centroid values min {v.min():.3f}, max {v.max():.3f}, mean {v.mean():.3f}")
    if v.min() < 0:
        raise SystemExit("Centroids contain negative values, so they are standardised, not kW.\n"
                         "Ask Member 1 for the centroids in kW (average of the real kW profiles per cluster).")
    if v.max() > 10:
        print("[adapter] WARNING: values above 10 look too large for kW per household. Check the units (maybe Wh?).")
    if v.mean() < 0.02:
        print("[adapter] WARNING: values are very small. Are they kWh per half hour? Multiply by 2 to get kW.")
    return cent

def make_households(cent, households_path=None, profiles_path=None):
    """Real households if both files are given, else one copy of the centroid per household."""
    if households_path and profiles_path:
        hh = pd.read_csv(households_path)
        hid = _pick(hh.columns, ["household_id", "household", "id", "lclid", "meter"]) or hh.columns[0]
        cl = _pick(hh.columns, ["cluster", "cluster_id", "label", "cluster_label"])
        hh = pd.DataFrame({"household_id": hh[hid].astype(str), "cluster": hh[cl].astype(int)})
        pr = pd.read_csv(profiles_path, index_col=0); pr.index = pr.index.astype(str)
        pr = pr[_hour_columns(pr)]; pr.columns = HOURS
        hh = hh[hh.household_id.isin(pr.index)]
        return hh, pr.loc[hh.household_id]
    ids, rows, cls = [], [], []
    for c in cent.index:
        for i in range(int(cent.loc[c, "size"])):
            ids.append(f"c{c}_{i}"); cls.append(int(c)); rows.append(cent.loc[c, HOURS].values.astype(float))
    prof = pd.DataFrame(rows, columns=HOURS, index=pd.Index(ids, name="household_id"))
    hh = pd.DataFrame({"household_id": ids, "cluster": cls})
    print("[adapter] no household files given: each household is represented by its cluster centroid")
    return hh, prof

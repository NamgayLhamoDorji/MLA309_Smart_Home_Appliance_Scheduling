"""
Runs the WHOLE pipeline and writes everything to outputs/.
    python run_all.py --data data/your_file.csv
    python run_all.py --centroids results/cluster_centroids.csv   (uses Member 1's files)
    python run_all.py --synthetic          (test run on fake data - NEVER report these numbers)
"""
import argparse, json, os
import numpy as np, pandas as pd
import config as C
import data_prep, stage1_clustering as s1, stage2_fuzzy as s2, stage3_ga as s3

ap = argparse.ArgumentParser()
ap.add_argument("--data", default=None)
ap.add_argument("--synthetic", action="store_true")
ap.add_argument("--centroids", default=None, help="Member 1 cluster_centroids.csv (skips raw data and stage 1)")
ap.add_argument("--households", default=None, help="optional: household cluster file")
ap.add_argument("--profiles", default=None, help="optional: household x 24 kW profile file")
ap.add_argument("--quick", action="store_true", help="fewer seeds, skip weight sweep (fast check)")
args = ap.parse_args()
if args.synthetic:
    if not os.path.exists("data/synthetic_wide.csv"):
        import subprocess, sys; subprocess.run([sys.executable, "make_synthetic_data.py"], check=True)
    args.data = "data/synthetic_wide.csv"
if args.quick: C.N_SEEDS = 3
os.makedirs(C.FIG_DIR, exist_ok=True)

if args.centroids:                                               # use MEMBER 1's files
    import adapter
    cent = adapter.load_centroids(args.centroids)
    hh, profiles = adapter.make_households(cent, args.households, args.profiles)
    k_info = dict(k=len(cent), silhouette=float("nan"))          # take silhouette from Member 1's cluster_summary
else:
    profiles = data_prep.prepare(args.data)                      # Steps 1-2
    profiles.to_csv(os.path.join(C.OUT_DIR, "household_profiles_kw.csv"))
    cent, hh, k_info = s1.run(profiles)                          # Stage 1  (member 1)
s2.run(cent[s3.HOURS].values.max() * 1.1)                        # Stage 2  (member 2)
df, best, hist = s3.run(cent, hh, profiles)                      # Stage 3  (GA + ablation)
summ = s3.overall(df)
s3.plot_all(df, summ, best, hist, cent, profiles, hh)
sweep = None if args.quick else s3.weight_sweep(cent, hh, profiles)

df.to_csv(os.path.join(C.OUT_DIR, "ablation_raw.csv"), index=False)
summ.to_csv(os.path.join(C.OUT_DIR, "ablation_summary.csv"), index=False)
per_cl = df.groupby(["cluster", "variant"], sort=False)[["peak_red", "par", "delay", "inconvenience"]].agg(["mean", "std"]).round(3)
per_cl.to_csv(os.path.join(C.OUT_DIR, "ablation_per_cluster.csv"))

# best schedule of the full hybrid for each cluster
sched = []
for cl in cent.index:
    st, fit, win = best[("Clustering + Fuzzy + GA", cl)]
    for i, a in enumerate(C.APPLIANCES):
        sched.append(dict(cluster=cl, cluster_name=cent.loc[cl, "name"], appliance=a["name"], preferred_h=a["preferred"],
                          allowed_window_h=round(float(win[i]), 1), optimized_start_h=int(st[i]), delay_h=int(s3.delays(st)[i])))
pd.DataFrame(sched).to_csv(os.path.join(C.OUT_DIR, "best_schedule.csv"), index=False)

# markdown tables for the report
def md(d): return d.to_markdown(index=False)
try:
    base = summ.iloc[0]
    t = pd.DataFrame({"Variant": summ.variant,
        "Peak reduction (%)": [f"{m:.1f} ± {s:.1f}" for m, s in zip(summ.peak_red_mean, summ.peak_red_std)],
        "Peak-to-average ratio": [f"{m:.2f} ± {s:.2f}" for m, s in zip(summ.par_mean, summ.par_std)],
        "Average delay (h)": [f"{m:.2f} ± {s:.2f}" for m, s in zip(summ.delay_mean, summ.delay_std)],
        "Urgency-weighted delay": [f"{m:.2f} ± {s:.2f}" for m, s in zip(summ.inconvenience_mean, summ.inconvenience_std)]})
    open(os.path.join(C.OUT_DIR, "results_tables.md"), "w").write(
        f"Baseline (no scheduling): peak-to-average ratio {base.par_base_mean:.2f}\n\n" + md(t) +
        f"\n\nSilhouette (k={k_info['k']}): {k_info['silhouette']:.3f}\n\n" + md(pd.DataFrame(sched)))
except ImportError:
    pass
json.dump(dict(k=k_info["k"], silhouette=k_info["silhouette"], n_households=len(profiles)),
          open(os.path.join(C.OUT_DIR, "run_info.json"), "w"))
print("\n=== OVERALL (mean +- std over seeds) ===")
print(summ.round(2).to_string(index=False))
print(f"\nBaseline PAR (unscheduled): {summ.par_base_mean.iloc[0]:.2f}\nAll files are in: {C.OUT_DIR}/")

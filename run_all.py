"""
MEMBER 4 - integration, evaluation and ablation.
Runs Stage 2 (fuzzy) and Stage 3 (GA) on MEMBER 1's real result files and writes everything to outputs/.

Run Member 1's scripts first (they create the files in results/):
    python scripts/clean_london_data.py
    python scripts/run_clustering.py
Then:
    python run_all.py              # full run: 10 seeds per variant + weight sweep (a few minutes)
    python run_all.py --quick      # 3 seeds, no weight sweep (fast check)

Reads  (results/): cluster_centroids.csv, cluster_summary.csv, household_clusters.csv,
                   profiles_kw.csv, clustering_summary.txt
Writes (outputs/): ablation_raw.csv, ablation_summary.csv, ablation_per_cluster.csv, best_schedule.csv,
                   results_tables.md, fuzzy_rules.txt, run_info.json, weight_sweep.csv, figures/fig3-fig8
"""
import argparse, json, os, re
import numpy as np, pandas as pd
import config as C
import adapter, stage2_fuzzy as s2, stage3_ga as s3

ap = argparse.ArgumentParser()
ap.add_argument("--results", default="results", help="folder with Member 1's files")
ap.add_argument("--quick", action="store_true", help="3 seeds and no weight sweep (fast check)")
ap.add_argument("--seeds", type=int, default=None, help="GA seeds per variant (default from config.py)")
args = ap.parse_args()

R = args.results
need = ["cluster_centroids.csv", "household_clusters.csv", "profiles_kw.csv"]
missing = [f for f in need if not os.path.exists(os.path.join(R, f))]
if missing:
    raise SystemExit(f"Missing in {R}/: {missing}\nRun scripts/clean_london_data.py and scripts/run_clustering.py first.")

if args.quick: C.N_SEEDS = 3
if args.seeds: C.N_SEEDS = args.seeds
os.makedirs(C.FIG_DIR, exist_ok=True)

# ------------------------------------------------------------- Member 1's files
cent = adapter.load_centroids(os.path.join(R, "cluster_centroids.csv"))
hh, profiles = adapter.make_households(cent, os.path.join(R, "household_clusters.csv"),
                                       os.path.join(R, "profiles_kw.csv"))
sizes = hh.cluster.value_counts()
for c in cent.index: cent.loc[c, "size"] = int(sizes.get(c, 0))
print(f"[run_all] {len(hh)} real households in {len(cent)} clusters: {sizes.sort_index().to_dict()}")

sil, k = float("nan"), len(cent)
summ_txt = os.path.join(R, "clustering_summary.txt")
if os.path.exists(summ_txt):
    m = re.search(r"Silhouette score \(K-means, k=(\d+)\):\s*([\d.]+)", open(summ_txt, encoding="utf-8").read())
    if m: k, sil = int(m.group(1)), float(m.group(2))

# ------------------------------------------------------------------ Stage 2 and 3
load_max = s2.load_range_from_profiles(profiles[s3.HOURS].values)     # 'high load' = 95th percentile of household-hours
s2.run(load_max)
df, best, hist = s3.run(cent, hh, profiles)
summ = s3.overall(df)
s3.plot_all(df, summ, best, hist, cent, profiles, hh)
sweep = None if args.quick else s3.weight_sweep(cent, hh, profiles)

# ---------------------------------------------------------------------- outputs
df.to_csv(os.path.join(C.OUT_DIR, "ablation_raw.csv"), index=False)
summ.to_csv(os.path.join(C.OUT_DIR, "ablation_summary.csv"), index=False)
per_cl = df.groupby(["cluster", "variant"], sort=False)[["peak_red", "par", "delay", "inconvenience"]].agg(["mean", "std"]).round(3)
per_cl.to_csv(os.path.join(C.OUT_DIR, "ablation_per_cluster.csv"))

sched = []                                                       # best schedule of the full hybrid per cluster
for cl in cent.index:
    st, fit, win = best[("Clustering + Fuzzy + GA", cl)]
    for i, a in enumerate(C.APPLIANCES):
        sched.append(dict(cluster=cl, cluster_name=cent.loc[cl, "name"], appliance=a["name"], urgency=a["urgency"],
                          preferred_h=a["preferred"], allowed_window_h=round(float(win[i]), 1),
                          optimized_start_h=int(st[i]), delay_h=int(s3.delays(st)[i])))
sched = pd.DataFrame(sched)
sched.to_csv(os.path.join(C.OUT_DIR, "best_schedule.csv"), index=False)

pm = lambda m, s, f=".1f": [format(a, f) + " ± " + format(b, f) for a, b in zip(m, s)]
table = pd.DataFrame({
    "Variant": summ.variant,
    "Peak reduction (%)": pm(summ.peak_red_mean, summ.peak_red_std),
    "Peak-to-average ratio": pm(summ.par_mean, summ.par_std, ".2f"),
    "Average delay (h)": pm(summ.delay_mean, summ.delay_std, ".2f"),
    "Urgency-weighted delay": pm(summ.inconvenience_mean, summ.inconvenience_std, ".2f")})
cl_tab = pd.DataFrame({"Cluster": cent.index, "Name": cent["name"].values, "Households": cent["size"].values})
try:
    md = (f"Baseline (no scheduling): peak-to-average ratio {summ.par_base_mean.iloc[0]:.2f}\n\n"
          f"Mean ± std over {C.N_SEEDS} GA seeds, household-weighted over {len(hh)} real households.\n\n"
          + table.to_markdown(index=False) + f"\n\nSilhouette (K-means, k={k}): {sil:.3f}\n\n"
          + cl_tab.to_markdown(index=False) + "\n\n" + sched.to_markdown(index=False) + "\n")
    with open(os.path.join(C.OUT_DIR, "results_tables.md"), "w", encoding="utf-8") as f: f.write(md)
except ImportError:
    print("[run_all] pip install tabulate to also write results_tables.md")

json.dump(dict(k=k, silhouette=sil, n_households=len(profiles), n_seeds=C.N_SEEDS,
               fuzzy_load_universe_kw=round(load_max, 3), load_percentile=C.LOAD_PERCENTILE),
          open(os.path.join(C.OUT_DIR, "run_info.json"), "w"))

print("\n=== OVERALL (mean ± std over seeds) ===")
print(table.to_string(index=False))
print(f"\nBaseline PAR (unscheduled): {summ.par_base_mean.iloc[0]:.2f}   Silhouette: {sil:.3f}\nAll files are in: {C.OUT_DIR}/")

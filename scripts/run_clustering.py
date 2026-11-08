"""
Stage 1 - Steps 3 and 4: K-means clustering, choose k, name the groups.

Run from the repo root, after clean_london_data.py:
    python scripts/run_clustering.py

Reads  (results/):  profiles_normalized.csv, profiles_kw.csv
Writes (results/):
  cluster_centroids.csv    HAND-OVER to Member 2: one row per cluster, hours 0-23, average kW
  household_clusters.csv   HAND-OVER: household_id, cluster, cluster_name
  cluster_selection.csv    k, inertia, silhouette for k = 2..10
  cluster_summary.csv      size, mean kW, peak hour, name per cluster
  elbow_plot.png, silhouette_plot.png, centroids_plot.png, cluster_sizes.png
  clustering_summary.txt   numbers to copy into the report
"""

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import silhouette_score

# ------------------------- SETTINGS (edit these) -------------------------
ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
K_RANGE = range(2, 11)       # k = 2 to 10
CLUSTER_ON = "normalized"    # "normalized" = group by daily SHAPE, "kw" = shape AND size (finds low-use group)
K_OVERRIDE = None            # set e.g. 4 to force k after looking at the elbow and silhouette plots
MIN_K_AUTO = 3               # automatic choice ignores k=2 (usually too coarse to name useful groups)
RANDOM_SEED = 42
NAME_OVERRIDES = {}          # e.g. {0: "evening-peak", 1: "daytime-use"} after you look at the plots
# --------------------------------------------------------------------------


def load_profiles():
    kw = pd.read_csv(RESULTS / "profiles_kw.csv", index_col="household_id")
    norm = pd.read_csv(RESULTS / "profiles_normalized.csv", index_col="household_id")
    kw.columns = kw.columns.astype(int)
    norm.columns = norm.columns.astype(int)
    kw.index = kw.index.astype(str)
    norm.index = norm.index.astype(str)
    return kw, norm.loc[kw.index]


def evaluate_k(X):
    rows = []
    for k in K_RANGE:
        km = KMeans(n_clusters=k, n_init=10, random_state=RANDOM_SEED).fit(X)
        rows.append({"k": k, "inertia": km.inertia_, "silhouette": silhouette_score(X, km.labels_)})
    return pd.DataFrame(rows)


def choose_k(table):
    if K_OVERRIDE:
        return int(K_OVERRIDE)
    eligible = table[table["k"] >= MIN_K_AUTO]
    return int(eligible.loc[eligible["silhouette"].idxmax(), "k"])


def suggest_names(centroids, sizes):
    """Suggest a name from each centroid's shape. Check against the plot and edit if needed."""
    means = centroids.mean(axis=1)
    peak_hour = centroids.idxmax(axis=1)
    overall = (means * sizes).sum() / sizes.sum()   # average over households, not over clusters
    names = {}
    for c in centroids.index:
        h = int(peak_hour[c])
        if means[c] < 0.75 * overall:
            base = "low-use"
        elif 17 <= h <= 22:
            base = "evening-peak"
        elif 6 <= h <= 10:
            base = "morning-peak"
        elif 11 <= h <= 16:
            base = "daytime-use"
        else:
            base = "night-use"
        if means[c] > 1.4 * overall and base != "low-use":
            base = "high-use " + base.replace("-use", "")
        names[c] = base
    # if two clusters got the same name, add their peak hour so they can be told apart
    dup = pd.Series(names).value_counts()
    for c in centroids.index:
        if dup[names[c]] > 1:
            names[c] = f"{names[c]} (peak {int(peak_hour[c]):02d}:00)"
    names.update(NAME_OVERRIDES)
    return names


def make_plots(table, best_k, centroids, names, sizes):
    # elbow
    plt.figure(figsize=(6, 4))
    plt.plot(table["k"], table["inertia"], "o-")
    plt.axvline(best_k, color="red", ls="--", label=f"chosen k = {best_k}")
    plt.xlabel("Number of clusters (k)"); plt.ylabel("Inertia"); plt.title("Elbow plot")
    plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(RESULTS / "elbow_plot.png", dpi=200); plt.close()

    # silhouette
    plt.figure(figsize=(6, 4))
    plt.plot(table["k"], table["silhouette"], "o-", color="green")
    plt.axvline(best_k, color="red", ls="--", label=f"chosen k = {best_k}")
    plt.xlabel("Number of clusters (k)"); plt.ylabel("Silhouette score"); plt.title("Silhouette score by k")
    plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(RESULTS / "silhouette_plot.png", dpi=200); plt.close()

    # centroids
    plt.figure(figsize=(8, 4.5))
    for c in centroids.index:
        plt.plot(centroids.columns, centroids.loc[c], marker="o", ms=3,
                 label=f"{names[c]} (n={sizes[c]})")
    plt.xticks(range(0, 24, 2)); plt.xlabel("Hour of day"); plt.ylabel("Average load (kW)")
    plt.title("Cluster centroids: average 24-hour load"); plt.legend(fontsize=8)
    plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(RESULTS / "centroids_plot.png", dpi=200); plt.close()

    # sizes
    plt.figure(figsize=(6, 4))
    plt.bar([names[c] for c in centroids.index], [sizes[c] for c in centroids.index])
    plt.ylabel("Number of households"); plt.title("Cluster sizes")
    plt.xticks(rotation=20, ha="right"); plt.tight_layout()
    plt.savefig(RESULTS / "cluster_sizes.png", dpi=200); plt.close()


def main():
    kw, norm = load_profiles()
    X = (norm if CLUSTER_ON == "normalized" else kw).values
    print(f"Clustering {X.shape[0]} households on '{CLUSTER_ON}' profiles ...")

    table = evaluate_k(X)
    best_k = choose_k(table)
    print(table.round(3).to_string(index=False))
    print(f"\nChosen k = {best_k}")

    km = KMeans(n_clusters=best_k, n_init=10, random_state=RANDOM_SEED).fit(X)
    labels = pd.Series(km.labels_, index=kw.index, name="cluster")
    sil = silhouette_score(X, km.labels_)
    hier_sil = silhouette_score(X, AgglomerativeClustering(n_clusters=best_k).fit_predict(X))

    # centroids in kW: average of the members' real kW profiles (so Member 2 gets kW, not normalized values)
    centroids = kw.groupby(labels).mean()
    centroids.index.name = "cluster"
    sizes = labels.value_counts().sort_index()
    names = suggest_names(centroids, sizes)

    # hand-over files
    centroids.to_csv(RESULTS / "cluster_centroids.csv")
    hh = labels.to_frame()
    hh["cluster_name"] = hh["cluster"].map(names)
    hh.index.name = "household_id"
    hh.to_csv(RESULTS / "household_clusters.csv")

    table.to_csv(RESULTS / "cluster_selection.csv", index=False)
    summary = pd.DataFrame({
        "name": pd.Series(names),
        "households": sizes,
        "mean_kw": centroids.mean(axis=1).round(3),
        "peak_hour": centroids.idxmax(axis=1),
        "peak_kw": centroids.max(axis=1).round(3),
    })
    summary.index.name = "cluster"
    summary.to_csv(RESULTS / "cluster_summary.csv")

    make_plots(table, best_k, centroids, names, sizes)

    text = [
        f"Households clustered: {X.shape[0]}",
        f"Clustered on: {CLUSTER_ON} profiles",
        f"k range tried: {min(K_RANGE)} to {max(K_RANGE)}",
        f"Chosen k: {best_k}",
        f"Silhouette score (K-means, k={best_k}): {sil:.3f}",
        f"Silhouette score (hierarchical, k={best_k}): {hier_sil:.3f}",
        "",
        summary.to_string(),
    ]
    (RESULTS / "clustering_summary.txt").write_text("\n".join(text))
    print("\n" + "\n".join(text))
    print(f"\nSaved to {RESULTS}")


if __name__ == "__main__":
    main()
    
       
"""
Stage 3 (+ evaluation and ablation): Genetic algorithm for appliance start times (LO6, LO7).
Chromosome = one start hour (0-23) per appliance, e.g. [21, 3, 14].
Fitness (minimise) = w_peak*peak + w_par*PAR + w_delay*mean delay + w_out*penalty for leaving the allowed window.
All terms are divided by the unscheduled baseline so the weights are comparable.
"""
import os
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import config as C
import stage2_fuzzy as fz

HOURS = [f"h{h:02d}" for h in range(24)]
APPS = C.APPLIANCES
PREF = np.array([a["preferred"] for a in APPS])
URG = np.array([C.URGENCY_VALUES[a["urgency"]] for a in APPS])
NAMES = [a["name"] for a in APPS]

# ------------------------------------------------------------------ model
def appliance_load(starts):
    load = np.zeros(24)
    for s, a in zip(starts, APPS):
        for j in range(a["duration_h"]):
            load[(int(s) + j) % 24] += a["power_kw"]
    return load

def delays(starts):
    return (np.asarray(starts) - PREF) % 24          # hours after the preferred start (forward only, 24 h cycle)

def fitness(starts, bg, windows, base):
    tot = bg + appliance_load(starts)
    peak, par = tot.max(), tot.max() / tot.mean()
    d = delays(starts)
    pen = np.maximum(0, d - windows).sum() / C.MAX_SHIFT_HOURS
    w = C.FITNESS_WEIGHTS
    return (w["peak"] * peak / base["peak"] + w["par"] * par / base["par"]
            + w["delay"] * d.mean() / C.MAX_SHIFT_HOURS + w["outside_window"] * pen)

def baseline(bg):
    tot = bg + appliance_load(PREF)
    return dict(peak=tot.max(), par=tot.max() / tot.mean())

# --------------------------------------------------------------------- GA
def run_ga(bg, windows, seed):
    g = C.GA; rng = np.random.default_rng(seed); n = len(APPS)
    base = baseline(bg); wi = np.floor(windows).astype(int)
    pop = np.array([(PREF + rng.integers(0, wi + 1)) % 24 for _ in range(g["pop_size"])])   # start inside window
    history = []
    for _ in range(g["generations"]):
        fit = np.array([fitness(p, bg, windows, base) for p in pop])
        order = np.argsort(fit); pop, fit = pop[order], fit[order]
        history.append(fit[0])
        new = [p.copy() for p in pop[:g["elite"]]]                                   # elitism
        while len(new) < g["pop_size"]:
            def tour():
                idx = rng.integers(0, len(pop), g["tournament_size"]); return pop[idx[np.argmin(fit[idx])]].copy()
            a, b = tour(), tour()
            if rng.random() < g["crossover_rate"]:                                    # uniform crossover
                m = rng.random(n) < 0.5; a[m], b[m] = b[m].copy(), a[m].copy()
            for child in (a, b):
                for i in range(n):                                                    # mutation
                    if rng.random() < g["mutation_rate"]:
                        child[i] = (child[i] + rng.choice([-3, -2, -1, 1, 2, 3])) % 24 if rng.random() < 0.7 else rng.integers(0, 24)
                new.append(child)
        pop = np.array(new[:g["pop_size"]])
    fit = np.array([fitness(p, bg, windows, base) for p in pop]); best = int(np.argmin(fit))
    return pop[best], float(fit[best]), history

# ------------------------------------------------------------ the variants
def make_windows(bg, use_fuzzy):
    if not use_fuzzy:
        return np.full(len(APPS), float(C.MAX_SHIFT_HOURS))           # no user awareness: same limit for all
    return np.array([fz.delay_window(bg[a["preferred"]], a["urgency"], a["available_h"]) for a in APPS])

VARIANTS = {   # name: (use_clustering, use_fuzzy)
    "GA only": (False, False),
    "Clustering + GA": (True, False),
    "Fuzzy + GA": (False, True),
    "Clustering + Fuzzy + GA": (True, True),
}

def evaluate(starts, H):
    """Evaluate a schedule on REAL household profiles H (n x 24). Returns mean metrics."""
    app_base, app_opt = appliance_load(PREF), appliance_load(starts)
    b, o = H + app_base, H + app_opt
    pk_b, pk_o = b.max(1), o.max(1)
    d = delays(starts)
    return dict(peak_red=float(np.mean((pk_b - pk_o) / pk_b * 100)),
                par=float(np.mean(o.max(1) / o.mean(1))), par_base=float(np.mean(b.max(1) / b.mean(1))),
                delay=float(d.mean()), inconvenience=float((URG * d).mean()))

# --------------------------------------------------------------- experiment
def run(cent, hh, profiles, weights=None, n_seeds=None, collect=True):
    if weights: C.FITNESS_WEIGHTS = weights
    n_seeds = n_seeds or C.N_SEEDS
    flat = np.full(24, float(profiles[HOURS].values.mean()))
    rows, best, hist = [], {}, {}
    sizes = hh.cluster.value_counts().to_dict()
    for cl in cent.index:
        H = profiles.loc[hh.household_id[hh.cluster == cl], HOURS].values
        centroid = cent.loc[cl, HOURS].values.astype(float)
        for vname, (use_c, use_f) in VARIANTS.items():
            bg = centroid if use_c else flat
            win = make_windows(bg, use_f)
            for s in range(n_seeds):
                st, fit, h = run_ga(bg, win, seed=1000 * s + int(cl))
                m = evaluate(st, H); m.update(variant=vname, cluster=cl, seed=s, size=sizes[cl], fit=fit)
                rows.append(m)
                if collect:
                    if (vname, cl) not in best or fit < best[(vname, cl)][1]:
                        best[(vname, cl)] = (st, fit, win)
                    if vname == "Clustering + Fuzzy + GA": hist[(cl, s)] = h
    return pd.DataFrame(rows), best, hist

def overall(df):
    """Household-weighted mean over clusters per seed, then mean +- std over seeds."""
    out = []
    for v, g in df.groupby("variant", sort=False):
        per_seed = g.groupby("seed").apply(lambda x: pd.Series({
            c: np.average(x[c], weights=x["size"]) for c in ["peak_red", "par", "par_base", "delay", "inconvenience"]}), include_groups=False)
        out.append(dict(variant=v, **{f"{c}_mean": per_seed[c].mean() for c in per_seed},
                        **{f"{c}_std": per_seed[c].std(ddof=1) for c in per_seed}))
    return pd.DataFrame(out)

# -------------------------------------------------------------------- plots
def plot_all(df, summ, best, hist, cent, profiles, hh):
    os.makedirs(C.FIG_DIR, exist_ok=True)
    clusters = list(cent.index)
    big = hh.cluster.value_counts().idxmax()
    fig, ax = plt.subplots(figsize=(7, 4))
    for (cl, s), h in hist.items():
        if cl == big: ax.plot(h, alpha=.6)
    ax.set(xlabel="Generation", ylabel="Best fitness (lower is better)", title=f"GA convergence, {cent.loc[big,'name']} ({C.N_SEEDS} seeds)")
    ax.grid(alpha=.3); fig.tight_layout(); fig.savefig(os.path.join(C.FIG_DIR, "fig5_ga_convergence.png"), dpi=150); plt.close(fig)

    ncol = min(2, len(clusters)); nrow = int(np.ceil(len(clusters) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(6.2 * ncol, 3.6 * nrow), squeeze=False)
    for a in axes.ravel(): a.axis("off")
    for a, cl in zip(axes.ravel(), clusters):
        a.axis("on"); c = cent.loc[cl, HOURS].values.astype(float)
        st = best[("Clustering + Fuzzy + GA", cl)][0]
        a.step(range(24), c + appliance_load(PREF), where="post", lw=2, color="tab:red", label="Before (preferred times)")
        a.step(range(24), c + appliance_load(st), where="post", lw=2, color="tab:green", label="After (hybrid schedule)")
        a.plot(range(24), c, "--", color="grey", lw=1, label="Background only")
        a.set(title=cent.loc[cl, "name"], xlabel="Hour", ylabel="kW", xticks=range(0, 24, 3)); a.grid(alpha=.3)
    axes[0][0].legend(fontsize=7); fig.tight_layout()
    fig.savefig(os.path.join(C.FIG_DIR, "fig6_before_after.png"), dpi=150); plt.close(fig)

    fig, ax = plt.subplots(1, 3, figsize=(13, 3.8)); x = np.arange(len(summ))
    for a, (c, t) in zip(ax, [("peak_red", "Peak load reduction (%)  higher = better"),
                              ("par", "Peak-to-average ratio  lower = better"),
                              ("delay", "Average delay (h)  lower = better")]):
        a.bar(x, summ[f"{c}_mean"], yerr=summ[f"{c}_std"], capsize=4, color=["#9aa", "#7ab", "#ca7", "#5a5"])
        a.set_xticks(x); a.set_xticklabels([v.replace(" + ", "\n+ ") for v in summ.variant], fontsize=7); a.set_title(t, fontsize=9); a.grid(axis="y", alpha=.3)
    fig.tight_layout(); fig.savefig(os.path.join(C.FIG_DIR, "fig7_ablation.png"), dpi=150); plt.close(fig)

def weight_sweep(cent, hh, profiles):
    """Step 8: tune the delay weight - shows the peak-reduction vs delay trade-off."""
    rows = []; keep = dict(C.FITNESS_WEIGHTS)
    for wd in [0.0, 0.1, 0.3, 0.6, 1.0, 2.0]:
        w = dict(keep); w["delay"] = wd
        df, _, _ = run(cent, hh, profiles, weights=w, n_seeds=3, collect=False)
        s = overall(df[df.variant == "Clustering + Fuzzy + GA"]).iloc[0]
        rows.append(dict(w_delay=wd, peak_red=s.peak_red_mean, delay=s.delay_mean))
    C.FITNESS_WEIGHTS = keep
    sw = pd.DataFrame(rows); sw.to_csv(os.path.join(C.OUT_DIR, "weight_sweep.csv"), index=False)
    fig, ax = plt.subplots(figsize=(5.5, 4))
    ax.plot(sw.delay, sw.peak_red, "o-")
    for _, r in sw.iterrows(): ax.annotate(f"w={r.w_delay:g}", (r.delay, r.peak_red), fontsize=8, xytext=(4, 4), textcoords="offset points")
    ax.set(xlabel="Average delay (h)", ylabel="Peak load reduction (%)", title="Trade-off when tuning the delay weight"); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(os.path.join(C.FIG_DIR, "fig8_weight_tradeoff.png"), dpi=150); plt.close(fig)
    return sw

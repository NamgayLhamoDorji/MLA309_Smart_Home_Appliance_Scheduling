"""
MEMBER 2 - Stage 2: Mamdani fuzzy system (LO5), written in plain numpy.
(It does exactly what scikit-fuzzy does: triangular membership functions, min for AND,
 min implication, max aggregation, centroid defuzzification - but with no extra install.)

Public function handed to stage 3:
    delay_window(load_kw, urgency, available_hours) -> allowed delay in HOURS
"""
import os
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import config as C

LOAD_MAX = 2.0                           # kW; set from the data with set_load_range()
AVAIL_MAX = 24.0
LEVELS = ["low", "medium", "high"]       # 0, 1, 2
AVAIL_LEVELS = ["short", "medium", "long"]

def set_load_range(max_kw):
    global LOAD_MAX
    LOAD_MAX = float(max_kw)

def load_range_from_profiles(profiles, percentile=None):
    """Universe of the 'expected load' input = the `percentile`-th percentile of all household x hour
    kW values (default C.LOAD_PERCENTILE). 'High' then means a load that only ~5% of household-hours reach.
    Do NOT use the largest centroid: a tiny outlier cluster would push every normal household into 'low'."""
    p = C.LOAD_PERCENTILE if percentile is None else percentile
    return float(np.percentile(np.asarray(profiles, float), p))

def trimf(x, a, b, c):
    x = np.asarray(x, float)
    left  = np.where(b > a, (x - a) / (b - a + 1e-12), (x >= a).astype(float))
    right = np.where(c > b, (c - x) / (c - b + 1e-12), (x <= c).astype(float))
    return np.clip(np.minimum(left, right), 0, 1)

def mfs():
    """Membership functions of every variable: {var: (universe, {label: (a,b,c)})}"""
    L, A = LOAD_MAX, AVAIL_MAX
    return {
        "load":    (np.linspace(0, L, 401), {"low": (0, 0, .5 * L), "medium": (.25 * L, .5 * L, .75 * L), "high": (.5 * L, L, L)}),
        "urgency": (np.linspace(0, 1, 401), {"low": (0, 0, .5), "medium": (0, .5, 1), "high": (.5, 1, 1)}),
        "available": (np.linspace(0, A, 401), {"short": (0, 0, 8), "medium": (4, 12, 20), "long": (16, 24, 24)}),
        "delay":   (np.linspace(0, 1, 401), {"low": (0, 0, .5), "medium": (0, .5, 1), "high": (.5, 1, 1)}),
    }

# ---------------------------------------------------------------- the 27 rules
# (load, urgency, available) -> delay tolerance
def build_rules():
    R = {}
    for l in LEVELS:
        for u in LEVELS:
            for a in AVAIL_LEVELS:
                if u == "high":                      # IF urgency is high THEN delay is low
                    d = "medium" if (l == "high" and a == "long") else "low"   # (only exception)
                elif l == "low":                     # IF load is low THEN delay is low (nothing to gain)
                    d = "low"
                elif l == "high" and u == "low":
                    d = {"long": "high", "medium": "high", "short": "medium"}[a]
                elif l == "high" and u == "medium":
                    d = {"long": "high", "medium": "medium", "short": "low"}[a]
                elif l == "medium" and u == "low":
                    d = {"long": "high", "medium": "medium", "short": "low"}[a]
                else:                                # medium load, medium urgency
                    d = {"long": "medium", "medium": "low", "short": "low"}[a]
                R[(l, u, a)] = d
    assert len(R) == 27
    return R
RULES = build_rules()

def rule_text():
    return [f"IF load is {l} AND urgency is {u} AND available time is {a} THEN delay tolerance is {d}"
            for (l, u, a), d in RULES.items()]

# ------------------------------------------------------------------ inference
def delay_tolerance(load_kw, urgency, available_hours):
    """Mamdani inference -> crisp delay tolerance in [0, 1]."""
    if isinstance(urgency, str): urgency = C.URGENCY_VALUES[urgency.lower()]
    M = mfs()
    mu = {
        "load":      {k: float(trimf(min(load_kw, LOAD_MAX), *p)) for k, p in M["load"][1].items()},
        "urgency":   {k: float(trimf(urgency, *p)) for k, p in M["urgency"][1].items()},
        "available": {k: float(trimf(min(available_hours, AVAIL_MAX), *p)) for k, p in M["available"][1].items()},
    }
    ux, dmf = M["delay"][0], M["delay"][1]
    agg = np.zeros_like(ux)
    for (l, u, a), d in RULES.items():
        strength = min(mu["load"][l], mu["urgency"][u], mu["available"][a])     # AND = min
        if strength > 0:
            agg = np.maximum(agg, np.minimum(strength, trimf(ux, *dmf[d])))     # implication + aggregation
    return float(np.sum(ux * agg) / np.sum(agg)) if agg.sum() > 0 else 0.0     # centroid

def delay_window(load_kw, urgency, available_hours):
    """Allowed delay in hours = tolerance x MAX_SHIFT, never more than the time available."""
    tol = delay_tolerance(load_kw, urgency, available_hours)
    return float(min(tol * C.MAX_SHIFT_HOURS, available_hours))

# --------------------------------------------------------------------- plots
def plot_membership(path):
    M = mfs()
    titles = {"load": "Expected load (kW)", "urgency": "User urgency (0-1)",
              "available": "Available time (h)", "delay": "Delay tolerance (output, 0-1)"}
    fig, ax = plt.subplots(1, 4, figsize=(15, 3.3))
    for a, (v, (x, labs)) in zip(ax, M.items()):
        for lab, p in labs.items(): a.plot(x, trimf(x, *p), lw=2, label=lab)
        a.set(title=titles[v], ylim=(-.02, 1.05)); a.grid(alpha=.3); a.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)

def plot_rule_surface(path):
    fig = plt.figure(figsize=(12, 5))
    ax = fig.add_subplot(1, 2, 1, projection="3d")
    lo, ur = np.meshgrid(np.linspace(0, LOAD_MAX, 25), np.linspace(0, 1, 25))
    Z = np.vectorize(lambda l, u: delay_tolerance(l, u, 12))(lo, ur)
    ax.plot_surface(lo, ur, Z, cmap="viridis"); ax.set(xlabel="Load (kW)", ylabel="Urgency", zlabel="Delay tolerance", title="Available time = 12 h")
    ax = fig.add_subplot(1, 2, 2, projection="3d")
    lo, av = np.meshgrid(np.linspace(0, LOAD_MAX, 25), np.linspace(0, 24, 25))
    Z = np.vectorize(lambda l, a: delay_tolerance(l, 0.15, a))(lo, av)
    ax.plot_surface(lo, av, Z, cmap="viridis"); ax.set(xlabel="Load (kW)", ylabel="Available (h)", zlabel="Delay tolerance", title="Urgency = low")
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)

def run(max_load_kw):
    os.makedirs(C.FIG_DIR, exist_ok=True)
    set_load_range(max_load_kw)
    plot_membership(os.path.join(C.FIG_DIR, "fig3_membership_functions.png"))
    plot_rule_surface(os.path.join(C.FIG_DIR, "fig4_rule_surface.png"))
    with open(os.path.join(C.OUT_DIR, "fuzzy_rules.txt"), "w") as f:
        f.write("\n".join(f"R{i+1}: {t}" for i, t in enumerate(rule_text())))
    print(f"[stage2] 27 rules built, load universe 0-{LOAD_MAX:.2f} kW; "
          f"example delay_window(high load, low urgency, 12h) = {delay_window(LOAD_MAX*0.9, 'low', 12):.1f} h")

if __name__ == "__main__":
    run(2.0)

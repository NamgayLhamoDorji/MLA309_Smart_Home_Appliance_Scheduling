"""
Checks for Stage 2 (fuzzy) and Stage 3 (GA). Run from the repo root:
    python tests/test_stages.py
No test framework needed. Every line that starts with PASS is a check that held.
"""
import itertools, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as C
import stage2_fuzzy as fz
import stage3_ga as ga

def ok(cond, msg):
    print(("PASS  " if cond else "FAIL  ") + msg)
    if not cond: sys.exit(1)

# ------------------------------------------------------------------ fuzzy
fz.set_load_range(1.0)
ok(len(fz.RULES) == 27 and len(set(fz.RULES)) == 27, "27 distinct rules (3 x 3 x 3)")
ok(set(fz.RULES.values()) <= {"low", "medium", "high"}, "every rule outputs low, medium or high")

grid = [(l, u, a) for l in np.linspace(0, 1.2, 13) for u in np.linspace(0, 1, 11) for a in np.linspace(0, 24, 13)]
tol = np.array([fz.delay_tolerance(l, u, a) for l, u, a in grid])
ok(tol.min() >= 0 and tol.max() <= 1, f"delay tolerance always in [0, 1] (seen {tol.min():.2f} to {tol.max():.2f})")
win_ok = all(fz.delay_window(l, u, a) <= min(a, C.MAX_SHIFT_HOURS) + 1e-9 for l, u, a in grid)
ok(win_ok, "allowed window never exceeds the available time or MAX_SHIFT_HOURS")

for l in np.linspace(0, 1, 6):                         # urgent appliances must not be allowed more delay than relaxed ones
    for a in (2, 8, 16, 24):
        ok(fz.delay_tolerance(l, 0.9, a) <= fz.delay_tolerance(l, 0.1, a) + 1e-9,
           f"urgency 0.9 gets no more delay than urgency 0.1 (load {l:.1f} kW, available {a} h)")
for a in (12, 24):                                     # a busier hour gives more reason to move
    ok(fz.delay_tolerance(1.0, 0.1, a) >= fz.delay_tolerance(0.0, 0.1, a), f"high load gets at least as much delay as zero load (available {a} h)")
ok(fz.delay_tolerance(1.0, 0.1, 24) > fz.delay_tolerance(0.2, 0.9, 4) + 0.2,
   "relaxed appliance at a busy hour gets clearly more delay than an urgent one at a quiet hour")

# --------------------------------------------------------------------- GA
ok(abs(ga.appliance_load(ga.PREF).sum() - sum(a["power_kw"] * a["duration_h"] for a in C.APPLIANCES)) < 1e-9,
   "appliance energy (kWh) is the same wherever the appliances start")
ok(np.allclose(ga.appliance_load([23, 0, 0, 0])[[23, 0]], [0.8, 5.5]), "an appliance running past midnight wraps to hour 0")
ok(ga.delays(ga.PREF).sum() == 0, "no delay at the preferred start times")

rng = np.random.default_rng(1)
bg = 0.3 + 0.4 * np.exp(-((np.arange(24) - 19) / 2.5) ** 2) + rng.normal(0, 0.01, 24)     # a made-up evening-peak household
for win in (np.full(4, 3.0), np.full(4, 6.0)):
    base = ga.baseline(bg)
    ranges = [range(int(p), int(p) + int(w) + 1) for p, w in zip(ga.PREF, win)]
    bf = min(ga.fitness(np.array(s) % 24, bg, win, base) for s in itertools.product(*ranges))
    results = [ga.run_ga(bg, win, seed=s) for s in range(5)]
    best = min(results, key=lambda r: r[1])
    ok(abs(best[1] - bf) < 1e-9, f"GA finds the brute-force optimum (windows {win[0]:.0f} h, best of 5 seeds)")
    ok(all(r[1] <= ga.fitness(ga.PREF, bg, win, base) + 1e-9 for r in results), "GA is never worse than the unscheduled times")
    ok(all((ga.delays(r[0]) <= np.floor(win)).all() for r in results), "GA schedules stay inside the allowed windows")
    ok(all(h[-1] <= h[0] + 1e-12 for _, _, h in results), "best fitness never gets worse over the generations")

print("\nAll checks passed.")

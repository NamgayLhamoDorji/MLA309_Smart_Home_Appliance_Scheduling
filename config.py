"""
Central settings for the whole project. Change things HERE, not inside the other files.
"""
import os

# ---------------------------------------------------------------- data
DATA_PATH = "data/low_carbon_london.csv"   # a CSV file, or a folder of CSV files
YEAR = 2013                                # use one full year only
N_HOUSEHOLDS = 300                         # 200-500 as in the plan
MAX_MISSING = 0.10                         # drop households with >10% missing readings
RANDOM_STATE = 42

# ------------------------------------------------------------- clustering
K_RANGE = range(2, 11)                     # k = 2..10
K_CHOSEN = None                            # None = automatic (best silhouette for k in AUTO_K_RANGE)
AUTO_K_RANGE = range(3, 7)                 # we want at least 3 groups so there is something to interpret

# ------------------------------------------------------------------ fuzzy
MAX_SHIFT_HOURS = 12                       # tolerance 1.0 = may move 12 h
LOAD_PERCENTILE = 95                       # "high load" = the load that only 5% of household-hours exceed.
                                           # (NOT the largest centroid: one 5-household cluster would make
                                           #  every normal household look "low load".)
URGENCY_VALUES = {"low": 0.15, "medium": 0.50, "high": 0.85}   # numeric urgency on a 0-1 scale

# --------------------------------------------------- appliances (Step 0)
# ASSUMPTIONS - the data has no appliance readings, so state these in the report.
# power_kw, duration_h (whole hours), preferred start hour, urgency, available_h =
# hours between the preferred start and the time the appliance result is needed.
APPLIANCES = [
    {"name": "Washing machine", "power_kw": 0.8, "duration_h": 2, "preferred": 18, "urgency": "medium", "available_h": 10},
    {"name": "Dishwasher",      "power_kw": 1.2, "duration_h": 2, "preferred": 20, "urgency": "low",    "available_h": 12},
    {"name": "Tumble dryer",    "power_kw": 2.0, "duration_h": 1, "preferred": 19, "urgency": "medium", "available_h": 8},
    {"name": "Water heater",    "power_kw": 1.5, "duration_h": 2, "preferred": 19, "urgency": "high",   "available_h": 4},
]
TARIFF_NU_PER_KWH = 4.0                    # flat, illustrative; NOT used as a result

# --------------------------------------------------------------------- GA
GA = dict(pop_size=60, generations=120, crossover_rate=0.8, mutation_rate=0.1,
          tournament_size=3, elite=2)
FITNESS_WEIGHTS = dict(peak=1.0, par=0.5, delay=0.3, outside_window=10.0)
N_SEEDS = 10                               # GA is stochastic: repeat each variant with several seeds

# ---------------------------------------------------------------- outputs
OUT_DIR = "outputs"
FIG_DIR = os.path.join(OUT_DIR, "figures")

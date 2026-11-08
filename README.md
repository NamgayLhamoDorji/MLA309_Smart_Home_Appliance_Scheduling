# Smart Home Appliance Scheduling (MLA309)

Hybrid Computational Intelligence project: K-means clustering of households,
Mamdani fuzzy logic for appliance delay tolerance, and a genetic algorithm
for appliance start times. One flat electricity rate, so the goal is peak load and delay, not cost.

## Data
Low Carbon London smart meter data (UK Power Networks, CC-BY). Not stored in this repo.
- https://zenodo.org/records/3898331  (file: london_smart_meters_dataset_without_missing_values.zip)
- https://research.tudelft.nl/en/datasets/low-carbon-london-smart-meter-data-refactored/

Save the unzipped file in the `data/` folder.

## Setup
    pip install -r requirements.txt

## Run (from the repo root)
    python scripts/clean_london_data.py     # Member 1: clean data, build 24-hour household profiles -> results/
    python scripts/run_clustering.py        # Member 1: K-means, cluster centroids in kW -> results/
    python tests/test_stages.py             # Members 2 and 3: checks for the fuzzy system and the GA
    python run_all.py                       # Member 4: fuzzy + GA + ablation on the real households -> outputs/
    python run_all.py --quick               # same, 3 seeds and no weight sweep (about 10 seconds)

## Pipeline and owners
| Step | File | Owner |
|---|---|---|
| Data cleaning | scripts/clean_london_data.py | Member 1 |
| Clustering | scripts/run_clustering.py | Member 1 |
| Fuzzy logic (27 rules, Mamdani, written in numpy) | stage2_fuzzy.py | Member 2 |
| Genetic algorithm | stage3_ga.py | Member 3 |
| Integration, evaluation, ablation | run_all.py, adapter.py, config.py | Member 4 |

All settings (appliances, GA parameters, fitness weights, fuzzy ranges) are in `config.py`.

## Files passed between stages
- `results/cluster_centroids.csv`: one row per cluster, hours 0-23, average kW
- `results/cluster_summary.csv`: cluster name and number of households
- `results/household_clusters.csv`: household id and cluster
- `results/profiles_kw.csv`: household x 24-hour average kW
- `outputs/`: ablation tables, best schedules, figures (fig3 to fig8), results_tables.md

## Notes for the report
- The appliances (power, duration, preferred start, urgency) are assumptions; the data has no appliance readings.
- The tariff is one flat rate, so scheduling does not change the energy cost. Do not report a cost saving.
- Std values in the results are over GA random seeds, not over households.

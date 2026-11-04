# Smart Home Appliance Scheduling (MLA309)

Hybrid Computational Intelligence project: K-means clustering of households,
Mamdani fuzzy logic for appliance delay tolerance, and a genetic algorithm
for appliance start times.

## Data
Low Carbon London smart meter data (UK Power Networks, CC-BY). Not stored in this repo.
- https://research.tudelft.nl/en/datasets/low-carbon-london-smart-meter-data-refactored/
- https://zenodo.org/records/3898331

Save the files in the `data/` folder.

## Setup
pip install -r requirements.txt

## Pipeline and owners
1. Data cleaning (scripts/clean_london_data.py): Member 1
2. Clustering (K-means): Member 1
3. Fuzzy logic delay tolerance: Member 2
4. Genetic algorithm schedule: Member 3
5. Integration, evaluation and ablation study: Member 4

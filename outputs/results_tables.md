Baseline (no scheduling): peak-to-average ratio 6.65

Mean ± std over 10 GA seeds, household-weighted over 300 real households.

| Variant                 | Peak reduction (%)   | Peak-to-average ratio   | Average delay (h)   | Urgency-weighted delay   |
|:------------------------|:---------------------|:------------------------|:--------------------|:-------------------------|
| GA only                 | 46.4 ± 0.3           | 3.51 ± 0.02             | 1.43 ± 0.13         | 0.86 ± 0.10              |
| Clustering + GA         | 53.2 ± 0.3           | 3.09 ± 0.02             | 2.43 ± 0.07         | 1.28 ± 0.14              |
| Fuzzy + GA              | 45.3 ± 2.0           | 3.58 ± 0.13             | 1.39 ± 0.15         | 0.59 ± 0.10              |
| Clustering + Fuzzy + GA | 49.2 ± 1.8           | 3.33 ± 0.12             | 1.87 ± 0.15         | 0.73 ± 0.07              |

Silhouette (K-means, k=3): 0.189

|   Cluster | Name                      |   Households |
|----------:|:--------------------------|-------------:|
|         0 | evening-peak (peak 18:00) |          189 |
|         1 | evening-peak (peak 20:00) |          106 |
|         2 | high-use night            |            5 |

|   cluster | cluster_name              | appliance       | urgency   |   preferred_h |   allowed_window_h |   optimized_start_h |   delay_h |
|----------:|:--------------------------|:----------------|:----------|--------------:|-------------------:|--------------------:|----------:|
|         0 | evening-peak (peak 18:00) | Washing machine | medium    |            18 |                3.8 |                  18 |         0 |
|         0 | evening-peak (peak 18:00) | Dishwasher      | low       |            20 |                5.8 |                  23 |         3 |
|         0 | evening-peak (peak 18:00) | Tumble dryer    | medium    |            19 |                3.9 |                  22 |         3 |
|         0 | evening-peak (peak 18:00) | Water heater    | high      |            19 |                2.3 |                  20 |         1 |
|         1 | evening-peak (peak 20:00) | Washing machine | medium    |            18 |                3.7 |                  18 |         0 |
|         1 | evening-peak (peak 20:00) | Dishwasher      | low       |            20 |                6   |                   0 |         4 |
|         1 | evening-peak (peak 20:00) | Tumble dryer    | medium    |            19 |                4.8 |                  23 |         4 |
|         1 | evening-peak (peak 20:00) | Water heater    | high      |            19 |                2.4 |                  20 |         1 |
|         2 | high-use night            | Washing machine | medium    |            18 |                6   |                  22 |         4 |
|         2 | high-use night            | Dishwasher      | low       |            20 |                7.5 |                  22 |         2 |
|         2 | high-use night            | Tumble dryer    | medium    |            19 |                6   |                  21 |         2 |
|         2 | high-use night            | Water heater    | high      |            19 |                2.3 |                  19 |         0 |

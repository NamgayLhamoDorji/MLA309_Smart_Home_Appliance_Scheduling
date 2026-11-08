Baseline (no scheduling): peak-to-average ratio 5.55

| Variant                 | Peak reduction (%)   | Peak-to-average ratio   | Average delay (h)   | Urgency-weighted delay   |
|:------------------------|:---------------------|:------------------------|:--------------------|:-------------------------|
| GA only                 | 41.9 ± 3.7           | 3.15 ± 0.15             | 1.48 ± 0.13         | 0.81 ± 0.07              |
| Clustering + GA         | 51.2 ± 0.1           | 2.65 ± 0.01             | 2.45 ± 0.23         | 1.31 ± 0.22              |
| Fuzzy + GA              | 39.6 ± 0.2           | 3.34 ± 0.01             | 1.02 ± 0.04         | 0.35 ± 0.03              |
| Clustering + Fuzzy + GA | 40.1 ± 0.2           | 3.32 ± 0.01             | 1.15 ± 0.04         | 0.45 ± 0.03              |

Silhouette (k=3): nan

|   cluster | cluster_name   | appliance       |   preferred_h |   allowed_window_h |   optimized_start_h |   delay_h |
|----------:|:---------------|:----------------|--------------:|-------------------:|--------------------:|----------:|
|         0 | Cluster 0      | Washing machine |            18 |                2.2 |                  20 |         2 |
|         0 | Cluster 0      | Dishwasher      |            20 |                2.2 |                  22 |         2 |
|         0 | Cluster 0      | Tumble dryer    |            19 |                2.3 |                  19 |         0 |
|         0 | Cluster 0      | Water heater    |            19 |                2.3 |                  20 |         1 |
|         1 | Cluster 1      | Washing machine |            18 |                2.2 |                  18 |         0 |
|         1 | Cluster 1      | Dishwasher      |            20 |                2.2 |                  22 |         2 |
|         1 | Cluster 1      | Tumble dryer    |            19 |                2.3 |                  21 |         2 |
|         1 | Cluster 1      | Water heater    |            19 |                2.3 |                  19 |         0 |
|         2 | Cluster 2      | Washing machine |            18 |                2.3 |                  20 |         2 |
|         2 | Cluster 2      | Dishwasher      |            20 |                4.7 |                  22 |         2 |
|         2 | Cluster 2      | Tumble dryer    |            19 |                2.5 |                  19 |         0 |
|         2 | Cluster 2      | Water heater    |            19 |                2.5 |                  20 |         1 |
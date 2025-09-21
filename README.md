# Cluster Profiler

A statistical package for analyzing cluster characteristics by comparing each cluster against the rest of the dataset.

## Features

- **Statistical Testing**: Uses Kolmogorov-Smirnov test for continuous features and Chi-square/Fisher's exact test for categorical features
- **Skewness Assessment**: Helps identify highly skewed features that may benefit from preprocessing
- **Preprocessing Options**: Log, quantile, and Yeo-Johnson transformations for continuous features
- **Cluster-vs-Rest Approach**: Tests each cluster against all other clusters combined

## Quick Start

```python
from cluster_profiler import ClusterProfiler
import pandas as pd

# Initialize profiler
profiler = ClusterProfiler(preprocessing='yeo-johnson')

# Assess feature skewness
skewness_report = profiler.assess_skewness(data)
print(skewness_report)

# Profile clusters
results = profiler.profile_clusters(data, cluster_labels)

# Get summary
profiler.summary()

# Get top characteristics for a specific cluster
cluster_chars = profiler.get_cluster_characteristics(cluster_id=0, top_n=10)
```

## Preprocessing Options

- `None`: No preprocessing (default)
- `'log'`: Log transformation (log1p)
- `'quantile'`: Quantile transformation to uniform distribution
- `'yeo-johnson'`: Yeo-Johnson power transformation

## Interpretation

- **Low p-values** indicate features that significantly differentiate a cluster from the rest
- **Significant results** (p < α) suggest cluster-specific characteristics
- Use skewness assessment to decide on preprocessing needs
import numpy as np
import pandas as pd
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from cluster_profiler import ClusterProfiler

# Generate example data with meaningful categorical and skewed features
np.random.seed(42)
X, true_labels = make_blobs(n_samples=300, centers=3, n_features=4, random_state=42)

# Create DataFrame
data = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(4)])

# Add categorical feature that correlates with clusters
# Cluster 0 -> mostly 'Type_A', Cluster 1 -> mostly 'Type_B', Cluster 2 -> mostly 'Type_C'
category_meaningful = []
for label in true_labels:
    if label == 0:
        category_meaningful.append(np.random.choice(['Type_A', 'Type_B'], p=[0.8, 0.2]))
    elif label == 1:
        category_meaningful.append(np.random.choice(['Type_B', 'Type_C'], p=[0.7, 0.3]))
    else:
        category_meaningful.append(np.random.choice(['Type_C', 'Type_A'], p=[0.75, 0.25]))

data['meaningful_category'] = category_meaningful

# Add skewed feature that correlates with clusters
# Different exponential parameters for different clusters
skewed_meaningful = []
for i, label in enumerate(true_labels):
    if label == 0:
        skewed_meaningful.append(np.random.exponential(1.5))  # Lower values
    elif label == 1:
        skewed_meaningful.append(np.random.exponential(3.0))  # Medium values
    else:
        skewed_meaningful.append(np.random.exponential(5.0))  # Higher values

data['meaningful_skewed'] = skewed_meaningful

# Add random features (should not be significant)
data['random_category'] = np.random.choice(['X', 'Y', 'Z'], size=300)
data['random_normal'] = np.random.normal(0, 1, 300)

# Perform clustering (use true number of clusters)
kmeans = KMeans(n_clusters=3, random_state=42)
cluster_labels = kmeans.fit_predict(X)

print("Example with Meaningful Categorical and Skewed Features")
print("="*60)

# Assess skewness
profiler = ClusterProfiler()
skewness_report = profiler.assess_skewness(data)
print("\nSkewness Assessment:")
print(skewness_report[['feature', 'skewness', 'interpretation', 'recommend_preprocessing']])
print("\n" + "="*60 + "\n")

# Profile clusters
results = profiler.profile_clusters(data, cluster_labels)
profiler.summary()

print("\nFormatted Cluster Profiles:")
for cluster_id in results['cluster'].unique():
    print(f"\nCluster {cluster_id} Characteristics:")
    print("="*40)
    cluster_data = results[results['cluster'] == cluster_id].sort_values('p_value')
    for _, row in cluster_data.iterrows():
        sig = "***" if row['p_value'] < 0.001 else "**" if row['p_value'] < 0.01 else "*" if row['p_value'] < 0.05 else ""
        print(f"  {row['feature']:20} p={row['p_value']:.4f} {sig}")

print("\n" + "="*60 + "\n")
print("Expected Results:")
print("- Blob features (feature_0-3): Should be highly significant")
print("- meaningful_category: Should be significant (different distributions per cluster)")
print("- meaningful_skewed: Should be significant (different exponential parameters)")
print("- random_category: Should NOT be significant")
print("- random_normal: Should NOT be significant")
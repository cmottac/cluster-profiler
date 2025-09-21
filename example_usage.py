import numpy as np
import pandas as pd
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from cluster_profiler import ClusterProfiler

# Generate example data
np.random.seed(42)
X, _ = make_blobs(n_samples=300, centers=4, n_features=5, random_state=42)

# Add some categorical features
categorical_data = np.random.choice(['A', 'B', 'C'], size=(300, 2))

# Create DataFrame
data = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(5)])
data['category_1'] = categorical_data[:, 0]
data['category_2'] = categorical_data[:, 1]

# Make some features skewed
data['skewed_feature'] = np.random.exponential(2, 300)
data['normal_feature'] = np.random.normal(0, 1, 300)

# Perform clustering
kmeans = KMeans(n_clusters=3, random_state=42)
cluster_labels = kmeans.fit_predict(X)

# Example 1: Assess skewness first
profiler = ClusterProfiler()
skewness_report = profiler.assess_skewness(data)
print("Skewness Assessment:")
print(skewness_report)
print("\n" + "="*50 + "\n")

# Example 2: Profile clusters without preprocessing
print("Profiling without preprocessing:")
results_no_prep = profiler.profile_clusters(data, cluster_labels)
profiler.summary()
print("\nTop results:")
print(results_no_prep.nsmallest(10, 'p_value')[['cluster', 'feature', 'p_value', 'significant']])
print("\n" + "="*50 + "\n")

# Example 3: Profile clusters with preprocessing
print("Profiling with Yeo-Johnson preprocessing:")
profiler_prep = ClusterProfiler(preprocessing='yeo-johnson')
results_prep = profiler_prep.profile_clusters(data, cluster_labels)
profiler_prep.summary()
print("\nTop results:")
print(results_prep.nsmallest(10, 'p_value')[['cluster', 'feature', 'p_value', 'significant']])
print("\n" + "="*50 + "\n")

# Example 4: Get characteristics for specific cluster
print("Characteristics of Cluster 0:")
cluster_0_chars = profiler.get_cluster_characteristics(cluster_id=0, top_n=5)
print(cluster_0_chars[['feature', 'feature_type', 'p_value', 'test']])
print("\n" + "="*50 + "\n")

# Example 5: Show formatted output by cluster (all features)
print("Formatted Cluster Profiles (All Features):")
for cluster_id in results_no_prep['cluster'].unique():
    print(f"\nCluster {cluster_id} Characteristics:")
    print("="*40)
    cluster_data = results_no_prep[results_no_prep['cluster'] == cluster_id].sort_values('p_value')
    for _, row in cluster_data.iterrows():
        sig = "***" if row['p_value'] < 0.001 else "**" if row['p_value'] < 0.01 else "*" if row['p_value'] < 0.05 else ""
        print(f"  {row['feature']:20} p={row['p_value']:.4f} {sig}")
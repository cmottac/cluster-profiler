import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import ks_2samp, chi2_contingency, fisher_exact
from sklearn.preprocessing import QuantileTransformer, PowerTransformer
try:
    from statsmodels.stats.multitest import multipletests
except ImportError:
    multipletests = None
import warnings

class ClusterProfiler:
    """
    Statistical profiler for cluster analysis that compares each cluster vs rest
    for both continuous and categorical features.
    """
    
    def __init__(self, alpha=0.05, preprocessing=None):
        """
        Parameters:
        -----------
        alpha : float, default=0.05
            Significance level for statistical tests
        preprocessing : str or None, default=None
            Preprocessing method for continuous features:
            - None: No preprocessing
            - 'log': Log transform (log1p)
            - 'quantile': Quantile transform to uniform distribution
            - 'yeo-johnson': Yeo-Johnson power transform
        """
        self.alpha = alpha
        self.preprocessing = preprocessing
        self.results_ = None
        self.skewness_report_ = None
    
    def assess_skewness(self, data, feature_cols=None):
        """
        Assess skewness of continuous features to help users decide on preprocessing.
        
        Parameters:
        -----------
        data : pd.DataFrame
            Input data
        feature_cols : list or None
            List of continuous feature columns. If None, uses numeric columns.
            
        Returns:
        --------
        pd.DataFrame : Skewness statistics for each feature
        """
        if feature_cols is None:
            feature_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        
        skew_stats = []
        for col in feature_cols:
            skew_val = stats.skew(data[col].dropna())
            # Interpretation thresholds
            if abs(skew_val) < 0.5:
                interpretation = "approximately symmetric"
            elif abs(skew_val) < 1:
                interpretation = "moderately skewed"
            else:
                interpretation = "highly skewed"
                
            skew_stats.append({
                'feature': col,
                'skewness': skew_val,
                'abs_skewness': abs(skew_val),
                'interpretation': interpretation,
                'recommend_preprocessing': abs(skew_val) > 1
            })
        
        self.skewness_report_ = pd.DataFrame(skew_stats).sort_values('abs_skewness', ascending=False)
        return self.skewness_report_
    
    def _preprocess_continuous(self, data, skewness_threshold=1.0):
        """Apply preprocessing only to highly skewed continuous features."""
        if self.preprocessing is None:
            return data
        
        processed_data = data.copy()
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            skew_val = abs(stats.skew(data[col].dropna()))
            
            # Only preprocess if highly skewed
            if skew_val > skewness_threshold:
                if self.preprocessing == 'log':
                    processed_data[col] = np.log1p(data[col] - data[col].min() + 1)
                elif self.preprocessing == 'quantile':
                    transformer = QuantileTransformer(output_distribution='uniform', random_state=42)
                    processed_data[col] = transformer.fit_transform(data[[col]]).flatten()
                elif self.preprocessing == 'yeo-johnson':
                    transformer = PowerTransformer(method='yeo-johnson', standardize=False)
                    processed_data[col] = transformer.fit_transform(data[[col]]).flatten()
        
        return processed_data
    
    def _test_continuous_feature(self, in_cluster, out_cluster):
        """Perform KS test for continuous feature and calculate effect size."""
        try:
            statistic, p_value = ks_2samp(in_cluster, out_cluster)
            # Calculate Cohen's d as effect size
            cohens_d = self._calculate_cohens_d(in_cluster, out_cluster)
            return p_value, statistic, cohens_d
        except:
            return np.nan, np.nan, np.nan
    
    def _calculate_cohens_d(self, group1, group2):
        """
        Calculate Cohen's d effect size for continuous variables.
        
        Cohen's d measures the standardized difference between two group means:
        - Small effect: d ≈ 0.2
        - Medium effect: d ≈ 0.5  
        - Large effect: d ≈ 0.8+
        
        Returns:
        --------
        float : Cohen's d effect size (absolute value)
        """
        try:
            n1, n2 = len(group1), len(group2)
            mean1, mean2 = np.mean(group1), np.mean(group2)
            var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
            
            # Pooled standard deviation
            pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
            
            # Cohen's d
            d = abs(mean1 - mean2) / pooled_std
            return d
        except:
            return np.nan
    
    def _test_categorical_feature(self, in_cluster, out_cluster):
        """Perform chi-square or Fisher's exact test for categorical feature and calculate effect size."""
        try:
            # Create contingency table
            in_counts = in_cluster.value_counts()
            out_counts = out_cluster.value_counts()
            
            # Align indices
            all_categories = set(in_counts.index) | set(out_counts.index)
            contingency = []
            for cat in all_categories:
                contingency.append([
                    in_counts.get(cat, 0),
                    out_counts.get(cat, 0)
                ])
            
            contingency = np.array(contingency).T
            
            # Calculate Cramér's V as effect size
            cramers_v = self._calculate_cramers_v(contingency)
            
            # Choose test based on expected frequencies
            expected = stats.contingency.expected_freq(contingency)
            if np.any(expected < 5) and contingency.shape == (2, 2):
                # Use Fisher's exact for 2x2 tables with low expected frequencies
                _, p_value = fisher_exact(contingency)
                statistic = np.nan
            else:
                # Use chi-square test
                statistic, p_value, _, _ = chi2_contingency(contingency)
            
            return p_value, statistic, cramers_v
        except:
            return np.nan, np.nan, np.nan
    
    def _calculate_cramers_v(self, contingency_table):
        """
        Calculate Cramér's V effect size for categorical variables.
        
        Cramér's V measures the strength of association between categorical variables:
        - Small effect: V ≈ 0.1
        - Medium effect: V ≈ 0.3
        - Large effect: V ≈ 0.5+
        
        Returns:
        --------
        float : Cramér's V effect size (0 to 1)
        """
        try:
            chi2_stat, _, _, _ = chi2_contingency(contingency_table)
            n = np.sum(contingency_table)
            min_dim = min(contingency_table.shape) - 1
            
            if min_dim == 0:
                return 0
            
            cramers_v = np.sqrt(chi2_stat / (n * min_dim))
            return cramers_v
        except:
            return np.nan
    
    def profile_clusters(self, data, cluster_labels, continuous_cols=None, categorical_cols=None):
        """
        Profile clusters by testing each feature for cluster vs rest differences.
        
        Parameters:
        -----------
        data : pd.DataFrame
            Input data with features
        cluster_labels : array-like
            Cluster assignments for each sample
        continuous_cols : list or None
            List of continuous feature columns. If None, infers from numeric dtypes.
        categorical_cols : list or None  
            List of categorical feature columns. If None, infers from object dtypes.
            
        Returns:
        --------
        pd.DataFrame : Results with p-values and statistics for each cluster-feature pair
        """
        data = data.copy()
        cluster_labels = np.array(cluster_labels)
        
        # Infer column types if not specified
        if continuous_cols is None:
            continuous_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        if categorical_cols is None:
            categorical_cols = data.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # Apply preprocessing to continuous features
        if continuous_cols and self.preprocessing:
            data = self._preprocess_continuous(data)
        
        results = []
        unique_clusters = np.unique(cluster_labels)
        
        for cluster_id in unique_clusters:
            cluster_mask = cluster_labels == cluster_id
            
            # Test continuous features
            for col in continuous_cols:
                in_cluster = data.loc[cluster_mask, col].dropna()
                out_cluster = data.loc[~cluster_mask, col].dropna()
                
                if len(in_cluster) > 0 and len(out_cluster) > 0:
                    p_value, statistic, effect_size = self._test_continuous_feature(in_cluster, out_cluster)
                    
                    results.append({
                        'cluster': cluster_id,
                        'feature': col,
                        'feature_type': 'continuous',
                        'test': 'kolmogorov_smirnov',
                        'p_value': p_value,
                        'statistic': statistic,
                        'effect_size': effect_size,
                        'effect_size_type': 'cohens_d',
                        'significant': p_value < self.alpha if not np.isnan(p_value) else False,
                        'n_in_cluster': len(in_cluster),
                        'n_out_cluster': len(out_cluster)
                    })
            
            # Test categorical features
            for col in categorical_cols:
                in_cluster = data.loc[cluster_mask, col].dropna()
                out_cluster = data.loc[~cluster_mask, col].dropna()
                
                if len(in_cluster) > 0 and len(out_cluster) > 0:
                    p_value, statistic, effect_size = self._test_categorical_feature(in_cluster, out_cluster)
                    
                    results.append({
                        'cluster': cluster_id,
                        'feature': col,
                        'feature_type': 'categorical',
                        'test': 'chi_square_or_fisher',
                        'p_value': p_value,
                        'statistic': statistic,
                        'effect_size': effect_size,
                        'effect_size_type': 'cramers_v',
                        'significant': p_value < self.alpha if not np.isnan(p_value) else False,
                        'n_in_cluster': len(in_cluster),
                        'n_out_cluster': len(out_cluster)
                    })
        
        self.results_ = pd.DataFrame(results)
        return self.results_
    
    def get_cluster_characteristics(self, cluster_id, top_n=10):
        """
        Get top characteristic features for a specific cluster.
        
        Parameters:
        -----------
        cluster_id : int/str
            Cluster identifier
        top_n : int, default=10
            Number of top features to return
            
        Returns:
        --------
        pd.DataFrame : Top characteristic features for the cluster
        """
        if self.results_ is None:
            raise ValueError("Must run profile_clusters() first")
        
        cluster_results = self.results_[self.results_['cluster'] == cluster_id]
        return cluster_results.nsmallest(top_n, 'p_value')
    
    def summary(self):
        """Print summary of profiling results."""
        if self.results_ is None:
            raise ValueError("Must run profile_clusters() first")
        
        print(f"Cluster Profiling Summary")
        print(f"========================")
        print(f"Total tests performed: {len(self.results_)}")
        print(f"Significant results (α={self.alpha}): {self.results_['significant'].sum()}")
        print(f"Clusters analyzed: {self.results_['cluster'].nunique()}")
        print(f"Features analyzed: {self.results_['feature'].nunique()}")
        print(f"Preprocessing applied: {self.preprocessing or 'None'}")
        
        if self.skewness_report_ is not None:
            highly_skewed = (self.skewness_report_['abs_skewness'] > 1).sum()
            print(f"Highly skewed features: {highly_skewed}")
        
        # Summary of corrections and effect sizes
        if 'p_value_corrected' in self.results_.columns:
            corrected_sig = self.results_['significant_corrected'].sum()
            print(f"Significant after FDR correction: {corrected_sig}")
            
            # Effect size summary
            large_effects = (self.results_['effect_size'] > 0.8).sum()
            medium_effects = ((self.results_['effect_size'] > 0.5) & 
                            (self.results_['effect_size'] <= 0.8)).sum()
            print(f"Large effects (>0.8): {large_effects}")
            print(f"Medium effects (0.5-0.8): {medium_effects}")
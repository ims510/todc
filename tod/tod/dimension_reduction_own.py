from .corpus import Corpus
from abc import ABC
import pandas as pd
import numpy as np
from typing import List
import plotly.express as px

class DimensionReductionOwn(ABC):
    indices_to_remove: List[int]
    indices_to_keep: List[int]

class HighCorrelation(DimensionReductionOwn):
    """
    High Correlation feature removal.
    This class removes features that are highly correlated with each other.
    """
    def __init__(self, corpus: Corpus, threshold: float = 0.9):
        if not (0 <= threshold <= 1):
            raise ValueError("Threshold must be between 0 and 1.")
        self.corpus = corpus
        self.threshold = threshold
        self.indices_to_remove = []
        self.indices_to_keep= []
        self.correlated_pairs = []
        self.x_cleaned = None
        self._remove_high_correlation_features()


    def _remove_high_correlation_features(self):
        """
        Remove features that are highly correlated with each other.
        """
        X = self.corpus.feature_matrix
        X_df = pd.DataFrame(X, columns=[self.corpus.idx2feature(idx) for idx in range(X.shape[1])])
        correlation_matrix = X_df.corr().abs() # Absolute correlation values
        features_to_remove_corr = set()

        highly_correlated = np.where(correlation_matrix > self.threshold)
        correlated_pairs = [
            (correlation_matrix.index[i], correlation_matrix.columns[j], correlation_matrix.iloc[i, j])
            for i, j in zip(*highly_correlated)
            if i != j and i < j  # Avoid self-correlation and duplicate pairs
        ]
        for feature1, feature2, _ in correlated_pairs:
            # Decide which feature to remove (keep the one with higher variance)
            if X_df[feature1].var() > X_df[feature2].var():
                features_to_remove_corr.add(feature2)
            else:
                features_to_remove_corr.add(feature1)

        self.correlated_pairs = correlated_pairs

        self.indices_to_remove = [self.corpus.feature2idx(feature) for feature in features_to_remove_corr]
        self.indices_to_keep = [i for i in range(X.shape[1]) if i not in self.indices_to_remove]
        X_cleaned = np.delete(X, list(self.indices_to_remove), axis=1)
        self.x_cleaned = X_cleaned

        
    def show_result(self):
        print(f"Number of features removed: {len(self.indices_to_remove)}")
        print(f"Number of features kept: {len(self.indices_to_keep)}")
        print("Highly correlated pairs and their correlation values:")
        for feature1, feature2, correlation in self.correlated_pairs:
            print(f"   {feature1} and {feature2}: {correlation:.2f}")
        print("#" + "-" * 50)
        print("Removed features:")
        for index in self.indices_to_remove:
            print(f"   {self.corpus.idx2feature(index)}")
        print("#" + "-" * 50)
        print("Kept features:")
        for index in self.indices_to_keep:
            print(f"   {self.corpus.idx2feature(index)}")

class Frequency(DimensionReductionOwn):
    def __init__(self, corpus: Corpus, dim_red_corr: DimensionReductionOwn, lower_bound: int = 10, upper_bound: int = 5000):
        self.corpus = corpus
        self.dim_red_corr = dim_red_corr
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.indices_to_remove = []
        self.indices_to_keep = []
        self.frequency_df = self._compute_frequency_df()
        self._remove_frequency_features()
    
    def _compute_frequency_df(self):
        frequency_matrix = self.corpus.co_occurrence_matrix
        column_sums = np.sum(frequency_matrix, axis=0)
        return pd.DataFrame({
            'Feature': [self.corpus.idx2feature(i) for i in self.dim_red_corr.indices_to_keep],
            'Frequency': column_sums[self.dim_red_corr.indices_to_keep]
        })
    
    def show_data(self):
        """
        Display the frequency DataFrame.
        """
        return self.frequency_df.sort_values(by='Frequency', ascending=False)
    
    def show_graph(self):
        # Add an index column to the DataFrame
        self.frequency_df_sorted = self.frequency_df.sort_values(by="Frequency", ascending=True).reset_index(drop=True)
        self.frequency_df_sorted["Index"] = self.frequency_df_sorted.index

        # Create the bar plot
        fig = px.bar(
            self.frequency_df_sorted,
            x="Index",  # Use the index as the x-axis
            y="Frequency",  # Use the frequency as the y-axis
            hover_data={"Feature": True, "Frequency": True},  # Show feature and frequency on hover
            labels={"Index": "Feature Index", "Frequency": "Frequency"},  # Axis labels
            title="Feature Frequencies Ordered by Value"
        )

        fig.add_shape(
            type="line",
            x0=0, x1=len(self.frequency_df_sorted) - 1,  # Span the entire x-axis
            y0=self.lower_bound, y1=self.lower_bound,  # Horizontal line at self.lower_bound
            line=dict(color="red", width=2, dash="dash"),  # Red dashed line
            name="Lower Bound"
        )

        fig.add_shape(
            type="line",
            x0=0, x1=len(self.frequency_df_sorted) - 1,  # Span the entire x-axis
            y0=self.upper_bound, y1=self.upper_bound,  # Horizontal line at upper_bound
            line=dict(color="red", width=2, dash="dash"),  # Red dashed line
            name="Upper Bound"
        )

        # Update layout for better readability
        fig.update_layout(
            xaxis=dict(title="Feature Index"),
            yaxis=dict(title="Frequency"),
            showlegend=False
        )

        # Show the plot
        fig.show()

    def _remove_frequency_features(self):
        split_frequency_df = self.frequency_df.copy()
        # Split the 'Feature' column into 'Feature' and 'Value'
        split_frequency_df[['Feature', 'Value']] = self.frequency_df['Feature'].str.rsplit("=", n=1, expand=True)

        # Find features that have exactly two unique values
        features_with_two_values = split_frequency_df.groupby('Feature').filter(lambda x: x['Value'].nunique() == 2)


        # For each pair, keep only the row with the higher frequency
        features_with_max_frequency = features_with_two_values.loc[
            features_with_two_values.groupby('Feature')['Frequency'].idxmax()
        ]

        # Identify the rows that were removed (not the maximum frequency)
        removed_features = features_with_two_values.loc[
            ~features_with_two_values.index.isin(features_with_max_frequency.index)
        ]

        # Create a list of the full features (e.g., "node:X:prev:upos=X") for the kept rows
        kept_features_list = features_with_max_frequency['Feature'] + "=" + features_with_max_frequency['Value']

        # Create a list of the full features for the removed rows
        removed_features_list = removed_features['Feature'] + "=" + removed_features['Value']

        # Convert to Python lists
        kept_features_list = kept_features_list.tolist()
        removed_features_list = removed_features_list.tolist()

        features_to_keep_freq_df = self.frequency_df[(self.frequency_df['Frequency'] > self.lower_bound) & (self.frequency_df['Frequency'] < self.upper_bound) & (np.isin(self.frequency_df['Feature'], test_elements=removed_features_list, invert=True))]
        features_to_remove_freq_df = self.frequency_df[(self.frequency_df['Frequency'] < self.lower_bound) | (self.frequency_df['Frequency'] > self.upper_bound)| (np.isin(self.frequency_df['Feature'], test_elements=removed_features_list))]


        self.indices_to_keep = [self.corpus.feature2idx(feature) for feature in features_to_keep_freq_df['Feature']]
        self.indices_to_remove = [self.corpus.feature2idx(feature) for feature in features_to_remove_freq_df['Feature']]
    
    def show_result(self):
        print(f"Number of features removed: {len(self.indices_to_remove)}")
        print(f"Number of features kept: {len(self.indices_to_keep)}")
        print("Frequency DataFrame:")
        print(self.show_data().to_markdown())
        print("#" + "-" * 50)
        print("Removed features:")
        for index in self.indices_to_remove:
            print(f"   {self.corpus.idx2feature(index)}")
        print("#" + "-" * 50)
        print("Kept features:")
        for index in self.indices_to_keep:
            print(f"   {self.corpus.idx2feature(index)}")

class Variance(DimensionReductionOwn):
    def __init__(self, corpus: Corpus, dim_red_corr: DimensionReductionOwn, feature_matrix, threshold: float = 0.25):
        print("Make sure you have chosen the correct matrix to extract the variance from. You can do that by calling the show_variance_graphs() function in the corpus class.")
        self.corpus = corpus
        self.dim_red_corr = dim_red_corr
        self.feature_matrix = feature_matrix
        self.threshold = threshold
        self.indices_to_remove = []
        self.indices_to_keep = []
        self.variance_df = self._compute_variance_df()
        self.lower_bound = self.variance_df['Variance'].quantile(self.threshold) # lower bound should be the bottom 25% of the variance values
        self._remove_low_variance_features()

    def _compute_variance_df(self):
        column_variances = np.var(self.feature_matrix, axis=0)
        var_df = pd.DataFrame({
            'Feature': [self.corpus.idx2feature(i) for i in self.dim_red_corr.indices_to_keep],
            'Variance': column_variances[self.dim_red_corr.indices_to_keep]
        })
        return var_df
    
    def show_data(self):
        """
        Display the variance DataFrame.
        """
        return self.variance_df.sort_values(by='Variance', ascending=False)
    
    def show_graph(self):
        var_df_sorted = self.variance_df.sort_values(by="Variance", ascending=True).reset_index(drop=True)
        var_df_sorted["Index"] = var_df_sorted.index

        # Create the bar plot
        fig = px.bar(
            var_df_sorted,
            x="Index",  # Use the index as the x-axis
            y="Variance",  # Use the frequency as the y-axis
            hover_data={"Feature": True, "Variance": True},  # Show feature and frequency on hover
            labels={"Index": "Feature Index", "Variance": "Variance"},  # Axis labels
            title="Feature Variances Ordered by Value"
        )

        fig.add_shape(
            type="line",
            x0=0, x1=len(var_df_sorted) - 1,  # Span the entire x-axis
            y0=self.lower_bound, y1=self.lower_bound,  # Horizontal line at lower_bound
            line=dict(color="red", width=2, dash="dash"),  # Red dashed line
            name="Lower Bound"
        )

        # Update layout for better readability
        fig.update_layout(
            xaxis=dict(title="Feature Index"),
            yaxis=dict(title="Variance"),
            showlegend=False
        )

        # Show the plot
        fig.show()

    def _remove_low_variance_features(self):
        features_to_keep_var_df = self.variance_df[(self.variance_df['Variance'] > self.lower_bound)] 
        features_to_remove_var_df = self.variance_df[(self.variance_df['Variance'] < self.lower_bound)]

        self.indices_to_keep = [self.corpus.feature2idx(feature) for feature in features_to_keep_var_df['Feature']]
        self.indices_to_remove = [self.corpus.feature2idx(feature) for feature in features_to_remove_var_df['Feature']]
    
    def show_result(self):
        print(f"Number of features removed: {len(self.indices_to_remove)}")
        print(f"Number of features kept: {len(self.indices_to_keep)}")
        print("Variance DataFrame:")
        print(self.show_data().to_markdown())
        print("#" + "-" * 50)
        print("Removed features:")
        for index in self.indices_to_remove:
            print(f"   {self.corpus.idx2feature(index)}")
        print("#" + "-" * 50)
        print("Kept features:")
        for index in self.indices_to_keep:
            print(f"   {self.corpus.idx2feature(index)}")

class Random(DimensionReductionOwn):
    def __init__(self, corpus: Corpus, dim_red_corr: DimensionReductionOwn, threshold: float = 0.1):
        self.corpus = corpus
        self.dim_red_corr = dim_red_corr
        self.threshold = threshold
        self.indices_to_remove = []
        self.indices_to_keep = []
        self.vmr_df = self._compute_vmr_df()
        self._remove_vmr_features()

    def _compute_vmr_df(self):
        means = np.mean(self.corpus.co_occurrence_matrix, axis=0)  # Mean for each column
        variances = np.var(self.corpus.co_occurrence_matrix, axis=0)  # Variance for each column
        vmr = variances / means
        vmr_df = pd.DataFrame({
            'Feature': [self.corpus.idx2feature(i) for i in self.dim_red_corr.indices_to_keep],
            'VMR': vmr[self.dim_red_corr.indices_to_keep]
        })
        return vmr_df
    
    def show_data(self):
        """
        Display the variance DataFrame.
        """
        return self.vmr_df.sort_values(by='VMR', ascending=False)
    
    def show_graph(self):
        vmr_df_sorted = self.vmr_df.sort_values(by="VMR", ascending=True).reset_index(drop=True)
        vmr_df_sorted["Index"] = vmr_df_sorted.index

        # Create the bar plot
        fig = px.bar(
            vmr_df_sorted,
            x="Index",  # Use the index as the x-axis
            y="VMR",  # Use the frequency as the y-axis
            hover_data={"Feature": True, "VMR": True},  # Show feature and frequency on hover
            labels={"Index": "Feature Index", "VMR": "VMR"},  # Axis labels
            title="VMR Ordered by Value"
        )

        lower_bound = self.threshold + 1

        fig.add_shape(
            type="line",
            x0=0, x1=len(vmr_df_sorted) - 1,  # Span the entire x-axis
            y0=lower_bound, y1=lower_bound,  # Horizontal line at lower_bound
            line=dict(color="red", width=2, dash="dash"),  # Red dashed line
            name="Lower Bound"
        )

        # Update layout for better readability
        fig.update_layout(
            xaxis=dict(title="Feature Index"),
            yaxis=dict(title="VMR"),
            showlegend=False
        )

        # Show the plot
        fig.show()

    def _remove_vmr_features(self):
        features_to_keep_vmr_df = self.vmr_df[np.abs(self.vmr_df["VMR"] - 1) >= self.threshold]
        features_to_remove_vmr_df = self.vmr_df[np.abs(self.vmr_df["VMR"] - 1) < self.threshold] 

        self.indices_to_keep = [self.corpus.feature2idx(feature) for feature in features_to_keep_vmr_df['Feature']]
        self.indices_to_remove = [self.corpus.feature2idx(feature) for feature in features_to_remove_vmr_df['Feature']]

    def show_result(self):
        print(f"Number of features removed: {len(self.indices_to_remove)}")
        print(f"Number of features kept: {len(self.indices_to_keep)}")
        print("VMR DataFrame:")
        print(self.show_data().to_markdown())
        print("#" + "-" * 50)
        print("Removed features:")
        for index in self.indices_to_remove:
            print(f"   {self.corpus.idx2feature(index)}")
        print("#" + "-" * 50)
        print("Kept features:")
        for index in self.indices_to_keep:
            print(f"   {self.corpus.idx2feature(index)}")

class DimensionReductionComplete(DimensionReductionOwn):
    def __init__(self, corpus, dim_red_correlation, dim_red_frequency, dim_red_variance, dim_red_random):
        self.corpus = corpus
        self.dim_red_correlation = dim_red_correlation
        self.dim_red_frequency = dim_red_frequency
        self.dim_red_variance = dim_red_variance
        self.dim_red_random = dim_red_random
        self.all_indices_to_remove = []
        self.all_indices_to_keep = []
        self.final_matrix = self._remove_features()

    def _remove_features(self):
        self.all_indices_to_remove = set(self.dim_red_correlation.indices_to_remove + self.dim_red_frequency.indices_to_remove + self.dim_red_variance.indices_to_remove + self.dim_red_random.indices_to_remove)
        self.all_indices_to_keep = set(self.dim_red_correlation.indices_to_keep + self.dim_red_frequency.indices_to_keep + self.dim_red_variance.indices_to_keep + self.dim_red_random.indices_to_keep)
        X = self.corpus.feature_matrix
        X_final = X[:, [i for i in range(X.shape[1]) if (i in self.all_indices_to_keep) and (i not in self.all_indices_to_remove)]]
        return X_final
    
    def show_result(self):
        print("Kept features:")
        for i in range(self.corpus.feature_matrix.shape[1]):
            if (i in self.all_indices_to_keep) and (i not in self.all_indices_to_remove):
                print(f"   {self.corpus.idx2feature(i)}")
        
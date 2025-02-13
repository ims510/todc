"""
This is the second script to be used within the project.
This script is used to cluster the data using the KMeans algorithm.
The necessary input is a csv obtained from the previous script.
"""

import pandas as pd
from time import time

from sklearn import metrics
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

def bench_k_means(kmeans, name, data, labels):
    """Benchmark to evaluate the KMeans initialization methods.

    Parameters
    ----------
    kmeans : KMeans instance
        A :class:`~sklearn.cluster.KMeans` instance with the initialization
        already set.
    name : str
        Name given to the strategy. It will be used to show the results in a
        table.
    data : ndarray of shape (n_samples, n_features)
        The data to cluster.
    labels : ndarray of shape (n_samples,)
        The labels used to compute the clustering metrics which requires some
        supervision.
    """
    t0 = time()
    estimator = make_pipeline(StandardScaler(), kmeans).fit(data)
    fit_time = time() - t0
    results = [name, fit_time, estimator[-1].inertia_]

    # Define the metrics which require only the true labels and estimator
    # labels
    clustering_metrics = [
        metrics.homogeneity_score,
        metrics.completeness_score,
        metrics.v_measure_score,
        metrics.adjusted_rand_score,
        metrics.adjusted_mutual_info_score,
    ]
    results += [m(labels, estimator[-1].labels_) for m in clustering_metrics]

    # The silhouette score requires the full dataset
    results += [
        metrics.silhouette_score(
            data,
            estimator[-1].labels_,
            metric="euclidean",
            sample_size=300,
        )
    ]

    # Show the results
    formatter_result = (
        "{:9s}\t{:.3f}s\t{:.0f}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}"
    )
    print(formatter_result.format(*results))

def get_n_bins(removed_deps_df):
    """
    Returns the number of bins to be used in the KMeans algorithm.
    """
    n_bins = removed_deps_df['deprel'].nunique()
    return n_bins

original_data_df = pd.read_csv('data/output/ro_rrt-ud-train.csv')
complete_data_df = pd.read_csv('data/output/ro_rrt-ud-train_complete_categ.csv')
removed_deps_df = pd.read_csv('data/output/ro_rrt-ud-train_nummod-amod-det_categ.csv')

# original_data_df = pd.read_csv('data/output/test.csv')
# complete_data_df = pd.read_csv('data/output/test_complete_categ.csv')
# removed_deps_df = pd.read_csv('data/output/test_nummod-amod-det_categ.csv')

labels = complete_data_df['deprel']
data = removed_deps_df.drop(columns=['deprel'])
# n_bins = get_n_bins(removed_deps_df)
# print(f"Number of bins: {n_bins}")
# print(f"data shape: {data.shape}")

# print(82 * "_")
# print("init\t\ttime\tinertia\thomo\tcompl\tv-meas\tARI\tAMI\tsilhouette")

# kmeanspp_kmeans = KMeans(init="k-means++", n_clusters=n_bins, n_init=4, random_state=0)
# bench_k_means(kmeans=kmeanspp_kmeans, name="k-means++", data=data, labels=labels)

# random_kmeans = KMeans(init="random", n_clusters=n_bins, n_init=4, random_state=0)
# bench_k_means(kmeans=random_kmeans, name="random", data=data, labels=labels)

# if data.shape[1] < n_bins:
#     print("\nNote: Number of features is less than number of clusters.")
#     print("PCA will be performed on available dimensions only.")
#     pca = PCA(n_components=data.shape[1]).fit(data)
#     # Use k-means++ initialization instead of PCA components
#     pca_kmeans = KMeans(n_clusters=n_bins, n_init=4, random_state=0)
#     # Transform data using PCA before clustering
#     transformed_data = pca.transform(data)
#     bench_k_means(kmeans=pca_kmeans, name="PCA+kmeans", data=transformed_data, labels=labels)
# else:
#     pca = PCA(n_components=n_bins).fit(data)
#     pca_kmeans = KMeans(init=pca.components_, n_clusters=n_bins, n_init=1)
#     bench_k_means(kmeans=pca_kmeans, name="PCA-based", data=data, labels=labels)

# print(82 * "_")


############################### RESULTS ANALYSIS #########################################################
# import pandas as pd
# import numpy as np
# from collections import defaultdict

# def analyze_deprel_overlaps(data, encoded_labels, original_labels, cluster_labels):
#     """
#     Analyze which deprel categories tend to appear in the same clusters.
    
#     Parameters:
#     data: original feature data
#     encoded_labels: encoded deprel labels used for clustering
#     original_labels: original deprel labels (string names)
#     cluster_labels: cluster assignments from KMeans
    
#     Returns:
#     DataFrame with overlap statistics
#     """
#     # Create mapping from encoded to original labels
#     label_mapping = dict(zip(encoded_labels.unique(), original_labels.unique()))
    
#     # Create a mapping of clusters to their deprels
#     cluster_to_deprels = defaultdict(lambda: defaultdict(int))
    
#     # Count deprels in each cluster
#     for cluster_label, deprel in zip(cluster_labels, encoded_labels):
#         original_deprel = label_mapping[deprel]
#         cluster_to_deprels[cluster_label][original_deprel] += 1
    
#     # Create overlap statistics
#     overlaps = []
    
#     # For each cluster
#     for cluster_id, deprel_counts in cluster_to_deprels.items():
#         # Get total points in cluster
#         total_in_cluster = sum(deprel_counts.values())
        
#         # Get top deprels in this cluster
#         sorted_deprels = sorted(deprel_counts.items(), key=lambda x: x[1], reverse=True)
        
#         # Record top overlapping deprels
#         for deprel, count in sorted_deprels[:5]:  # Show top 5 deprels per cluster
#             overlaps.append({
#                 'cluster_id': cluster_id,
#                 'deprel': deprel,
#                 'count': count,
#                 'percentage_in_cluster': (count / total_in_cluster) * 100, # percentage of cluster that is this deprel
#                 'percentage_of_deprel': (count / sum(original_labels == deprel)) * 100 # percentage of this deprel that is in this cluster
#             })
    
#     # Convert to DataFrame
#     overlap_df = pd.DataFrame(overlaps)
    
#     # Sort by count
#     overlap_df = overlap_df.sort_values(['cluster_id', 'count'], ascending=[True, False])
    
#     return overlap_df

# def find_most_confused_pairs(overlap_df):
#     """
#     Find pairs of deprels that most commonly appear together in clusters
#     """
#     confusion_pairs = defaultdict(int)
    
#     # For each cluster
#     for cluster_id in overlap_df['cluster_id'].unique():
#         cluster_deprels = overlap_df[overlap_df['cluster_id'] == cluster_id]['deprel'].tolist()
        
#         # Create pairs of deprels in this cluster
#         for i in range(len(cluster_deprels)):
#             for j in range(i + 1, len(cluster_deprels)):
#                 pair = tuple(sorted([cluster_deprels[i], cluster_deprels[j]]))
#                 confusion_pairs[pair] += 1
    
#     # Convert to DataFrame
#     pairs_df = pd.DataFrame([
#         {'deprel1': pair[0], 'deprel2': pair[1], 'overlap_count': count}
#         for pair, count in confusion_pairs.items()
#     ])
    
#     return pairs_df.sort_values('overlap_count', ascending=False)

# def print_cluster_analysis(overlap_df):
#     """
#     Print a readable analysis of cluster overlaps
#     """
#     print("\nDeprel Overlap Analysis:")
#     print("=" * 80)
    
#     for cluster_id in overlap_df['cluster_id'].unique():
#         cluster_data = overlap_df[overlap_df['cluster_id'] == cluster_id]
#         print(f"\nCluster {cluster_id}:")
#         print("-" * 40)
        
#         for _, row in cluster_data.iterrows():
#             print(f"Deprel: {row['deprel']}")
#             print(f"  Count: {row['count']}")
#             print(f"  {row['percentage_in_cluster']:.1f}% of cluster")
#             print(f"  {row['percentage_of_deprel']:.1f}% of this deprel type")
#             print()

# # Example usage
# def analyze_deprel_clustering(data, removed_deps_df, complete_data_df, kmeans_model):
#     """
#     Wrapper function to perform complete analysis
    
#     Parameters:
#     data: feature data used for clustering
#     removed_deps_df: DataFrame with encoded deprels
#     complete_data_df: DataFrame with original deprel names
#     kmeans_model: trained KMeans model
#     """
#     cluster_labels = kmeans_model.labels_
    
#     # Get overlap analysis with original labels
#     overlap_df = analyze_deprel_overlaps(
#         data, 
#         removed_deps_df['deprel'],
#         complete_data_df['deprel'],
#         cluster_labels
#     )
    
#     # Print detailed cluster analysis
#     print_cluster_analysis(overlap_df)
    
#     # Find most confused pairs
#     pairs_df = find_most_confused_pairs(overlap_df)
    
#     print("\nMost Frequently Overlapping Deprel Pairs:")
#     print("=" * 80)
#     print(pairs_df.head(10))
    
#     return overlap_df, pairs_df

# # data = removed_deps_df.drop(columns=['deprel'])
# # encoded_labels = complete_data_df['deprel']
# # original_labels = original_data_df['deprel']
# # cluster_labels = kmeanspp_kmeans.labels_

# # After your clustering is done:
# overlap_df, pairs_df = analyze_deprel_clustering(
#     removed_deps_df.drop(columns=['deprel']),
#     complete_data_df,
#     original_data_df,
#     kmeanspp_kmeans
# )

# # Save the results if needed
# overlap_df.to_csv('kmeanspp_deprel_overlap_analysis.csv', index=False)
# pairs_df.to_csv('kmeanspp_deprel_confusion_pairs.csv', index=False)

########################################################################################

import plotly.express as px
from umap import UMAP

umap_2d = UMAP(random_state=0)
umap_2d.fit(data)
projections = umap_2d.transform(data)
str_labels = []
for i in range(len(projections)):
    str_labels.append(original_data_df.iloc[i]['deprel'])
fig = px.scatter(
    projections, x=0, y=1,
    color=str_labels )
# fig = px.scatter(
#     projections, x=0, y=1, 
#     color='deprel',)
fig.show()

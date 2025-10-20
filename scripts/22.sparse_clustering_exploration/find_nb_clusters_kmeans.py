from scipy.cluster.hierarchy import fcluster
from sklearn.metrics import davies_bouldin_score
import numpy as np
from sklearn.cluster import KMeans
from sklearn.utils import resample
from sklearn.metrics import jaccard_score
from pyclustertend import hopkins
from pyclustertend import vat

import sys
sys.path.insert(1, "/Users/madalina/Documents/M2TAL/stage/check_coherent_labels/tod")

import tod.corpus
import tod.outliers
import tod.clustering
import tod.plotting
import tod.dimension_reduction_classic

def delta(ck, cl):
    values = np.ones([len(ck), len(cl)])*10000
    
    for i in range(0, len(ck)):
        for j in range(0, len(cl)):
            values[i, j] = np.linalg.norm(ck[i]-cl[j])
            
    return np.min(values)
    
def big_delta(ci):
    values = np.zeros([len(ci), len(ci)])
    
    for i in range(0, len(ci)):
        for j in range(0, len(ci)):
            values[i, j] = np.linalg.norm(ci[i]-ci[j])
            
    return np.max(values)
    
def dunn(k_list):
    """ Dunn index [CVI]
    
    Parameters
    ----------
    k_list : list of np.arrays
        A list containing a numpy array for each cluster |c| = number of clusters
        c[K] is np.array([N, p]) (N : number of samples in cluster K, p : sample dimension)
    """
    deltas = np.ones([len(k_list), len(k_list)])*1000000
    big_deltas = np.zeros([len(k_list), 1])
    l_range = list(range(0, len(k_list)))
    
    for k in l_range:
        for l in (l_range[0:k]+l_range[k+1:]):
            deltas[k, l] = delta(k_list[k], k_list[l])
        
        big_deltas[k] = big_delta(k_list[k])

    di = np.min(deltas)/np.max(big_deltas)
    return di

def bootstrap_stability_kmeans(X, n_bootstrap=100, n_clusters=3, random_state=None):
    """
    Perform stability assessment for k-means clustering using bootstrapping.
    
    Parameters:
        X (numpy.ndarray): Original data matrix (m rows, n columns).
        n_bootstrap (int): Number of bootstrap samples to generate.
        n_clusters (int): Number of clusters to form.
        random_state (int or None): Random state for reproducibility.
    
    Returns:
        float: Mean Jaccard index as a measure of clustering stability.
    """
    # Step 1: Perform k-means clustering on the original data
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
    C0 = kmeans.fit_predict(X)
    
    jaccard_indices = []
    
    # Step 2: Generate bootstrap samples and cluster them
    for _ in range(n_bootstrap):
        # Generate a bootstrap sample
        X_bootstrap = resample(X, replace=True, n_samples=X.shape[0], random_state=random_state)
        
        # Perform k-means clustering on the bootstrap sample
        kmeans_bootstrap = KMeans(n_clusters=n_clusters, random_state=random_state)
        Ci = kmeans_bootstrap.fit_predict(X_bootstrap)
        
        # Step 3: Compute Jaccard index between C0 and Ci
        # Align clusters by comparing each cluster in C0 with the most similar cluster in Ci
        jaccard_sum = 0
        for cluster_id in np.unique(C0):
            A = (C0 == cluster_id).astype(int)
            max_jaccard = 0
            for cluster_id_bootstrap in np.unique(Ci):
                B = (Ci == cluster_id_bootstrap).astype(int)
                max_jaccard = max(max_jaccard, jaccard_score(A, B))
            jaccard_sum += max_jaccard
        
        # Average Jaccard index for this bootstrap sample
        jaccard_indices.append(jaccard_sum / n_clusters)
    
    # Step 4: Compute the mean Jaccard index across all bootstrap samples
    mean_jaccard_index = np.mean(jaccard_indices)
    return mean_jaccard_index

def plot_score_values(score_values, score_name: str):
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 6))
    plt.plot(range(2, len(score_values) + 2), score_values, marker='o')
    plt.title(f'{score_name} vs Number of Clusters')
    plt.xlabel('Number of Clusters')
    plt.ylabel(score_name)
    plt.xticks(range(2, len(score_values) + 2))
    plt.grid()
    plt.savefig(f'{score_name.replace(" ", "_").lower()}_vs_clusters.png')

def find_optimal_nb_clusters(corpus: tod.corpus.Corpus, max_clusters: int = 50):
    kmeans_dunn_scores = []
    kmeans_db_scores = []
    kmeans_jaccard_scores = []
    for i in range(2, max_clusters + 1):
        # print(f"Evaluating k={i}")
        kmeans = tod.clustering.KMeans(corpus=corpus, k=i, n_init=20)
        clusters_with_data = [corpus.feature_matrix[kmeans.clusters[j]] for j in range(i)]
        dunn_score = dunn(clusters_with_data)
        db_score = davies_bouldin_score(corpus.feature_matrix, kmeans.labels)
        kmeans_dunn_scores.append(dunn_score)
        # print(f"Dunn index for k={i}: {dunn_score}")
        kmeans_db_scores.append(db_score)
        # print(f"Davies-Bouldin score for k={i}: {db_score}")
        mean_jaccard = bootstrap_stability_kmeans(corpus.feature_matrix, n_bootstrap=100, n_clusters=i, random_state=42)
        kmeans_jaccard_scores.append(mean_jaccard)
    
    print("Cluster evaluation metrics:")
    print("---------------------------")

    max_dunn_value = max(kmeans_dunn_scores)
    dunn_optimal_k = kmeans_dunn_scores.index(max_dunn_value) + 2  # +2 because range starts from 2
    print("== Dunn Index ==")
    print("Ratio of the minimum inter-cluster distance to the maximum intra-cluster diameter. Measures how well separated and tight the clusters are. Higher values = better separation and cohesion")
    plot_score_values(kmeans_dunn_scores, "Dunn Index")
    print(f"Optimal number of clusters according to Dunn index: {dunn_optimal_k} with score {max_dunn_value}")
    
    min_db_value = min(kmeans_db_scores)
    db_optimal_k = kmeans_db_scores.index(min_db_value) + 2
    print("== Davies-Bouldin Score ==")
    print("Average similarity ratio of each cluster with its most similar cluster. Lower values = better clustering, more compact and distinct clusters.")
    plot_score_values(kmeans_db_scores, "Davies-Bouldin Score")
    print(f"Optimal number of clusters according to Davies-Bouldin score: {db_optimal_k} with score {min_db_value}")

    max_jaccard_value = max(kmeans_jaccard_scores)
    jaccard_optimal_k = kmeans_jaccard_scores.index(max_jaccard_value) + 2
    print("== Jaccard Stability ==")
    print("Measures the stability of clustering results across bootstrap samples. 0 to 1 scale, higher values = more stable clusters.")
    plot_score_values(kmeans_jaccard_scores, "Jaccard Stability")
    print(f"Optimal number of clusters according to Jaccard stability: {jaccard_optimal_k} with score {max_jaccard_value}")

treebank_path = "/Users/madalina/Documents/M2TAL/stage/check_coherent_labels/data/input/Universal_Dependencies/ud-treebanks-v2.15/UD_French-Sequoia"
grew_pattern = "pattern{X[upos<>PUNCT]}"
patterns_text_file = "/Users/madalina/Documents/M2TAL/stage/check_coherent_labels/scripts/3. probability_matrix/patterns_all_nodes.txt"
analysed_category = "all_nodes"

corpus = tod.corpus.Corpus(
    treebank_path=treebank_path,
    grew_pattern=grew_pattern,
    patterns_text_file=patterns_text_file,
    matrix_type="coverage"
)

# for testing with adverbs (it's much quicker)
# corpus = tod.corpus.Corpus(
#     treebank_path="/Users/madalina/Documents/M1TAL/stage-SK/Treebanks/UD_French-GSD-master",
#     grew_pattern="pattern {X[upos=ADV]} without {X[InIdiom=Yes];X[Idiom=Yes]}",
#     patterns_text_file="/Users/madalina/Documents/M2TAL/stage/check_coherent_labels/scripts/3. probability_matrix/patterns_adv.txt",
#     matrix_type="coverage"
# )
print("Made corpus")
print("Feature matrix shape:", corpus.feature_matrix.shape)
print("="*50)
print("Checking clustering tendency...")
hopkins_score = hopkins(corpus.feature_matrix, corpus.feature_matrix.shape[0])
print("== Hopkins statistic ==")
print("Measures the clustering tendency of a dataset. Values range from 0 to 0.5 with values close to 0.5 indicating random data and values close to 0 indicating highly clusterable data.")
print(f"Hopkins statistic: {hopkins_score}")

print("== Graphical assessment of clustering tendency (VAT) ==")
print("The VAT algorithm reorders the dissimilarity matrix to visually reveal cluster tendency. Dark blocks along the diagonal indicate potential clusters.")
# vat(corpus.feature_matrix)

print("="*50)
find_optimal_nb_clusters(corpus=corpus, max_clusters=20)

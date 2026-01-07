import grewpy
from grewpy import Corpus, CorpusDraft, Request
import sys
sys.path.insert(1, "/Users/madalina/Documents/M2TAL/stage/check_coherent_labels/tod")

import tod.corpus
import tod.clustering
import tod.plotting
import tod.dimension_reduction_classic
import numpy as np
import skfuzzy as fuzz
import pandas as pd

# -------------------------------------------------------------------
# LOAD CORPUS AND FEATURES
# -------------------------------------------------------------------

treebank_path = "/Users/madalina/Documents/M2TAL/stage/check_coherent_labels/data/input/Universal_Dependencies/ud-treebanks-v2.15/UD_Marathi-UFAL"
grew_pattern = "pattern{X[upos<>PUNCT]}"
patterns_text_file = "/Users/madalina/Documents/M2TAL/stage/check_coherent_labels/scripts/3. probability_matrix/patterns_all_nodes.txt"
analysed_category = "all_nodes"

treebank_name = treebank_path.split("/")[-1]

corpus = tod.corpus.Corpus(
    treebank_path=treebank_path,
    grew_pattern=grew_pattern,
    patterns_text_file=patterns_text_file,
    use_sud=False,
    matrix_type="coverage",
    excluded_feature_patterns=[r"Translit=", r"Ltranslit="]
)

X = np.asarray(corpus.feature_matrix, dtype=float)
n_samples, n_features = X.shape

# -------------------------------------------------------------------
# SPARSE K-MEANS (hard clustering)
# -------------------------------------------------------------------

# This is used to extract centroids and defining features for clusters.
# K should be determined using dunn and db index (use script scripts/22.sparse_clustering_exploration/find_nb_clusters_kmeans.py)

k = 10

clustering = tod.clustering.SparseKMeans(
    corpus=corpus, 
    k=k, 
    top_n_features=10, # change if you want more than 10 defining features per cluster
    cluster_defining_features=True, # extract the defining features per cluster
    write_to_file=True) # save them as output.txt - if this is false they'll just get printed to the terminal

centroids = clustering.get_centroids(corpus)
centroids = np.asarray(centroids, dtype=float)

# -------------------------------------------------------------------
# # TSNE + VISUALISATION (HTML graph)
# -------------------------------------------------------------------

dim_red = tod.dimension_reduction_classic.Tsne_corpus(corpus, n_components=2)
fig = tod.plotting.cluster_scatter_plot(corpus, dim_red, clustering)
fig.write_html(f"{treebank_name}_sparse_kmeans.html")


# -------------------------------------------------------------------
# FUZZY C-MEANS CLUSTERING ON SPARSE K-MEANS CENTROIDS
# -------------------------------------------------------------------

m = 1.5  # fuzziness exponent

u, _, _, _, _, fpc = fuzz.cluster.cmeans_predict(
    X.T,
    centroids,
    m=m,
    error=1e-5,
    maxiter=1000,
    seed=42
)

membership = u.T                      # shape = (samples × clusters)
hard_labels = membership.argmax(axis=1)

print("Fuzzy partition coefficient ranges from 0 to 1, with values closer to 1 indicating better clustering. It tells us how well the data is described by the clusters.")
print(f"Fuzzy partition coefficient (FPC): {fpc}")

# -------------------------------------------------------------------
# MULTI-CLUSTER ITEMS + DISPERSION METRICS
# -------------------------------------------------------------------

threshold = 0.1 # membership threshold for multi-cluster assignment (so will only consider clusters with membership >= threshold)

def cluster_fuzziness(probs):
    """Compute entropy, 1-max and top2 ratio for fuzzy memberships."""
    probs = np.asarray(probs)
    probs = probs[probs > 0]
    K = len(probs)

    H = - np.sum(probs * np.log(probs)) / np.log(K)   # normalised entropy
    u_max = np.max(probs)
    sorted_probs = np.sort(probs)[::-1]
    ratio = sorted_probs[0] / sorted_probs[1] if len(sorted_probs) > 1 else np.inf

    return H, (1 - u_max), ratio

rows = []
for i in range(n_samples):
    above = [(c, membership[i, c]) for c in range(k) if membership[i, c] >= threshold]
    if len(above) >= 2:
        H, one_minus_max, ratio = cluster_fuzziness([p for _, p in above])
        rows.append({
            "row_idx": i,
            "lexunit": corpus.idx2lexunit(i),
            "primary_cluster": int(hard_labels[i]),
            "clusters_above_threshold": above,
            "entropy": H,
            "one_minus_max": one_minus_max,
            "top2_ratio": ratio
        })

df_multi = pd.DataFrame(rows).sort_values("row_idx")

# -------------------------------------------------------------------
# SAVE SUBMATRIX OF IN-BETWEEN / MULTI-CLUSTER ITEMS
# -------------------------------------------------------------------
# Get indices of items with strictly more than 2 clusters above threshold (gt2 = greater than 2)

indices_gt2 = [
    row["row_idx"]
    for row in rows
    if len(row["clusters_above_threshold"]) > 2
]


if len(indices_gt2) == 0:
    print("No elements found with >2 clusters above threshold.")
else:
    X_sub = X[indices_gt2, :]      # extract submatrix
    np.save(f"{treebank_name}_features_gt2clusters.npy", X_sub)
    np.save(f"{treebank_name}_feature_names.npy", np.array(list(corpus._idx2feature.values())))
    co_occ_matrix = corpus.co_occurrence_matrix[indices_gt2, :]
    np.save(f"{treebank_name}_cooccurrence_gt2clusters.npy", co_occ_matrix)


    # Optional: also save the corresponding lexical units for reference
    lexunits_sub = [corpus.idx2lexunit(i) for i in indices_gt2]
    pd.DataFrame({
        "row_idx": indices_gt2,
        "lexunit": lexunits_sub
    }).to_csv(f"{treebank_name}_lexunits_gt2clusters.csv", index=False)

    print(f"Saved submatrix of {len(indices_gt2)} items (>2 clusters above threshold) as {treebank_name}_features_gt2clusters.npy")
    print(f"Saved corresponding lexical units to {treebank_name}_lexunits_gt2clusters.csv")

# -------------------------------------------------------------------
# SAVE FUZZY METRICS CSV
# -------------------------------------------------------------------
df_multi.to_csv(f"{treebank_name}_fuzzy_entropy_oneminmax_top2ratio.csv", index=False)

print(f"Saved {treebank_name}_sparse_kmeans.html")
print("Saved output.txt with defining features for each cluster as extracted from sparse k-means")
print(f"Saved {treebank_name}_fuzzy_entropy_oneminmax_top2ratio.csv")
print(f"Fuzzy partition coefficient (FPC): {fpc:.4f}")


# -------------------------------------------------------------------
# INTERACTIVE FEATURE INSPECTION TOOL
# -------------------------------------------------------------------

# After running the script, you inspect the csv and if any words are surprising or worth investigating,
# you can use this tool to see which features are pulling them toward each cluster.


"""
Explanation of 'diff' and 'contrib' used in feature explanations:

For a given lexical unit x and a cluster centroid c:

    diff = x - c
    contrib = (diff**2) / sum(diff**2 over all features)

Meaning of diff[j]:
    - Signed difference between lexical unit and centroid on feature j.
    - diff > 0  → lexical unit has more of this feature than the cluster expects
    - diff < 0  → lexical unit has less of this feature than the cluster expects
    - diff = 0  → lexical unit matches the centroid perfectly on this feature

Meaning of contrib[j]:
    - Fraction of the total squared distance from x to the centroid that comes
      from feature j.
    - contrib is always between 0 and 1 and sums to 1 across all features.
    - High contrib → feature is a major source of mismatch (opposing feature)
    - Low contrib → feature aligns well with centroid (supportive feature)

In the explanations:
    - Top supportive features: features with the smallest contrib values.
    - Top opposing features: features with the largest contrib values.

"""

def interactive_feature_inspect(threshold=0.1, top_n=5):
    """
    Interactive tool: for a given lexical unit, shows which clusters it belongs to
    (membership above threshold) and the top features pulling it toward each cluster.
    """
    print("\nInteractive lexical unit cluster inspector")
    print("Type a lexical unit exactly as in the CSV, e.g.: ('moins','PRON')")
    print("Type 'quit' to stop.\n")

    # Map lexunit string -> row index
    lex2idx = {str(corpus.idx2lexunit(i)): i for i in range(X.shape[0])}

    while True:
        query = input("Lexunit> ").strip()
        if query.lower() in {"quit", "exit"}:
            print("Done.")
            break

        if query not in lex2idx:
            print("Lexical unit not found. Make sure the format matches exactly.\n")
            continue

        idx = lex2idx[query]
        x = X[idx]
        row_membership = membership[idx]

        # clusters above threshold
        clusters_above = [(c, row_membership[c]) for c in range(membership.shape[1])
                          if row_membership[c] >= threshold]

        if not clusters_above:
            print(f"No clusters above threshold {threshold} for {query}.\n")
            continue

        clusters_above.sort(key=lambda x: x[1], reverse=True)

        print(f"\nLexical unit: {query}")
        print(f"Row index in feature matrix: {idx}")
        print(f"Clusters above threshold {threshold} (sorted): {[c for c,_ in clusters_above]}\n")

        for c, mem in clusters_above:
            centroid = centroids[c]
            diff = x - centroid # feature vector of the lexunit - cluster center for cluster c => we get how far each feature is from the cluster centre
            sq_diff = diff ** 2
            total_sq = sq_diff.sum()

            if total_sq == 0:
                print(f"Cluster {c} (membership {mem:.3f}): lexical unit exactly at centroid.")
                continue

            contrib = sq_diff / total_sq

            # Remove meaningless small differences or small contributions
            min_diff = 1e-4
            min_contrib = 1e-4

            informative_idx = np.where(
                (np.abs(diff) > min_diff) & (contrib > min_contrib)
            )[0]

            if informative_idx.size == 0:
                print(f"Cluster {c} (membership {mem:.3f}): no informative features.\n")
                continue

            # Sort informative features by contribution
            sorted_inf = informative_idx[np.argsort(contrib[informative_idx])]

            # Supportive = closest to centroid
            supportive_idx = sorted_inf[:top_n]

            # Opposing = farthest from centroid
            opposing_idx = sorted_inf[-top_n:][::-1]

            print(f"===== Cluster {c} (membership={mem:.3f}) =====")
            print("Top supportive features (pulling lexical unit toward cluster):")
            if len(supportive_idx) == 0:
                print("  (no supportive features above zero)")
            else:
                for i in supportive_idx:
                    print(f"  - {corpus.idx2feature(i):50s} contrib={contrib[i]:.4f} diff={diff[i]:.4f}")

            print("Top opposing features (pulling away from cluster):")
            if len(opposing_idx) == 0:
                print("  (no opposing features above zero)")
            else:
                for i in opposing_idx:
                    print(f"  - {corpus.idx2feature(i):50s} contrib={contrib[i]:.4f} diff={diff[i]:.4f}")
            print("\n" + "-"*60 + "\n")

# Run the interactive tool (can be commented out if not needed)
interactive_feature_inspect(threshold=0.1, top_n=5) # adjust threshold and number of features as needed
import numpy as np
from sklearn.neighbors import NearestNeighbors

def gaussian_kernel(x, sigma):
    """Compute Gaussian kernel value."""
    D = x.shape[1]  # Number of features
    coeff = (2 * np.pi * sigma**2) ** (-D / 2)
    return coeff * np.exp(-np.linalg.norm(x, axis=1)**2 / (2 * sigma**2))

def estimate_density(X, outlier_indices, s=10, sigma=1.0):
    """Estimate local density using Parzen window for given outliers."""
    nn = NearestNeighbors(n_neighbors=s).fit(X)
    densities = {}
    
    for idx in outlier_indices:
        neighbors = nn.kneighbors([X[idx]], return_distance=False)[0]  # Get nearest neighbors
        neighbor_points = X[neighbors]
        density = np.mean(gaussian_kernel(neighbor_points - X[idx], sigma))
        densities[idx] = density
    
    return densities


def compute_optimal_direction(A, B):
    """Compute the optimal direction w using SVD and eigen decomposition."""
    U, S, Vt = np.linalg.svd(A, full_matrices=False)
    S_inv = np.diag(1 / S[:len(S)])  # Invert nonzero singular values
    A_pseudo_inv = U @ S_inv**2 @ U.T
    
    eig_vals, eig_vecs = np.linalg.eig(A_pseudo_inv @ B @ B.T)
    w = eig_vecs[:, np.argmax(eig_vals)]  # Select the first eigenvector
    
    return w

def compute_anomaly_degree(w, o, R):
    """Compute AD(o) based on variance and distance metrics."""
    projected_o = w.T @ o
    projected_R = w.T @ R.T  # Project neighbors
    
    mean_proj_R = np.mean(projected_R, axis=0)
    var_R = np.var(projected_R)
    
    numerator = (projected_o - mean_proj_R).T @ (projected_o - mean_proj_R)
    anomaly_score = max(np.sqrt(numerator / var_R), np.sqrt(var_R))
    
    return anomaly_score

def compute_local_anomaly_degree(outliers, data, s=10):
    """Compute the LAD score for each outlier."""
    nbrs = NearestNeighbors(n_neighbors=s).fit(data)
    lad_scores, feature_contributions = {}, {}
    
    for o_idx in outliers:
        distances, indices = nbrs.kneighbors([data[o_idx]])
        R = data[indices[0][1:]]  # Exclude the outlier itself
        
        A = R - np.mean(R, axis=0)  # Centered neighbor matrix
        B = (data[o_idx] - R).T  # Distance matrix
        
        w = compute_optimal_direction(A, B)
        ad_o = compute_anomaly_degree(w, data[o_idx], R)
        
        ad_neighbors = np.array([compute_anomaly_degree(w, data[i], R) for i in indices[0][1:]])
        lad_o = ad_o / np.mean(ad_neighbors)
        
        lad_scores[o_idx] = lad_o
        feature_contributions[o_idx] = np.abs(w * (data[o_idx] - np.mean(R, axis=0)))

    
    return 


# Assume `X` is your dataset and `outlier_indices` are detected outliers
lad_scores, feature_contributions = compute_local_anomaly_degree(outlier_indices, X)

# Print results
for idx in outlier_indices:
    print(f"Outlier at index {idx}:")
    print(f"  LAD Score: {lad_scores[idx]}")
    print(f"  Feature Contributions: {feature_contributions[idx]}")

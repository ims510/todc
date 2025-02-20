"""
kmodes is used for clustering categorical data. 
article used for the code: https://www.analyticsvidhya.com/blog/2021/06/kmodes-clustering-algorithm-for-categorical-data/
"""
import pandas as pd
from kmodes.kmodes import KModes
import numpy as np
import matplotlib.pyplot as plt


data = pd.read_csv('data/output/ro_rrt-ud-train.csv')
# data = pd.read_csv("data/output/test.csv")
# replace nan values with 'nan'
data = data.fillna('nan')

# Elbow curve to find optimal K
# cost = []
# K = range(1,49)
# for num_clusters in list(K):
#     kmode = KModes(n_clusters=num_clusters, init = "random", n_init = 5, verbose=1)
#     kmode.fit_predict(data)
#     cost.append(kmode.cost_)
    
# plt.plot(K, cost, 'bx-')
# plt.xlabel('No. of clusters')
# plt.ylabel('Cost')
# plt.title('Elbow Method For Optimal k')
# plt.show()

# Building the model with 3 clusters
kmode = KModes(n_clusters=49, init = "random", n_init = 5, verbose=1)
clusters = kmode.fit_predict(data)
data.insert(0, "Cluster", clusters, True)
data.to_csv("data/output/ro_rrt-ud-train_clustered.csv", index=False)
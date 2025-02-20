import pandas as pd
import plotly.express as px
from umap import UMAP
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans

# Load data
complete_data_df = pd.read_csv('data/output/ro_rrt-ud-train_complete_categ.csv')
removed_deps_df = pd.read_csv('data/output/ro_rrt-ud-train_nummod-amod-det_categ.csv')

# Encode categorical features
label_encoder = LabelEncoder()
for column in complete_data_df.columns:
    if complete_data_df[column].dtype == 'object':
        complete_data_df[column] = label_encoder.fit_transform(complete_data_df[column].astype(str))

labels = complete_data_df['deprel']
data = complete_data_df.drop(columns=['deprel'])

# Perform clustering
n_bins = removed_deps_df['deprel'].nunique()
kmeans = KMeans(init="k-means++", n_clusters=n_bins, n_init=4, random_state=0)
kmeans.fit(data)

# Fit UMAP
umap_2d = UMAP(random_state=0)
umap_2d.fit(data)

# Transform data
projections = umap_2d.transform(data)

# Create a DataFrame for the projections
projections_df = pd.DataFrame(projections, columns=['UMAP1', 'UMAP2'])
projections_df['cluster'] = kmeans.labels_.astype(str)

# Visualize the UMAP projections
fig = px.scatter(
    projections_df, x='UMAP1', y='UMAP2',
    color='cluster', labels={'color': 'Cluster'}
)
fig.show()
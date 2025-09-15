from dash import Dash, dcc, html, Input, Output
import sys
sys.path.insert(1, "/Users/madalina/Documents/M2TAL/stage/check_coherent_labels/tod")
from tod.plotting import (
    lexunit_scatter_plot,
    cluster_scatter_plot,
    cluster_piechart_pos,
    outlier_scatter_plot,
    outlier_rainy_plot,
    outlier_dotted_line_plot,
    outlier_piechart_pos,
)
from tod.corpus import Corpus
from tod.dimension_reduction_classic import Tsne
from tod.clustering import HierarchicalClustering
from tod.outliers import LOF

# Initialize your data objects (replace with actual initialization)
corpus = Corpus( treebank_path="/Users/madalina/Documents/M1TAL/stage-SK/Treebanks/UD_French-GSD-master",
    grew_pattern="pattern{X[upos<>PUNCT]}",
    patterns_text_file="/Users/madalina/Documents/M2TAL/stage/check_coherent_labels/scripts/3. probability_matrix/patterns_all_nodes.txt",
    matrix_type="coverage")  
dimension_reduction = Tsne(corpus)  # Replace with your DimensionReduction initialization
clustering = HierarchicalClustering(corpus)  # Replace with your Clustering initialization
outlier_detector = LOF(corpus)  # Replace with your OutlierDetector initialization

# Initialize the Dash app
app = Dash(__name__)

# Define the layout
app.layout = html.Iframe()
app.layout = html.Div([
    dcc.Tabs([
        dcc.Tab(label="Scatter Plots", children=[
            html.Div([
                dcc.Dropdown(
                    id="scatter-plot-dropdown",
                    options=[
                        {"label": "Lexical Units Scatter Plot", "value": "lexunit"},
                        {"label": "Clusters Scatter Plot", "value": "clusters"},
                        {"label": "Outliers Scatter Plot", "value": "outliers"},
                    ],
                    value="lexunit",
                    placeholder="Select a scatter plot",
                ),
                dcc.Graph(id="scatter-plot"),
            ])
        ]),
        dcc.Tab(label="Pie Charts", children=[
            html.Div([
                dcc.Dropdown(
                    id="pie-chart-dropdown",
                    options=[
                        {"label": "Cluster POS Pie Chart", "value": "cluster_pie"},
                        {"label": "Outlier POS Pie Chart", "value": "outlier_pie"},
                    ],
                    value="cluster_pie",
                    placeholder="Select a pie chart",
                ),
                dcc.Graph(id="pie-chart"),
            ])
        ]),
        dcc.Tab(label="Other Plots", children=[
            html.Div([
                dcc.Dropdown(
                    id="other-plot-dropdown",
                    options=[
                        {"label": "Outlier Rainy Plot", "value": "rainy"},
                        {"label": "Outlier Dotted Line Plot", "value": "dotted_line"},
                    ],
                    value="rainy",
                    placeholder="Select another plot",
                ),
                dcc.Graph(id="other-plot"),
            ])
        ]),
    ])
])

# Define callbacks for scatter plots
@app.callback(
    Output("scatter-plot", "figure"),
    Input("scatter-plot-dropdown", "value")
)
def update_scatter_plot(plot_type):
    if plot_type == "lexunit":
        return lexunit_scatter_plot(corpus, dimension_reduction)
    elif plot_type == "clusters":
        return cluster_scatter_plot(corpus, dimension_reduction, clustering)
    elif plot_type == "outliers":
        return outlier_scatter_plot(corpus, dimension_reduction, outlier_detector)
    return {}

# Define callbacks for pie charts
@app.callback(
    Output("pie-chart", "figure"),
    Input("pie-chart-dropdown", "value")
)
def update_pie_chart(chart_type):
    if chart_type == "cluster_pie":
        return cluster_piechart_pos(corpus, clustering, dropdown=False)
    elif chart_type == "outlier_pie":
        return outlier_piechart_pos(corpus, outlier_detector, dropdown=False)
    return {}

# Define callbacks for other plots
@app.callback(
    Output("other-plot", "figure"),
    Input("other-plot-dropdown", "value")
)
def update_other_plot(plot_type):
    if plot_type == "rainy":
        return outlier_rainy_plot(corpus, outlier_detector)
    elif plot_type == "dotted_line":
        return outlier_dotted_line_plot(corpus, outlier_detector)
    return {}

# Run the app
if __name__ == "__main__":
    app.run_server(debug=True)
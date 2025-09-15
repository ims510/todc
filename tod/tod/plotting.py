from .corpus import Corpus
import numpy as np
from .dimension_reduction_classic import DimensionReduction
from .clustering import Clustering
from .outliers import OutlierDetector
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots


def lexunit_scatter_plot(corpus: Corpus, dimension_reduction: DimensionReduction):
    data = pd.DataFrame(
        {
            "Component 1": dimension_reduction.reduced_matrix[:, 0],
            "Component 2": dimension_reduction.reduced_matrix[:, 1],
            "Lexical Unit": [
                corpus.idx2lexunit(i)
                for i in range(len(dimension_reduction.reduced_matrix))
            ],
        }
    )

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=data["Component 1"],
            y=data["Component 2"],
            mode="markers",
            marker=dict(size=10),
            text=data["Lexical Unit"],
            hovertemplate="%{text}<extra></extra>",
            hoverinfo="text",
        )
    )
    fig.update_layout(
        title="Lexical Units Scatter Plot",
        xaxis_title="Component 1",
        yaxis_title="Component 2",
        showlegend=False,
    )

    return fig

def cluster_scatter_plot(
    corpus: Corpus, dimension_reduction: DimensionReduction, clustering: Clustering
):

    data = pd.DataFrame(
        {
            "Component 1": dimension_reduction.reduced_matrix[:, 0],
            "Component 2": dimension_reduction.reduced_matrix[:, 1],
            "Cluster": [
                clustering.lexunit2cluster(corpus.idx2lexunit(i))
                for i in range(len(dimension_reduction.reduced_matrix))
            ],
            "Lexical Unit": [
                corpus.idx2lexunit(i)
                for i in range(len(dimension_reduction.reduced_matrix))
            ],
        }
    )

    fig = go.Figure()
    for cluster in sorted(data["Cluster"].unique()):
        cluster_data = data[data["Cluster"] == cluster]
        fig.add_trace(
            go.Scatter(
                x=cluster_data["Component 1"],
                y=cluster_data["Component 2"],
                mode="markers",
                marker=dict(size=10),
                name=f"Cluster {cluster}",
                text=cluster_data["Lexical Unit"],
                hovertemplate="%{text}<extra></extra>",
                hoverinfo="text",
            )
        )
    fig.update_layout(
        title="Clusters Scatter Plot",
        xaxis_title="Component 1",
        yaxis_title="Component 2",
        updatemenus=[
            {
                "buttons": [
                    {
                        "label": "All Clusters",
                        "method": "update",
                        "args": [
                            {"visible": [True] * len(clustering.clusters)},
                            {"title": "All Clusters"},
                        ],
                    }
                ]
                + [
                    {
                        "label": f"Cluster {i}",
                        "method": "update",
                        "args": [
                            {
                                "visible": [
                                    j == i - 1 for j in range(len(clustering.clusters))
                                ]
                            },
                            {"title": f"Cluster {i}"},
                        ],
                    }
                    for i in range(1, len(clustering.clusters) + 1)
                ],
                "direction": "down",
                "showactive": True,
            }
        ],
    )
    return fig

def cluster_piechart_pos(corpus: Corpus, clustering: Clustering, dropdown: bool = True):
    clusters = {i: clustering.cluster2lexunit(i) for i in range(1,len(clustering.clusters))}
    pie_chart = {}
    for cluster, members in clusters.items():
        pie_chart[cluster] = {}
        for member in members:
            if member[1] in pie_chart[cluster]:
                pie_chart[cluster][member[1]] += 1
            else:
                pie_chart[cluster][member[1]] = 1

    if dropdown:
        # Function to create pie chart for a given cluster
        def create_pie_chart(cluster_number):
            labels = [f'{k} ({v})' for k, v in pie_chart[cluster_number].items()]
            values = list(pie_chart[cluster_number].values())
            fig = go.Figure(data=[go.Pie(labels=labels, values=values)])
            return fig
        
        fig = create_pie_chart(1)

        # Add dropdown menu
        dropdown_buttons = [
            {
                'label': f'Cluster {i}',
                'method': 'update',
                'args': [{'values': [list(pie_chart[i].values())], 'labels': [[f'{k} ({v})' for k, v in pie_chart[i].items()]]}]
            } for i in pie_chart.keys()
        ]

        fig.update_layout(
            updatemenus=[
                {
                    'buttons': dropdown_buttons,
                    'direction': 'down',
                    'showactive': True,
                }
            ]
        )

        return fig
    
    else:
        # Dynamically generate a color palette for all unique categories
        all_categories = set(cat for cluster in pie_chart.values() for cat in cluster.keys())
        color_palette = px.colors.qualitative.Plotly  # Use Plotly's default qualitative color palette
        category_colors = {cat: color_palette[i % len(color_palette)] for i, cat in enumerate(sorted(all_categories))}

        # Determine the number of rows and columns for the subplots
        num_clusters = len(pie_chart)
        cols = 4  # Fixed number of columns
        rows = -(-num_clusters // cols)  # Calculate rows dynamically (ceiling division)

        # Create a subplot layout
        fig = make_subplots(rows=rows, cols=cols, specs=[[{'type': 'domain'}]*cols]*rows)

        # Add each pie chart to the subplot
        for i, (cluster_number, cluster_data) in enumerate(pie_chart.items(), start=1):
            labels = list(cluster_data.keys())
            values = list(cluster_data.values())
            colors = [category_colors[label] for label in labels]
            
            fig.add_trace(
                go.Pie(labels=labels, values=values, marker=dict(colors=colors), name=f'Cluster {cluster_number}'),
                row=(i-1)//cols + 1, col=(i-1)%cols + 1
            )

        # Add a dummy pie chart to create a legend
        legend_labels = list(category_colors.keys())
        legend_values = [1] * len(legend_labels)  # Dummy values for the legend
        legend_colors = [category_colors[label] for label in legend_labels]

        fig.add_trace(
            go.Pie(
                labels=legend_labels,
                values=legend_values,
                marker=dict(colors=legend_colors),
                name="Legend",
                showlegend=True,
                hoverinfo="label"  # Only show labels on hover
            ),
            row=1, col=cols  # Place the legend in the first row, last column
        )

        # Update layout
        fig.update_layout(
            title_text="Dynamic Pie Charts for Each Cluster",
            height=300 * rows,  # Adjust height dynamically based on rows
            width=1200,  # Fixed width
            showlegend=True,
            legend=dict(
                x=1.05,  # Position the legend to the right of the chart
                y=0.5,
                traceorder="normal"
            )
        )

        return fig

def outlier_scatter_plot(
    corpus: Corpus,
    dimension_reduction: DimensionReduction,
    outlier_detector: OutlierDetector,
):
    data = pd.DataFrame(
        {
            "Component 1": dimension_reduction.reduced_matrix[:, 0],
            "Component 2": dimension_reduction.reduced_matrix[:, 1],
            "Outlier": [
                (
                    "Outlier"
                    if corpus.idx2lexunit(i) in outlier_detector.outliers
                    else "Inlier"
                )
                for i in range(len(dimension_reduction.reduced_matrix))
            ],
            "Lexical Unit": [
                corpus.idx2lexunit(i)
                for i in range(len(dimension_reduction.reduced_matrix))
            ],
        }
    )
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=data["Component 1"],
            y=data["Component 2"],
            mode="markers",
            marker=dict(
                size=10,
                color=["red" if o == "Outlier" else "blue" for o in data["Outlier"]],
            ),
            text=data["Lexical Unit"],
            customdata=data["Outlier"],
            hovertemplate="Word: %{text}<br>Status: %{customdata}<extra></extra>",
        )
    )

    fig.update_layout(
        title="Outlier Scatter Plot",
        xaxis_title="Component 1",
        yaxis_title="Component 2",
    )
    return fig

def outlier_rainy_plot(corpus: Corpus, outlier_detector: OutlierDetector):

    # Group elements by POS
    inlier_pos_elements = {}
    for element, pos in outlier_detector.inliers:
        inlier_pos_elements.setdefault(pos, []).append(element)

    outlier_pos_elements = {}
    for element, pos in outlier_detector.outliers:
        outlier_pos_elements.setdefault(pos, []).append(element)

    elements_score = {corpus.idx2lexunit(i): outlier_detector.scores[i] for i in range(len(outlier_detector.scores))}
    score_pos_elements = {}
    for i in range(len(outlier_detector.scores)):
        element, pos = corpus.idx2lexunit(i)
        score_pos_elements.setdefault(pos, []).append((element, elements_score[(element, pos)]))
    data = []
    for pos, words in score_pos_elements.items():
        for word, score in words:
            status = 'Outlier' if (word, pos) in outlier_detector.outliers else 'Inlier'
            color = 'red' if status == 'Outlier' else 'blue'
            data.append({'POS': pos, 'Word': word, 'Score': score, 'Status': status, 'Color': color})

    df = pd.DataFrame(data)

    # Prepare dropdown and scatter plot
    pos_tags = list(score_pos_elements.keys())
    fig = go.Figure()

    # Add traces for each POS
    def create_trace(pos):
        filtered_df = df[df['POS'] == pos]
        return go.Scatter(
            x=np.arange(len(filtered_df)),
            y=filtered_df['Score'],
            text=[f"{row['Word']} ({row['Status']}), Score: {row['Score']:.2f}" for _, row in filtered_df.iterrows()],
            mode='markers',
            marker=dict(color=filtered_df['Color'], size=10),
            name=pos,
            hoverinfo='text'
        )

    for pos in pos_tags:
        fig.add_trace(create_trace(pos))

    # Update layout with dropdown
    fig.update_layout(
        title="Outlier Score per POS",
        xaxis=dict(showticklabels=False),
        updatemenus=[
            {
                "buttons": [
                    {
                        "label": pos,
                        "method": "update",
                        "args": [
                            {"visible": [i == idx for i in range(len(pos_tags))]},
                            {"title": f"Outlier Score for {pos}"}
                        ]
                    }
                    for idx, pos in enumerate(pos_tags)
                ],
                "direction": "down",
                "showactive": True
            }
        ]
    )

    # Set all traces except the first one to be initially hidden
    for i in range(1, len(pos_tags)):
        fig.data[i].visible = False
    
    return fig

def outlier_dotted_line_plot(corpus: Corpus, outlier_detector: OutlierDetector):
    # Group elements by POS
    inlier_pos_elements = {}
    for element, pos in outlier_detector.inliers:
        inlier_pos_elements.setdefault(pos, []).append(element)

    outlier_pos_elements = {}
    for element, pos in outlier_detector.outliers:
        outlier_pos_elements.setdefault(pos, []).append(element)

    elements_score = {corpus.idx2lexunit(i): outlier_detector.scores[i] for i in range(len(outlier_detector.scores))}
    score_pos_elements = {}
    for i in range(len(outlier_detector.scores)):
        element, pos = corpus.idx2lexunit(i)
        score_pos_elements.setdefault(pos, []).append((element, elements_score[(element, pos)]))
    data = []
    for pos, words in score_pos_elements.items():
        for word, score in words:
            status = 'Outlier' if (word, pos) in outlier_detector.outliers else 'Inlier'
            # color = 'red' if status == 'Outlier' else 'blue'
            # data.append({'POS': pos, 'Word': word, 'Score': score, 'Status': status, 'Color': color})
            data.append({'POS': pos, 'Word': word, 'Score': abs(score), 'Status': status})

    df = pd.DataFrame(data)
    df = df.sort_values(by="Score")
    # Prepare dropdown and scatter plot
    # Add a color column based on the Outlier status
    df['Color'] = df['Status'].apply(lambda x: 'blue' if x == 'Inlier' else 'red')

    fig = go.Figure()
    # Create the scatter plot
    fig = px.scatter(
        df,
        x="Score",
        y="POS",
        # color="Color",  # Use the color column for coloring
        hover_data={"Color": False, "Score": True, "Word": True},  # Show Complete_feature on hover
        # color_discrete_map={"blue": "blue", "red": "red"}  # Map colors explicitly
    )

    # Update layout for better visualization
    fig.update_traces(marker=dict(size=10))  # Adjust marker size
    fig.update_layout(
        title="Outlier Score by POS",
        xaxis_title="Score",
        yaxis_title="POS",
        height=800,  # Set the height of the plot
        showlegend=False  # Hide legend since colors are self-explanatory
    )
    
    return fig

def outlier_piechart_pos(corpus: Corpus, outlier_detector: OutlierDetector, dropdown: bool = True):
    # Group elements by POS
    inlier_pos_elements = {}
    for element, pos in outlier_detector.inliers:
        inlier_pos_elements.setdefault(pos, []).append(element)

    outlier_pos_elements = {}
    for element, pos in outlier_detector.outliers:
        outlier_pos_elements.setdefault(pos, []).append(element)

    pos_tags = list(set(inlier_pos_elements.keys()).union(outlier_pos_elements.keys()))

    if dropdown:
        fig = go.Figure()

        # Add traces for each POS
        for pos in pos_tags:
            inlier_words = inlier_pos_elements.get(pos, [])
            outlier_words = outlier_pos_elements.get(pos, [])
            
            # Calculate percentages
            # total = len(inlier_words) + len(outlier_words)
            # inlier_percentage = len(inlier_words) / total * 100 if total > 0 else 0
            # outlier_percentage = len(outlier_words) / total * 100 if total > 0 else 0
            
            # Create hover text
            inlier_hover_text = '<br>'.join(inlier_words)
            outlier_hover_text = '<br>'.join(outlier_words)
            
            # Add a trace for this POS
            fig.add_trace(go.Pie(
                labels=['Inliers', 'Outliers'],
                values=[len(inlier_words), len(outlier_words)],
                hovertext=[inlier_hover_text, outlier_hover_text],
                hoverinfo="text",
                name=pos,
                marker=dict(colors=['blue', 'red'])
            ))

        # Update layout with dropdown
        fig.update_layout(
            title="POS Distribution of Inliers and Outliers",
            updatemenus=[
                {
                    "buttons": [
                        {
                            "label": pos,
                            "method": "update",
                            "args": [
                                {"visible": [i == idx for i in range(len(pos_tags))]},
                                {"title": f"POS Distribution for {pos}"}
                            ]
                        }
                        for idx, pos in enumerate(pos_tags)
                    ],
                    "direction": "down",
                    "showactive": True
                }
            ]
        )

        # Set all traces except the first one to be initially hidden
        for i in range(1, len(pos_tags)):
            fig.data[i].visible = False
        return fig
    else:
        # Determine the number of rows and columns for the subplots
        num_pos_tags = len(pos_tags)
        cols = 4  # Fixed number of columns
        rows = -(-num_pos_tags // cols)  # Calculate rows dynamically (ceiling division)

        # Create a subplot layout
        fig = make_subplots(rows=rows, cols=cols, specs=[[{'type': 'domain'}]*cols]*rows, subplot_titles=pos_tags)

        # Add a pie chart for each POS
        for i, pos in enumerate(pos_tags, start=1):
            inlier_words = inlier_pos_elements.get(pos, [])
            outlier_words = outlier_pos_elements.get(pos, [])
            
            # Calculate percentages
            total = len(inlier_words) + len(outlier_words)
            inlier_percentage = len(inlier_words) / total * 100 if total > 0 else 0
            outlier_percentage = len(outlier_words) / total * 100 if total > 0 else 0
            
            # Create hover text
            inlier_hover_text = '<br>'.join(inlier_words)
            outlier_hover_text = '<br>'.join(outlier_words)
            
            # Add the pie chart to the subplot
            fig.add_trace(
                go.Pie(
                    labels=['Inliers', 'Outliers'],
                    values=[len(inlier_words), len(outlier_words)],
                    hovertext=[inlier_hover_text, outlier_hover_text],
                    hoverinfo="text",
                    marker=dict(colors=['blue', 'red']),
                    name=pos
                ),
                row=(i-1)//cols + 1, col=(i-1)%cols + 1
            )

                        # Add a title for the subplot
            # fig.add_annotation(
            #     text=pos,
            #     x=((i-1) % cols + 0.5) / cols,  # Center the title in the column
            #     y=1 - ((i-1) // cols) / rows + 0.01,  # Increase the offset above the subplot
            #     xref="paper",
            #     yref="paper",
            #     showarrow=False,
            #     font=dict(size=12)
            # )

        # Update layout
        fig.update_layout(
            title="POS Distribution of Inliers and Outliers",
            height=300 * rows,  # Adjust height dynamically based on rows
            width=1200,  # Fixed width
            showlegend=False  # Hide the global legend
        )

        # # Add titles to each subplot
        # for i, pos in enumerate(pos_tags, start=1):
        #     fig.update_annotations([
        #         dict(
        #             text=pos,
        #             x=((i-1) % cols + 0.5) / cols,  # Center the title in the column
        #             y=1 - ((i-1) // cols) / rows,  # Position the title above the subplot
        #             xref="paper",
        #             yref="paper",
        #             showarrow=False,
        #             font=dict(size=12)
        #         )
        #     ])
        return fig
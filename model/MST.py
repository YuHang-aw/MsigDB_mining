import cupy as cp
import dask.array as da
import networkx as nx
import plotly.graph_objects as go
import numpy as np
from config import group_to_color,index_to_group

def reverse_dict(d):
    """Reverse the key-value pairs of a dictionary."""
    return {v: k for k, v in d.items()}

def generate_mst_from_files(files, distance_matrix, output_html='../pic/plotly_graph_filtered.html', output_pdf='../pic/plotly_graph_filtered.pdf'):
    """
    Generate the minimum spanning tree (MST) from the provided files and distance matrix.

    Parameters:
    - files: A tuple containing (data, term_to_index, gene_to_index).
    - distance_matrix: The distance matrix used to generate the MST.
    - output_html: The file path for saving the HTML plot.
    - output_pdf: The file path for saving the PDF plot.
    """
    
    _, term_to_index, _= files
    
    # Reverse the term_to_index dictionary to get index_to_term
    index_to_term = reverse_dict(term_to_index)

    # Calculate the minimum index to re-map indices starting from 0
    
    index_to_terms = {k: v.replace('human_t_', '') for k, v in index_to_term.items()}

    # Convert the distance matrix to a Dask array and use CuPy for computation
    distance_matrix_dask = da.from_array(distance_matrix, chunks=(1000, 1000))
    distance_matrix_gpu = cp.asarray(distance_matrix_dask)

    # Keep only the shortest 5% of distances, setting the rest to infinity
    percentile = 0.05
    flat_distance_matrix_gpu = distance_matrix_gpu.ravel()
    flat_distance_matrix_gpu = flat_distance_matrix_gpu.astype(cp.float32)
    threshold = cp.percentile(flat_distance_matrix_gpu, percentile)
    distance_matrix_gpu = cp.where(distance_matrix_gpu <= threshold, distance_matrix_gpu, cp.inf)

    # Filter the involved nodes
    valid_nodes = cp.any(distance_matrix_gpu != cp.inf, axis=0)
    valid_indices = cp.where(valid_nodes)[0].get()  # Convert CuPy array to NumPy array

    # Create a subgraph from valid nodes
    distance_matrix_cpu = cp.asnumpy(distance_matrix_gpu)
    distance_matrix = distance_matrix_cpu[np.ix_(valid_indices, valid_indices)]

    # Select the top 1000 closest nodes
    distance_matrix_cpu[distance_matrix_cpu == np.inf] = np.nan

    # Get the top 1000 smallest distances
    sorted_indices = np.unravel_index(np.argsort(distance_matrix_cpu, axis=None)[:1000], distance_matrix_cpu.shape)

    # Get the indices of the top 1000 closest nodes
    closest_nodes = np.unique(np.concatenate((sorted_indices[0], sorted_indices[1])))

    # Extract the submatrix corresponding to these nodes from the distance matrix
    filtered_distance_matrix = distance_matrix_cpu[np.ix_(closest_nodes, closest_nodes)]

    # Create a graph using the edge list
    graph = nx.Graph()

    for i in range(len(closest_nodes)):
        for j in range(i + 1, len(closest_nodes)):
            if not np.isnan(filtered_distance_matrix[i, j]):
                graph.add_edge(closest_nodes[i], closest_nodes[j], weight=filtered_distance_matrix[i, j])

    # Compute the minimum spanning tree
    mst = nx.minimum_spanning_tree(graph)

    # Rename the nodes
    graph = nx.relabel_nodes(graph, index_to_terms)
    mst = nx.relabel_nodes(mst, index_to_terms)

    node_colors = [group_to_color[index_to_group[node]] for node in mst.nodes()]

    # Determine node positions using a network layout algorithm
    pos = nx.spring_layout(mst, k=0.15, iterations=50)

    # Create edge coordinates for Plotly
    edge_x = []
    edge_y = []
    for u, v in mst.edges():
        edge_x.extend([pos[u][0], pos[v][0], None])
        edge_y.extend([pos[u][1], pos[v][1], None])

    # Create node coordinates for Plotly
    node_x = [pos[node][0] for node in mst.nodes()]
    node_y = [pos[node][1] for node in mst.nodes()]

    # Create the Plotly figure
    fig = go.Figure()

    fig.add_trace(go.Scattergl(
        x=edge_x, y=edge_y,
        mode='lines',
        line=dict(color='grey', width=1),
        hoverinfo='none'
    ))

    fig.add_trace(go.Scattergl(
        x=node_x, y=node_y,
        mode='markers',
        marker=dict(color=node_colors, size=5),  # Adjust node size
        text=[f'{node}: {index_to_group[node]}' for node in mst.nodes()],
        hoverinfo='text',
        hovertemplate='%{text}<extra></extra>',  # Only display text
        customdata=[node for node in mst.nodes()]
    ))

    # Add node labels to the PDF
    fig.add_trace(go.Scattergl(
        x=node_x, y=node_y,
        mode='text',
        text=[f'{node}' for node in mst.nodes()],
        textposition="top center",  # Adjust text position
        textfont=dict(size=8),  # Use smaller font size to avoid overlap
        hoverinfo='none'
    ))

    # Add legend
    for group, color in group_to_color.items():
        fig.add_trace(go.Scattergl(
            x=[None], y=[None], mode='markers',
            marker=dict(size=10, color=color),
            legendgroup=group,
            name=group
        ))

    fig.update_layout(
        showlegend=True,
        hovermode='closest',
        margin=dict(b=40, l=40, r=120, t=40),  # Adjust margins, increase right margin
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        template="plotly_white",
        width=2000,  # Adjust figure width
        height=2000,  # Adjust figure height
        legend=dict(x=1.05, y=1)  # Move the legend to the right side of the figure
    )

    # Save as HTML and PDF
    fig.write_html(output_html, include_plotlyjs='cdn', full_html=True, auto_open=False)
    fig.write_image(output_pdf)

    print(f'Graph successfully saved to {output_html} and {output_pdf}.')


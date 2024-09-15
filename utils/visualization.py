import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import torch
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from config import group_to_color, index_to_group
import umap.umap_ as umap

def reverse_dict(d):
    """Reverse the key-value pairs of a dictionary."""
    return {v: k for k, v in d.items()}

def norm_alpha(degree, degrees, min_alpha=0.1, max_alpha=1.0):
    """Normalize the alpha value to adapt to matplotlib scatter plot."""
    max_degree = degrees.max() if degrees.size > 0 else 1  # Ensure no division by zero
    return min_alpha + (max_alpha - min_alpha) * (degree / max_degree)

@torch.no_grad()
def plot_embeddings(files, model, method='tsne', output_file='output.pdf'):
    """Function to visualize embedding points of Terms, supporting t-SNE, PCA, and UMAP methods.
    
    Args:
    - files: Human_collections through preprocessing 
    - model: Model used to generate node embeddings.
    - method: Dimensionality reduction method, can be 'tsne', 'pca', or 'umap'.
    - output_file: Filename to save the output image.
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    data, term_to_index, gene_to_index = files

    index_to_term = reverse_dict(term_to_index)
    index_to_terms = {k: v.replace('human_t_', '') for k, v in index_to_term.items()}

    node_type = np.array([0 if index < len(term_to_index) else 1 for index in range(len(gene_to_index) + len(term_to_index))])
    degrees = data.edge_index[0].bincount().numpy()

    model.eval()
    z_all = model(torch.arange(data.num_nodes, device=device))
    z_term = z_all[node_type == 0].cpu().numpy()

    if method == 'tsne':
        z_term = TSNE(n_components=2, init='random', learning_rate=10).fit_transform(z_term)
        xlabel, ylabel = 't-SNE 1', 't-SNE 2'
    elif method == 'pca':
        z_term = PCA(n_components=2).fit_transform(z_term)
        xlabel, ylabel = 'PCA 1', 'PCA 2'
    elif method == 'umap':
        umap_model = umap.UMAP(n_components=2, init='random', n_neighbors=5, min_dist=0.1)
        z_term = umap_model.fit_transform(z_term)
        xlabel, ylabel = 'UMAP 1', 'UMAP 2'
    else:
        raise ValueError(f"Unsupported method: {method}")

    fig, ax = plt.subplots(figsize=(10, 10))
    for i, pos in enumerate(z_term):
        node_index = np.where(node_type == 0)[0][i]
        term_name = index_to_terms[node_index]
        group = index_to_group.get(term_name)
        color = group_to_color[group]
        alpha = norm_alpha(degrees[node_index], degrees)
        ax.scatter(pos[0], pos[1], color=color, alpha=alpha, s=50, edgecolor='k', linewidth=0.5)

    # Add custom legend
    handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=10, label=group)
            for group, color in group_to_color.items()]
    ax.legend(handles=handles, title='Groups', loc='upper right', fontsize='small', title_fontsize='medium')

    # Display axes and set labels
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    plt.savefig(output_file, dpi=1000)
    #plt.show()

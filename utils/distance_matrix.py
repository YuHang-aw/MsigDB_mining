import numpy as np
import torch
import os
import pickle
from tqdm import tqdm

def distance_matrix(model, files, file_path='../checkpoints/distance_matrix.pkl'):
    """
    Compute or load the distance matrix for terms.
    Returns:
    - distance_matrix: The computed or loaded distance matrix.
    """
    
    # If the file exists, load and return the distance matrix
    if os.path.exists(file_path):
        print(f"Loading distance matrix from {file_path}...")
        with open(file_path, 'rb') as f:
            distance_matrix = pickle.load(f)
        return distance_matrix
    
    data, term_to_index, gene_to_index = files
    node_type = np.array([0 if index < len(term_to_index) else 1 for index in range(len(gene_to_index) + len(term_to_index))])

    # If the file does not exist, compute the distance matrix
    print("Computing distance matrix...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    z_all = model(torch.arange(data.num_nodes, device=device))
    z_term = z_all[node_type == 0]

    # Move the data to the GPU
    z_term_tensor = z_term.to(device)

    N = z_term_tensor.shape[0]
    
    # Initialize a distance matrix to store the results
    distance_matrix = torch.zeros([N, N], device=device)

    # Compute the distance in batches on the GPU, processing a portion of the data at a time
    batch_size = 1000
    for i in tqdm(range(0, N, batch_size)):
        # Select a batch of data
        batch = z_term_tensor[i: i+batch_size]
        
        # Compute the distance between this batch and the entire dataset
        dists = torch.cdist(batch, z_term_tensor, p=2.0)
        
        # Store the results
        distance_matrix[i: i+batch_size] = dists

    # Move the results to the CPU and convert to a numpy array
    distance_matrix = distance_matrix.to('cpu').detach().numpy()

    # Save the distance matrix to a file
    with open(file_path, 'wb') as f:
        pickle.dump(distance_matrix, f)

    return distance_matrix

import pandas as pd
import torch
from torch_geometric.data import Data
import os
import pickle

def process_data(file_path):
    # Extract species prefix from the filename
    species_prefix = os.path.basename(file_path).split('_')[0]
    
    # Check if the output file already exists
    output_file = f'../checkpoints/{species_prefix}_network.pkl'
    if os.path.exists(output_file):
        print(f"File {output_file} already exists. Loading data.")
        with open(output_file, 'rb') as f:
            data, term_to_index, gene_to_index = pickle.load(f)
        return data, term_to_index, gene_to_index

    # Read the CSV file
    df = pd.read_csv(file_path)
    
    # Add species prefix to 'term' and 'gene'
    df.iloc[:, 0] = species_prefix + '_t_' + df.iloc[:, 0].astype(str)
    df.iloc[:, 1] = species_prefix + '_g_' + df.iloc[:, 1].astype(str)

    # Create indices
    terms = df.iloc[:, 0].unique().tolist()
    genes = df.iloc[:, 1].unique().tolist()

    term_to_index = {term: index for index, term in enumerate(terms)}
    gene_to_index = {gene: index + len(terms) for index, gene in enumerate(genes)}

    # Create edge index
    edge_index = torch.tensor(
        [[term_to_index[row[0]], gene_to_index[row[1]]] for row in df.values],
        dtype=torch.long
    ).t().contiguous()

    # Create PyG Data object
    num_nodes = len(terms) + len(genes)
    data = Data(edge_index=edge_index, num_nodes=num_nodes)

    # Save to pickle file
    with open(output_file, 'wb') as f:
        pickle.dump((data, term_to_index, gene_to_index), f)
    
    print(f"Processed and saved data to {output_file}")
    return data, term_to_index, gene_to_index

# # Example usage
# if __name__ == "__main__":
#     file_path = '../Data/mice_collections.csv'
#     data, term_to_index, gene_to_index = process_data(file_path)
#     print("Data object:", data)
#     print("Term to index mapping:", term_to_index)
#     print("Gene to index mapping:", gene_to_index)

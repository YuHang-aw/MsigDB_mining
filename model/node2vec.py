from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch_geometric.nn import Node2Vec

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class MyNode2Vec(BaseEstimator, TransformerMixin):
    def __init__(self, edge_index, embedding_dimension=128, walk_length=80, 
    context_size=20, walks_per_node=10, p=1, q=1, num_negative_samples=1, 
    calculate_distances=True):
        self.edge_index = edge_index
        self.embedding_dimension = embedding_dimension
        self.walk_length = walk_length
        self.context_size = context_size
        self.walks_per_node = walks_per_node
        self.p = p
        self.q = q
        self.num_negative_samples = num_negative_samples
        self.calculate_distances = calculate_distances
        
        # Initialize the model in the constructor
        self.model = Node2Vec(self.edge_index, embedding_dim=self.embedding_dimension, walk_length=self.walk_length,
                              context_size=self.context_size, walks_per_node=self.walks_per_node, 
                              num_negative_samples=self.num_negative_samples,
                              sparse=True).to(device)
    
    def fit(self, data, y=None):
        # You no longer need to initialize the model here, as it's done in __init__
        loader = self.model.loader(batch_size=128, shuffle=True, num_workers=0)
        optimizer = torch.optim.SparseAdam(self.model.parameters(), lr=0.01)
        
        def train():
            self.model.train()
            total_loss = 0
            for pos_rw, neg_rw in loader:
                optimizer.zero_grad()
                loss = self.model.loss(pos_rw.to(device), neg_rw.to(device))
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            return total_loss / len(loader)
        
        loss_list = []
        patience = 10  # early stop
        min_delta = 0.001  # min change
        best_loss = np.inf
        counter = 0
        
        for epoch in range(1, 61):
            loss = train()
            loss_list.append(loss)

            if loss < best_loss - min_delta:
                best_loss = loss
                counter = 0  
            else:
                counter += 1  
            
            if counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break  

        print(f"Parameters: edge_index={self.edge_index.shape}, embedding_dim={self.embedding_dimension}, "
            f"walk_length={self.walk_length}, context_size={self.context_size}, "
            f"walks_per_node={self.walks_per_node}, num_negative_samples={self.num_negative_samples}, "
            f"sparse=True")
        print(f"Loss: {loss:.4f}")    
        
        return self

    def transform(self, data):
        # Generate new embeddings
        new_embeddings = self.model.forward(data.x).detach().cpu()
        
        if self.calculate_distances:
            distances = torch.cdist(new_embeddings, new_embeddings, p=2)
            nearest_neighbors_indices = distances.argsort()[:, 1:6]
            return new_embeddings, nearest_neighbors_indices
        else:
            return new_embeddings, None
        # # Optionally calculate distances
        # if calculate_distances:
        #     # Calculate pairwise distances between embeddings
        #     distances = torch.cdist(new_embeddings, new_embeddings, p=2)
            
        #     # Get the nearest neighbors for each node (excluding itself)
        #     nearest_neighbors_indices = distances.argsort()[:, 1:6]  # Exclude self
        #     return new_embeddings, nearest_neighbors_indices
        # else:
        #     # Skip distance calculation, only return embeddings
        #     return new_embeddings, None



    


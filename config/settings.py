import pickle
import os

param_list = {
    'n2v__embedding_dimension': [32, 64],
    'n2v__walk_length': [10, 20, 30],
    'n2v__p': [0.25, 0.50, 1.0, 2.0, 4.0],
    'n2v__q': [0.25, 0.50, 1.0, 2.0, 4.0],
    'n2v__context_size': [10, 20],
    'n2v__num_negative_samples': [1, 5, 10],
    'n2v__walks_per_node': [10, 20, 30]
}

group_to_color = {
    'c1': '#ff9999',  # light red
    'c2': '#66b3ff',  # light blue
    'c3': '#99ff99',  # light green
    'c4': '#ffcc99',  # light orange
    'c5': '#c2c2f0',  # light purple
    'c6': '#ffb3e6',  # light pink
    'c7': '#c4e17f',  # light lime
    'c8': '#ff6666',  # light coral
}


current_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(current_dir, 'Hs_index_to_group.pkl')

with open(file_path,'rb') as f:
    index_to_group = pickle.load(f)
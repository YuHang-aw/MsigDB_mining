import os
import numpy as np
from itertools import product
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.cluster import DBSCAN, KMeans
from joblib import dump, load 
#from model import MyNode2Vec

def run_hyperparameter_search(pipeline, data, param_list, results_path='../checkpoints/hyperparameter_search_checkpoint.pkl'):
    """Run hyperparameter search to find the best model configuration."""
    
    def Myscore_function(estimator, X):
        """Calculate a custom score based on matrix rank and number of clusters."""
        embeddings = estimator.transform(X)[0]
        covariance_matrix = np.cov(embeddings.T)
        rank = np.linalg.matrix_rank(covariance_matrix)

        clustering = DBSCAN().fit(embeddings)
        labels = clustering.labels_
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

        rank_weight = 0.7
        clustering_weight = 0.3
        return rank_weight * rank + clustering_weight * n_clusters
    
    def combined_score(silhouette, davies_bouldin, my_score):
        """Combine different scores into a final score with weights."""
        silhouette_weight = 0.3
        davies_bouldin_weight = 0.3
        my_score_weight = 0.4

        normalized_silhouette = silhouette / (1 if silhouette <= 1 else np.abs(silhouette))
        normalized_davies_bouldin = 1 / (1 + davies_bouldin)
        normalized_my_score = my_score / (1 if my_score <= 1 else np.abs(my_score))

        return (silhouette_weight * normalized_silhouette +
                davies_bouldin_weight * normalized_davies_bouldin +
                my_score_weight * normalized_my_score)

    def evaluate(pipeline, data):
        """Evaluate the model using clustering scores."""
        transformed_data, _ = pipeline.transform(data)
        
        cluster_model = KMeans(n_clusters=5).fit(transformed_data)
        labels = cluster_model.labels_
        
        silhouette = silhouette_score(transformed_data, labels)
        davies_bouldin = davies_bouldin_score(transformed_data, labels)
        my_score = Myscore_function(pipeline, data)  

        return silhouette, davies_bouldin, my_score

    # Initialize or load results
    try:
        if os.path.exists(results_path):
            results = load(results_path)
            print('Existing results loaded')
        else:
            results = {}
            print('No existing results found, starting new training')
    except Exception as e:
        print(f"Error loading results: {e}")
        results = {}
        print('Starting new training due to error')

    max_score = float('-inf')
    best_configuration = None

    if not param_list:
        print("Parameter list is empty. Please provide valid parameters.")
        return

    params = list(product(*param_list.values()))
    for param in params:
        params_str = str(param)
        if params_str in results:
            print(f'Results for parameters: {params_str} already exist, skipping this training')
            continue
        
        param_dict = dict(zip(param_list.keys(), param))
        if param_dict.get('n2v__context_size', 0) > param_dict.get('n2v__walk_length', 0):
            print(f'Parameters: {params_str} have context_size greater than walk_length, removing this set')
            continue

        pipeline.set_params(**param_dict)
        pipeline.fit(data)

        silhouette, davies_bouldin, my_score = evaluate(pipeline, data)

        results[params_str] = {"params": param_dict, 
                               "silhouette_score": silhouette, 
                               "davies_bouldin_score": davies_bouldin,
                               "my_score": my_score}
        
        final_score = combined_score(silhouette, davies_bouldin, my_score)

        if final_score > max_score:
            max_score = final_score
            best_configuration = params_str

        if len(results) % 10 == 0:
            try:
                dump(results, results_path)
            except Exception as e:
                print(f"Error saving results: {e}")
            
    try:
        dump(results, results_path)
    except Exception as e:
        print(f"Final save failed: {e}")

    print(f"The best configuration is {best_configuration} with a score of {max_score}")

    return best_configuration, max_score

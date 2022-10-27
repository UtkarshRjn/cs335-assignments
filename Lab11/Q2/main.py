import numpy as np
from utils import visualize, store_labels
import pandas as pd

def get_clusters(X, n_clusters=10):
    '''
    Inputs:
        X: coordinates of the points
        n_clusters: (optional) number of clusters required
    Output:
        labels: The cluster index assigned to each point in X, same length as len(X)
    '''
    #### TODO: ####

    np.random.seed(0)

    centroids = np.zeros((n_clusters, X.shape[1])) 
    for k in range(n_clusters):
        centroid = X[np.random.choice(range(X.shape[0]))]
        centroids[k] = centroid
        
    while True:
        clusters = [[] for _ in range(n_clusters)]
        for i, point in enumerate(X):
            closest_centroid = np.argmin(
                np.sqrt(np.sum((point-centroids)**2, axis=1))
            ) 
            clusters[closest_centroid].append(i)

        previous_centroids = centroids
        centroids = np.zeros((n_clusters, X.shape[1]))
        for i, cluster in enumerate(clusters):
            new_centroid = np.mean(X[cluster], axis=0)
            centroids[i] = new_centroid
        
        if not (centroids - previous_centroids ).any():
            break
    
    
    labels = np.zeros(X.shape[0]) 
    for cluster_idx, cluster in enumerate(clusters):
        for sample_idx in cluster:
            labels[sample_idx] = cluster_idx
    
    return labels

if __name__ == "__main__":
    data = pd.read_csv("mnist_samples.csv").values
    labels = get_clusters(data)
    store_labels(labels)
    visualize(data, labels, alpha=0.2)

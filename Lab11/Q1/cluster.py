from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np
from utils import *

def kernel_pca(X: np.ndarray, kernel: str) -> np.ndarray:
    '''
    Returns projections of the the points along the top two PCA vectors in the high dimensional space.

        Parameters:
                X      : Dataset array of size (n,2)
                kernel : Kernel type. Can take values from ('poly', 'rbf', 'radial')

        Returns:
                X_pca : Projections of the the points along the top two PCA vectors in the high dimensional space of size (n,2)
    '''
    n = X.shape[0]
    K = np.zeros((n,n))

    if kernel == 'poly':
        d = 5
        K = (np.dot(X,X.T) + 1)**d
    elif kernel == 'rbf':
        gamma = 15
        norm = np.square(np.linalg.norm(X, axis=1,keepdims=True)) +  np.square(np.linalg.norm(X.T, axis=0,keepdims=True)) - 2 * np.dot(X,X.T)
        K = np.exp(-norm*gamma)
    elif kernel == 'radial':
        newX = np.zeros(X.shape)
        newX[:,0] = np.sqrt(np.square(X[:,0]) + np.square(X[:,1]))
        newX[:,1] = np.arctan2(X[:,1],X[:,0])
        K = np.dot(newX,newX.T)

    K_centered = K - np.dot(np.ones(K.shape)/n ,K) - np.dot(K,np.ones(K.shape)/n) + np.dot(np.ones(K.shape)/n, np.dot(K,np.ones(K.shape)/n))

    w,v = np.linalg.eigh(K_centered)
    v_pca = v[:,-2:]*np.sqrt(w[-2:])

    temp = np.copy(v_pca[:, 0])
    v_pca[:, 0] = v_pca[:, 1]
    v_pca[:, 1] = temp

    return v_pca

points_2D = load_points_from_json('points_2D.json')
points_2D_pca = kernel_pca(points_2D, 'radial')

kmeans = KMeans(init="k-means++", n_clusters=5, n_init=10)
labels = kmeans.fit_predict(points_2D_pca)

store_labels_to_json(5, labels, 'labels.json')

X = points_2D[:,0]
Y = points_2D[:,1]

colormap = np.array(['r', 'g', 'b', 'y', 'c'])

plt.scatter(X,Y,color = colormap[labels])
plt.xlabel('X')
plt.ylabel('Y')
# plt.savefig('scatter_2D_labeled.jpeg')
import matplotlib.pyplot as plt
import numpy as np

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

if __name__ == "__main__":
    from sklearn.datasets import make_moons, make_circles
    from sklearn.linear_model import LogisticRegression
  
    X_c, y_c = make_circles(n_samples = 500, noise = 0.02, random_state = 517)
    X_m, y_m = make_moons(n_samples = 500, noise = 0.02, random_state = 517)
 
    X_c_pca = kernel_pca(X_c, 'radial')
    X_m_pca = kernel_pca(X_m, 'rbf')

    plt.figure()
    plt.title("Data")
    plt.subplot(1,2,1)
    plt.scatter(X_c[:, 0], X_c[:, 1], c = y_c)
    plt.subplot(1,2,2)
    plt.scatter(X_m[:, 0], X_m[:, 1], c = y_m)
    plt.show()

    plt.figure()
    plt.title("Kernel PCA")
    plt.subplot(1,2,1)
    plt.scatter(X_c_pca[:, 0], X_c_pca[:, 1], c = y_c)
    plt.subplot(1,2,2)
    plt.scatter(X_m_pca[:, 0], X_m_pca[:, 1], c = y_m)
    plt.show()

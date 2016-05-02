# type here missing imports
import numpy as np
from scipy import spatial
from sklearn import cluster

def generate_data(n, centroids, sigma=1., random_state=42):
    """Generate sample data

    Parameters
    ----------
    n : int
        The number of samples
    centroids : array, shape=(k, p)
        The centroids
    sigma : float
        The standard deviation in each class

    Returns
    -------
    X : array, shape=(n, p)
        The samples
    y : array, shape=(n,)
        The labels
    """
    rng = np.random.RandomState(random_state)
    k, p = centroids.shape
    X = np.empty((n, p))
    y = np.empty(n)
    for i in range(k):
        X[i::k] = centroids[i] + sigma * rng.randn(len(X[i::k]), p)
        y[i::k] = i
    order = rng.permutation(n)
    X = X[order]
    y = y[order]
    return X, y


def compute_labels(X, centroids):
    """Compute labels

    Parameters
    ----------
    X : array, shape=(n, p)
        The samples

    Returns
    -------
    labels : array, shape=(n,)
        The labels of each sample
    """
    dist = spatial.distance.cdist(X, centroids, metric='euclidean')
    return dist.argmin(axis=1)


def compute_inertia_centroids(X, labels):
    """Compute inertia and centroids

    Parameters
    ----------
    X : array, shape=(n, p)
        The samples
    labels : array, shape=(n,)
        The labels of each sample

    Returns
    -------
    inertia: float
        The inertia
    centroids: array, shape=(k, p)
        The estimated centroids
    """
    # insert code here
    k_list = np.unique(labels)  
    k = k_list.size
    p = X.shape[1]
    centroids = np.zeros((k,p))
    inertia = 0
    for i,l in enumerate(k_list):
        centroids[i,:] = np.mean(X[labels==l,:],axis=0) 
        inertia = inertia + np.linalg.norm(X[labels==l,:]-centroids[i,:])
    return inertia, centroids


def kmeans(X, n_clusters, n_iter=100, tol=1e-7, random_state=42):
    """K-Means : Estimate position of centroids and labels

    Parameters
    ----------
    X : array, shape=(n, p)
        The samples
    n_clusters : int
        The desired number of clusters
    tol : float
        The tolerance to check convergence

    Returns
    -------
    centroids: array, shape=(k, p)
        The estimated centroids
    labels: array, shape=(n,)
        The estimated labels
    """
    # initialize centroids with random samples
    rng = np.random.RandomState(random_state)
    centroids = X[rng.permutation(len(X))[:n_clusters]]
    labels = compute_labels(X, centroids)
    old_inertia = np.inf
    for k in xrange(n_iter):
        inertia, centroids = compute_inertia_centroids(X, labels)
        if abs(inertia - old_inertia) < tol:
            print 'Converged !'
            break
        old_inertia = inertia
        labels = compute_labels(X, centroids)
        print inertia
    else:
        print 'Dit not converge...'
    # insert code here
    return centroids, labels


if __name__ == '__main__':
    n = 1000
    centroids = np.array([[0., 0.], [2., 2.], [0., 3.]])
    X, y = generate_data(n, centroids)
    n_clusters = 3
    centroids, labels = kmeans(X, n_clusters=n_clusters, random_state=42)
    # plotting code
    import matplotlib.pyplot as plt
    plt.close('all')
    plt.figure()
    plt.plot(X[y == 0, 0], X[y == 0, 1], 'or')
    plt.plot(X[y == 1, 0], X[y == 1, 1], 'ob')
    plt.plot(X[y == 2, 0], X[y == 2, 1], 'og')
    plt.title('Ground truth')

    plt.figure()
    plt.plot(X[labels == 0, 0], X[labels == 0, 1], 'or')
    plt.plot(X[labels == 1, 0], X[labels == 1, 1], 'ob')
    plt.plot(X[labels == 2, 0], X[labels == 2, 1], 'og')
    plt.title('Estimated')

    kmeans = cluster.KMeans(n_clusters=n_clusters,n_init=10)
    kmeans.fit(X)
    labels = kmeans.labels_

    plt.figure()
    plt.plot(X[labels == 0, 0], X[labels == 0, 1], 'or')
    plt.plot(X[labels == 1, 0], X[labels == 1, 1], 'ob')
    plt.plot(X[labels == 2, 0], X[labels == 2, 1], 'og')
    plt.title('Estimated (scikit-learn)')
    plt.show()

import numpy as np


class KMeans: 

    def __init__(self, n_clusters = 5, random_state = 42, max_iter = 100):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.centroids = []
        self.clusters = [] 
        self.X = None

        np.random.seed(random_state)
        
    
    def fit(self, X):
        '''Calculate and update the centroids 
        for the observed dataset.'''

        # randomly select k points as centroids
        self._init_centroids(X)

        prev_centroids, curr_iter = self.centroids, True
        
        while curr_iter < self.max_iter:
            
            # get clusters
            self.clusters = np.asarray([self._get_cluster(x) for x in X])

            # update centroids
            # select as centroid the closest element to
            # the mean point within a cluster
            self._update_centroids(X)

            # there was no update -> we reached the convergence
            if np.array_equiv(self.centroids,prev_centroids):
                break 
            
            # update
            prev_centroids = self.centroids
            curr_iter += 1


    def predict(self, X):
        '''For each vector, get its relative cluster number.'''

        return [self._get_cluster(x) for x in X]

    
    def fit_predict(self, X):
        '''Fit and predict.'''
        self.fit(X)
        return self.predict(X)
    

    def _init_centroids(self, X):
        '''Randomly select 'n_clusters' centroids and 
        initialize all the cluster to 0.'''

        self.centroids = X[np.random.randint(len(X),size=self.n_clusters),:]
        self.clusters = np.zeros(len(X))

    
    def _get_cluster(self, x):
        '''Return the index of the cluster having minimum
        distance with the current point.'''

        distances = [np.sqrt(np.sum((np.asarray(x) - c)**2)) for c in self.centroids]
        return np.argmin(distances)

    
    def _update_centroids(self, X):
        '''For every cluster, update the centroid as the
        average vector between all the ones belonging to
        that cluster.'''

        for idx in range(self.n_clusters):
            self.centroids[idx,:] = np.mean( X[self.clusters == idx,:] , 0)

if __name__ == '__main__':

    from sklearn.datasets import make_blobs

    import matplotlib.pyplot as plt
    import seaborn as sns
    

    X, y = make_blobs(n_samples=1000, centers=3, n_features=2,random_state=0)

    kmeans = KMeans(n_clusters=3, random_state=42)
    y_pred = kmeans.fit_predict(X)

    fig, axs = plt.subplots(1,2,figsize=(10,4))

    sns.scatterplot(x=X[:,0],y=X[:,1],hue=y_pred, ax=axs[0], palette="pastel")
    axs[0].set_title("KMeans")
    axs[0].set_xlabel("x1")
    axs[0].set_ylabel("x2")

    sns.scatterplot(x=X[:,0],y=X[:,1],hue=y, ax=axs[1], palette="pastel")
    axs[1].set_title("True clusters")
    axs[1].set_xlabel("x1")
    axs[1].set_ylabel("x2")

    plt.show()


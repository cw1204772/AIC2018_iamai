import numpy as np
from sklearn.cluster import MiniBatchKMeans, KMeans

class Seed_KMeans(object):
    """
    Seeded k-means
    
    Selecting some samples as seed to do k-means,
    other samples are predicted using distance to existing k-means center
    """
    def __init__(self, n_classes, n_seeds):
        self.k_means = KMeans(n_clusters=n_classes, n_init=10, max_iter=300, 
            n_jobs=-1, verbose=1, init='random')
        #self.k_means = MiniBatchKMeans(n_clusters=n_classes, max_iter=500, n_init=15,
        #     init_size=n_classes, verbose=1, max_no_improvement=250)
        self.n_seeds = n_seeds
    def fit_predict(self, X):
        self.labels_ = np.zeros(X.shape[0])
        seed = np.random.permutation(X.shape[0])[:self.n_seeds]
        exclude_seed = np.ones(X.shape[0])
        exclude_seed[seed] = 0
        exclude_seed = np.where(exclude_seed)[0]
        self.labels_[seed] = self.k_means.fit_predict(X[seed, :])
        self.labels_[exclude_seed] = self.k_means.predict(X[exclude_seed, :])
        return self.labels_

if __name__ == '__main__':
    X = np.random.rand(100000,512)
    clustering = Seed_KMeans(n_classes=10000, n_seeds=30000)
    labels = clustering.fit_predict(X)
    #print(labels)

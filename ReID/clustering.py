import numpy as np
import math
import time
from sklearn.cluster import MiniBatchKMeans, KMeans
class Top_Down(object):

    def __init__(self,n_classes):
        self.subcls = math.ceil(math.sqrt(n_classes))
        self.top_K = KMeans(n_clusters=self.subcls,n_init=10,max_iter=300,n_jobs=-1,verbose=0,init='random')
        self.down_Ks = []
        for i in range(self.subcls):
            self.down_Ks.append(KMeans(n_clusters=self.subcls,n_init=10,max_iter=100,n_jobs=-1,verbose=1,init='k-means++'))
        print('%d top classes and %d classes for each top classes'%(self.subcls,self.subcls))
    def fit_predict(self,X):
        # output labels with input order
        self.labels = np.zeros((X.shape[0],))
        # Top K-means
        top_cls = self.top_K.fit_predict(X)
        # generate the index with Top-k cls
        n_cls = []
        for i in range(self.subcls):
            n_cls.append(np.count_nonzero(top_cls == i))
        cls_idx = np.argsort(top_cls)
        
        start_idx = 0
        end_idx = 0
        offset = 0
        # do subcluster
        for i in range(self.subcls):
            end_idx += n_cls[i]
            X_idx = cls_idx[start_idx:end_idx]
            subcls_label = self.down_Ks[i].fit_predict(X[X_idx])
            self.labels[X_idx] = subcls_label + offset            
            start_idx = end_idx
            offset += self.subcls
        return self.labels
    
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
    X = np.random.rand(120000,512)
    # clustering = Seed_KMeans(n_classes=10000, n_seeds=30000)
    clustering = Top_Down(n_classes=10000)
    t0 = time.time() 
    labels = clustering.fit_predict(X)
    print('%.2f min'%((time.time()-t0)/60))
    # print(labels)
    #print(labels)

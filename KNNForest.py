from sklearn.model_selection import KFold

from ID3 import Tree, load, test
from random import sample
import numpy as np
import collections
import operator

class KNN:
    def __init__(self, K, N, p):  ##N- number of learned trees, K- number of best trees to decide classification, p- percent of attributes for each tree
        self.p = p
        self.K = K if K <= N else N
        self.Trees = [Tree() for i in range(N)]  #Array of N Trees
        self.Centroids = [0 for i in range(N)]  ##Array of Arrays, primary index for tree, secondary for feature

    def fit(self, attributes, classifications, M=0):
        elements = [i for i in range(len(classifications))]
        n = len(elements)
        for i in range(len(self.Trees)):
            samples = sample(elements, int(self.p * n))
            attr = [attributes[i] for i in samples]
            classfs = [classifications[i] for i in samples]
            self.Trees[i].fit(attr, classfs, M)
            self.Centroids[i] = centroid(attr)

    def predict(self, obj):  #obj-set of attributes describing an object
        dist = [(self.distance(obj, self.Centroids[i]), i) for i in range(len(self.Centroids))]
        dist.sort()
        decision = {}
        for vec in dist[:self.K]:
            i = vec[1]
            c = self.Trees[i].predict(obj)[0]
            if c in decision.keys():
                decision[c] += 1
            else:
                decision[c] = 1

        return [max(decision.items(), key=operator.itemgetter(1))[0]]

    @staticmethod
    def distance(v1, v2):  #vectors\lists as input
        return np.linalg.norm(np.array(v1) - np.array(v2))

def centroid(attributes):
    if attributes is None:
        return 0
    num_samples = len(attributes)
    num_features = len(attributes[0])  ##number of features
    cent = [0 for i in range(num_features)]
    for i in range(num_features):
        cent[i] = sum(attributes[k][i] for k in range(num_samples)) / num_samples

    return cent

def cross_validation(attrs, clas, K, N, p, m=0):
    kf = KFold(n_splits=2)
    elements = [i for i in range(len(attrs))]
    res = 0
    # create the split of elements
    for train_index, test_index in kf.split(elements):
        train_index = list(train_index)
        test_index = list(test_index)
        tree = KNN(K, N, p)
        # trains the tree on given elements to train
        a = [attrs[i] for i in train_index]
        c = [clas[i] for i in train_index]
        tree.fit(a, c, m)
        # tests the tree on given elements to test
        res = max(test(tree, [attrs[i] for i in test_index], [clas[i] for i in test_index]), res)
    return res


def calibration(attributes, classifications):
    n = len(classifications)
    if n == 0:
        return 0, 0, 0
    best = [0, 0, 0, 0]
    NArr = [7, 8, 9, 10, 11, 12]
    pArr = [i for i in np.linspace(0.3, 0.7, 5)]
    ##MArr = [i for i in range(1, int(n/10) + 1, int(n/40) + 1)]
    for N in NArr:
        KArr = [i for i in range(int(0.5*N), int(0.8*N) + 1, int(0.1 * N) + 1)]
        for K in KArr:
            for p in pArr:
                acc = cross_validation(attributes, classifications, K, N, p)  #could test prunning as well
                if acc >= best[0]:
                    best = [acc, K, N, p]
    return tuple(best[1:])


a, c = load('train.csv')

#best = calibration(a, c)


forest = KNN(6, 8, 0.7)

forest.fit(a, c)
a, c = load('test.csv')

print(test(forest, a, c))
#print(f"K:{best[0]},N:{best[1]},p:{best[2]}")



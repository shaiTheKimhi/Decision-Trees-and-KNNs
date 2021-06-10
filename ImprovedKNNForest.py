from sklearn.model_selection import KFold

from ID3 import Tree, load, test
from random import choices
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
            samples = choices(elements, k=int(self.p * n))
            attr = [attributes[i] for i in samples]
            classfs = [classifications[i] for i in samples]
            self.Trees[i].fit(attr, classfs, M)
            self.Centroids[i] = centroid(attr)

    def predict(self, obj):  #obj-set of attributes describing an object
        dist = [(self.distance(obj, self.Centroids[i]), i) for i in range(len(self.Centroids))]
        dist.sort()
        decision = self.decisions(self.Trees, dist, obj, self.K)

        if 'B' in decision.keys() and 'M' in decision.keys():
            pos = decision['M']
            neg = decision['B']
            total = pos + neg
            if abs(pos - neg) / total <= (self.p / 2):
                d2 = self.decisions(self.Trees, dist, obj, len(self.Trees))
                ##return [max(d2.items(), key=operator.itemgetter(1))[0]]
                d3 = self.decisions(self.Trees, dist, obj, int(self.K / 2))
                p2 = d2['M'] if 'M' in d2.keys() else 0
                n2 = d2['B'] if 'B' in d2.keys() else 0
                t2 = n2 + p2
                p3 = d3['M'] if 'M' in d3.keys() else 0
                n3 = d3['B'] if 'B' in d3.keys() else 0
                t3 = p3 + n3
                if abs(p2 - n2) / t2 > (self.p / 2):
                    return [max(d2.items(), key=operator.itemgetter(1))[0]]
                elif abs(p3 - n3) / t3 > (self.p / 2):
                    return [max(d3.items(), key=operator.itemgetter(1))[0]]
                else:
                    d1 = [max(decision.items(), key=operator.itemgetter(1))[0]]
                    d2 = [max(d2.items(), key=operator.itemgetter(1))[0]]
                    d3 = [max(d3.items(), key=operator.itemgetter(1))[0]]
                    if [d1, d2, d3].count('B') >= 2:
                        return ['B']
                    else:
                        return ['M']
            else:
                return [max(decision.items(), key=operator.itemgetter(1))[0]]
        else:
            return [max(decision.items(), key=operator.itemgetter(1))[0]]

    @staticmethod
    def decisions(trees, dist, obj, k):  ##-distances of trees centroids to features of object, ordered by distance
        decision = {}
        for vec in dist[:k]:
            i = vec[1]
            c = trees[i].predict(obj)[0]
            if c in decision.keys():
                decision[c] += 1 / (vec[0] ** 2)
            else:
                decision[c] = 1 / (vec[0] ** 2)
        return decision

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
    NArr = [10, 30, 60]
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
#forest = KNN(best[0], best[1], best[2])




forest = KNN(21, 30, 0.5)



forest.fit(a, c)
a, c = load('test.csv')

print(test(forest, a, c))



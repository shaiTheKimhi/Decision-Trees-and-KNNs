import math
# import numpy as np
from sklearn.model_selection import KFold
from matplotlib import pyplot as plt
import collections
import operator


class Tree:
    def __init__(self):
        self.is_leaf = False
        self.left = None
        self.right = None
        self.val = 0
        self.attr_index = -1

        self.Attr = None
        self.Classf = None

        self.M = 0

    def fit(self, attributes, classifications, m=0):
        if len(attributes) != len(classifications):
            print("Attributes and Classifications not in same size")
            raise Exception("Attributes and Classifications not in same size")

        self.Attr = tuple(attributes)
        self.Classf = tuple(classifications)
        self.M = m

        elements = [i for i in range(len(self.Attr))]
        self.train(elements)

    def train(self, elements, val=0):
        if len(elements) == 0:
            raise Exception("empty elements to train")
        # if all elements have same classification
        if self.same(elements):
            self.is_leaf = True
            self.val = self.Classf[elements[0]]
        elif len(elements) < self.M:
            self.is_leaf = True
            self.val = val  ##max(count.items(), key=operator.itemgetter(1))[0]
        else:
            attr, split = self.choose_best(elements)
            if attr is None:
                self.is_leaf = True
                self.val = val
                return

            left_elements = [e for e in elements if self.Attr[e][attr] >= split]
            right_elements = [e for e in elements if self.Attr[e][attr] < split]
            count = dict(collections.Counter([self.Classf[i] for i in elements]))

            self.val = split
            self.attr_index = attr
            self.train_side(left_elements, right_elements, max(count.items(), key=operator.itemgetter(1))[0])

    # create side sub tree
    def train_side(self, left_elements, right_elements, val):
        self.left = Tree()
        self.left.Attr = self.Attr
        self.left.Classf = self.Classf
        self.left.M = self.M
        self.left.train(left_elements, val)

        self.right = Tree()
        self.right.Attr = self.Attr
        self.right.Classf = self.Classf
        self.right.M = self.M
        self.right.train(right_elements, val)

    # chooses best attribute and splitter for split
    def choose_best(self, elements):
        num_attrs = len(self.Attr[elements[0]])
        n_elements = len(elements)

        max_attr = None
        max_split = None
        max_ig = -float('inf')
        en = self.entropy(elements)
        # searches for best attributes
        for i in range(num_attrs):
            # searches for best splitter
            vals = list(set([self.Attr[e][i] for e in elements]))
            vals.sort()

            split = [(vals[i] + vals[i + 1]) / 2 for i in range(len(vals) - 1)]
            best_ig = -float('inf')
            best_split = None
            for s in split:
                left = [e for e in elements if self.Attr[e][i] >= s]
                right = [e for e in elements if self.Attr[e][i] < s]
                ig = en - self.entropy(left) * len(left) / n_elements - self.entropy(right) * len(right) / n_elements
                if ig >= best_ig:
                    best_ig = ig
                    best_split = s
            if best_ig >= max_ig and best_ig > 0:
                max_ig = best_ig
                max_attr = i
                max_split = best_split
        return max_attr, max_split

    ##HELPER FUNCTIONS FOR TRAIN

    # calculates entropy for the group of elements
    def entropy(self, elements):
        # checks number of elements for each classification (used for probability for classification)
        prob = dict(collections.Counter([self.Classf[i] for i in elements]))
        num_e = len(elements)
        # if num_e == 0:
        #    return -float('inf')
        # diff = 1 / num_e
        # for e in elements:
        #    c = self.Classf[e]
        #    if c in prob.keys():
        #        prob[c] += 1
        #    else:
        #        prob[c] = 1

        return -sum(prob[k] / num_e * math.log(prob[k] / num_e, 2) for k in prob.keys())

    # checks if elements are the same
    def same(self, elements):
        classfs = [self.Classf[e] for e in elements]
        values = set(classfs)
        return len(values) == 1

    # PREDICT AND TEST

    def predict(self, obj):  # object- Attributes describing an object
        if self.is_leaf:
            return [self.val]
        else:
            if obj[0][self.attr_index] >= self.val:
                return self.left.predict(obj)
            else:
                return self.right.predict(obj)


def test(tree, attributes, classifications):
    s = 0
    for i in range(len(attributes)):
        if tree.predict([attributes[i]])[0] == classifications[i]:
            s += 1
    return s / len(attributes)


def load(file_name):
    file = open('data/' + file_name, 'r')
    lines = file.readlines()[1:]
    file.close()
    parts = [line.replace('\n', '').split(',') for line in lines]
    c = [p[0] for p in parts]
    attrs = [p[1:] for p in parts]
    attrs = [[float(i) for i in p] for p in attrs]
    return attrs, c


# calculates the 5 cross validation of ID3 for given m, with given attributes and classifications
def validation(m, attrs, clas):  # attrs- attributes array, clas- classifications array
    kf = KFold(n_splits=5, shuffle=True, random_state=318605516)
    elements = [i for i in range(len(attrs))]
    res = 0
    # create the split of elements
    for train_index, test_index in kf.split(elements):
        train_index = list(train_index)
        test_index = list(test_index)
        tree = Tree()
        # trains the tree on given elements to train
        a = [attrs[i] for i in train_index]
        c = [clas[i] for i in train_index]
        tree.fit(a, c, m)
        # tests the tree on given elements to test
        res += test(tree, [attrs[i] for i in test_index], [clas[i] for i in test_index])
    return res / kf.get_n_splits(elements)


# shows the results for k cross validation on the ID3 tree for different M values and given attributes and classifications
# To Run- call get a,c = load('train.csv'), then run experiment(a, c)
def expreiment(attributes, classifications):
    accuracies = []
    m_values = [i for i in range(4, 13, 2)]  ## 4,6,8,10,12- 5 different M values
    for m in m_values:
        res = validation(m, attributes, classifications)
        # print(f"M:{m} -> accuracy:{res}")
        accuracies.append(res)

    fig, ax1 = plt.subplots()
    p1, = ax1.plot(m_values, accuracies, '-b', label='Tree Accuracy')
    plt.title('Accuracy for m prunning')
    plt.show()


if __name__ == '__main__':
    attrs, c = load('train.csv')
    # print(validation(9, attrs, c))
    #expreiment(attrs, c)
    t = Tree()
    t.fit(attrs, c)

    attrs, c = load('test.csv')

    print(test(t, attrs, c))

import math
import numpy as np
from sklearn.model_selection import KFold

from ID3 import Tree
class CostTree(Tree):
    def train(self, elements, val=0):
        if len(elements) == 0:
            raise Exception("empty elements to train")
        # if all elements have same classification
        if self.same(elements):
            self.is_leaf = True
            self.val = self.Classf[elements[0]]
        elif len(elements) < self.M:
            count = dict(collections.Counter([self.Classf[i] for i in elements]))
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
            count = sum(-0.1 if self.Classf[i] == 'B' else 1 for i in elements)  #negative sums healthy multiplied by 0.1(fp weight), positive for deceased (fn weight is 1)

            self.val = split
            self.attr_index = attr
            self.train_side(left_elements, right_elements, 'B' if count <= 0 else 'M')  #if more points to healthy, then 'B' else 'M'

    def entropy(self, elements):
        prob = {'B': sum(1 for i in elements if self.Classf[i] == 'B'), 'M': sum(1 for i in elements if self.Classf[i] == 'M')}
        prob['B'] *= 9
        prob['M'] *= 1
        num_e = len(elements)

        return -sum(prob[k] / num_e * math.log(prob[k] / num_e, 2) for k in prob.keys() if prob[k] > 0)


##TODO: create cross validation function for loss domain
##TODO: test cross validation for different M values and use the optimal one, (validation will be out of Tree Class)

def load(file_name):
    file = open('data/' + file_name, 'r')
    lines = file.readlines()[1:]
    file.close()
    parts = [line.replace('\n', '').split(',') for line in lines]
    c = [p[0] for p in parts]
    attrs = [p[1:] for p in parts]
    attrs = [[float(i) for i in p] for p in attrs]
    return attrs, c


def test_cost(tree, attributes, classifications):
    fp = 0
    fn = 0
    for i in range(len(attributes)):
        if tree.predict([attributes[i]])[0] != classifications[i]:
            if classifications[i] == 'B':
                fp += 1
            else:
                fn += 1
    return (fn + 0.1 * fp) / len(attributes)


# calculates the 5 cross validation of ID3 for given m, with given attributes and classifications
def validation(m, attrs, clas):  # attrs- attributes array, clas- classifications array
    kf = KFold(n_splits=5, shuffle=True, random_state=318605516)
    elements = [i for i in range(len(attrs))]
    res = 0
    # create the split of elements
    for train_index, test_index in kf.split(elements):
        train_index = list(train_index)
        test_index = list(test_index)
        tree = CostTree()
        # trains the tree on given elements to train
        a = [attrs[i] for i in train_index]
        c = [clas[i] for i in train_index]
        tree.fit(a, c, m)
        # tests the tree on given elements to test
        res += test_cost(tree, [attrs[i] for i in test_index], [clas[i] for i in test_index])
    return res / kf.get_n_splits(elements)


def expreiment(attributes, classifications):
    accuracies = []
    m_values =[1,2,5,8,10]  ##[i for i in range(4, 13, 2)]  ## 4,6,8,10,12- 5 different M values
    for m in m_values:
        res = validation(m, attributes, classifications)
        # print(f"M:{m} -> accuracy:{res}")
        accuracies.append(res)

    return m_values[np.argmax(np.array(accuracies))]


if __name__ == "__main__":
    a, c = load('train.csv')
    m = expreiment(a, c)
    tree = CostTree()
    tree.fit(a, c, m)
    a, c = load('test.csv')
    print(test_cost(tree, a, c))

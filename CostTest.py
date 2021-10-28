from ID3 import load, Tree
from CostSensitiveID3 import CostTree

#gets test result with accordance to cost
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


if __name__ == '__main__':
    t = Tree()
    attr, c = load('train.csv')
    t.fit(attr, c)

    ct = CostTree()
    ct.fit(attr, c)

    atest, ctest = load('test.csv')
    old = test_cost(t, atest, ctest)
    print(f'Basic:{old}')
    newcost = test_cost(ct, atest, ctest)
    print(f'Improved:{newcost}')
    print(f'improved by {old/newcost} times')
    '''
    for m in range(2, 15):
        ct = CostTree()
        ct.fit(attr, c, m)
        print(f'M:{m}-> loss:{test_cost(ct, atest, ctest)}')
    '''

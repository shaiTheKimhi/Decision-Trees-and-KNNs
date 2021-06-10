from os import system, chdir

chdir('..')

N = 10

for i in range(N):
    system(f'python KNNForest.py > Test(not_send)\KNN{i}.txt')
    system(f'python ImprovedKNNForest.py > Test(not_send)\ImpKNN{i}.txt')
   
    
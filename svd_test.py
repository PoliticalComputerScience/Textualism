import numpy as np
from numpy.linalg import svd

a = np.array([[1,1,1],[2,2,2],[3,3,3], [4,4,4]])
for i in range(2):
    print(a)

    print('-------')

    U, s, V = svd(a)

    print('U: ')
    print(U)
    print('------')

    print('s: ')
    print(s)
    print('------')

    print('V: ')
    print(V)
    a = a.T
    print(' ')
    print(' ')
    print('--------------------- TRANSPOSED VERSION -----------------')

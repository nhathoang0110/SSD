import numpy as np

X= np.array([4,3])

#L0 norm
# Số lượng phần tử khác 0

l0norm= np.linalg.norm(X, ord=0)
print(l0norm)

#L1 norm  Mahattan

l1norm= np.linalg.norm(X, ord=1)
print(l1norm)

#L2 norm  Euclid

l2norm= np.linalg.norm(X, ord=2)
print(l2norm)
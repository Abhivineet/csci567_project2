import numpy as np

X = [[1,2,3],[2,3,4],[3,4,5],[4,5,6]]
X = np.asarray(X)
ref = X
p = 3
for j in range(2, p + 1):
    for i in range(len(ref[0])):
        # np.insert(X[j-1][i],ref[i] ** j)
        X = np.insert(X, len(X[0]), ref[:,i]**j, axis =1)
print(X)
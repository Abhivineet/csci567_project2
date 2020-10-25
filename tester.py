import numpy as np

X = [[1,2,3],[2,3,4],[3,4,5],[4,5,6]]
X = np.asarray(X)
p = 3
ref = np.asarray(X)
X = [X]
for i in range(p - 1):
    X.append(np.zeros(ref.shape))
for i in range(len(ref)):
    for j in range(2, p + 1):
        X[j - 1][i] = ref[i] ** j
print(p)
print(ref.shape)
print(np.shape(X))


print(X)
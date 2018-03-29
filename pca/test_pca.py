import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.decomposition import PCA
from pca import PCA as SIMPLE_PCA

iris = datasets.load_iris()

X = iris.data
y = iris.target
target_names = iris.target_names



pca = PCA(n_components=2)
X_r = pca.fit_transform(X)

plt.figure()
for c, i, target_name in zip("rgb", [0, 1, 2], target_names):
    plt.scatter(X_r[y == i, 0], X_r[y == i, 1], c=c, label=target_name)
plt.legend()
plt.title('sklearn\'s PCA of IRIS dataset')




simple_pca = SIMPLE_PCA(n_components=2)
simple_pca.fit(X)
X_r2 = simple_pca.transform(X)

plt.figure()
for c, i, target_name in zip("rgb", [0, 1, 2], target_names):
    plt.scatter(X_r2[y == i, 0], X_r2[y == i, 1], c=c, label=target_name)
plt.legend()
plt.title('simple_pca\'s PCA of IRIS dataset')


plt.show()
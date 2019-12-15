import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np
from sklearn.cluster import KMeans

style.use('ggplot')


data = np.array([[1, 2], 
            [1.5, 1.8],
            [5, 8],
            [8, 8],
            [1, 0.6],
            [9, 11]])

plt.scatter(data[:, 0], data[:, 1], s=50)

clf = KMeans(n_clusters=2)
clf.fit(data)

centroids = clf.cluster_centers_
labels = clf.labels_

colors = 10*['g', 'r', 'c', 'b', 'k']

for i in range(len(data)):
    plt.plot(data[i][0], data[i][1], colors[labels[i]], markersize = 25)

plt.scatter(centroids[:, 0], centroids[:, 1], marker='x', s=150, linewidths=5)
plt.show()

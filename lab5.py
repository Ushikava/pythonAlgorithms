import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics.pairwise import pairwise_distances_argmin
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
import random
import math


data = pd.read_csv(r'iris.data', header=None)
no_labeled_data = data.drop(4, axis=1)
print(no_labeled_data)

k_means = KMeans(init='k-means++', n_clusters=3, n_init=15)
k_means.fit(no_labeled_data)

k_means_cluster_centers = k_means.cluster_centers_
k_means_labels = pairwise_distances_argmin(no_labeled_data, k_means_cluster_centers)

f, ax = plt.subplots(1, 3)
colors = ['#4EACC5', '#FF9C34', '#4E9A06']
print(ax)
for i in range(3):
    my_members = k_means_labels == i
    cluster_center = k_means_cluster_centers[i]
    for j in range(3):
        ax[j].plot(no_labeled_data[my_members][j], no_labeled_data[my_members][j + 1], 'w', markerfacecolor=colors[i], marker='o', markersize=4)
        ax[j].plot(cluster_center[j], cluster_center[j + 1], 'o', markerfacecolor=colors[i], markeredgecolor='k', markersize=8)
plt.show()

pca = PCA(n_components=2)
reduced_data = pca.fit_transform(no_labeled_data)
kmeans = KMeans(init='k-means++', n_clusters=3, n_init=15)
kmeans.fit(reduced_data)
h = 0.02
x_min, x_max = reduced_data[:, 0].min() - 1, reduced_data[:, 0].max() + 1
y_min, y_max = reduced_data[:, 1].min() - 1, reduced_data[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plt.figure(1)
plt.clf()
plt.imshow(Z, interpolation="nearest", extent=(xx.min(), xx.max(), yy.min(), yy.max()), cmap=plt.cm.Paired,
           aspect="auto", origin="lower")
plt.plot(reduced_data[:, 0], reduced_data[:, 1], "k.", markersize=2)
centroids = kmeans.cluster_centers_
plt.scatter(centroids[:, 0], centroids[:, 1], marker="x", s=169, linewidths=3, color="w", zorder=10)
plt.title("K-means\n"
          "Centroids are marked by white cross")
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.xticks(())
plt.yticks(())
plt.show()

k_means = KMeans(init='random', n_clusters=3, n_init=15)
k_means.fit(no_labeled_data)
k_means_cluster_centers = k_means.cluster_centers_
k_means_labels = pairwise_distances_argmin(no_labeled_data, k_means_cluster_centers)
f, ax = plt.subplots(1, 3)
colors = ['#4EACC5', '#FF9C34', '#4E9A06']
print(ax)
for i in range(3):
     my_members = k_means_labels == i
     cluster_center = k_means_cluster_centers[i]
     for j in range(3):
        ax[j].plot(no_labeled_data[my_members][j], no_labeled_data[my_members][j + 1], 'w', markerfacecolor=colors[i], marker='o', markersize=4)
        ax[j].plot(cluster_center[j], cluster_center[j + 1], 'o', markerfacecolor=colors[i], markeredgecolor='k', markersize=8)
plt.show()

k_means = KMeans(init=np.array([[1.2, 1.5, 2.5, 1.7], [2.5, 4.2, 1.3, 1.8], [1.1, 3.5, 4.1, 3.1]]), n_clusters=3, n_init=1)
k_means.fit(no_labeled_data)
k_means_cluster_centers = k_means.cluster_centers_
k_means_labels = pairwise_distances_argmin(no_labeled_data, k_means_cluster_centers)
f, ax = plt.subplots(1, 3)
colors = ['#4EACC5', '#FF9C34', '#4E9A06']
print(ax)
for i in range(3):
    my_members = k_means_labels == i
    cluster_center = k_means_cluster_centers[i]
    for j in range(3):
        ax[j].plot(no_labeled_data[my_members][j], no_labeled_data[my_members][j + 1], 'w', markerfacecolor=colors[i], marker='o', markersize=4)
        ax[j].plot(cluster_center[j], cluster_center[j + 1], 'o', markerfacecolor=colors[i], markeredgecolor='k', markersize=8)
plt.show()

wcss = []
for i in range(1, 15):
    kmean = KMeans(n_clusters=i, init="k-means++")
    kmean.fit_predict(no_labeled_data)
    wcss.append(kmean.inertia_)
plt.plot(range(1, 15), wcss)
plt.title('Метод локтя')
plt.xlabel("Кластеры")
plt.ylabel("WCSS")
plt.show()

data = no_labeled_data
n_clusters = 3
np.random.seed(1)
kmeans = KMeans(init='k-means++', n_clusters=3, n_init=15)
kmeans.fit(data)
mbk = MiniBatchKMeans(init='k-means++', n_clusters=3, batch_size=100, n_init=15)
mbk.fit(data)
fig = plt.figure(figsize=(25, 8))
k_means_cluster_centers = kmeans.cluster_centers_
order = pairwise_distances_argmin(kmeans.cluster_centers_, mbk.cluster_centers_)
mbk_means_cluster_centers = mbk.cluster_centers_[order]
k_means_labels = pairwise_distances_argmin(data, k_means_cluster_centers)
mbk_means_labels = pairwise_distances_argmin(data, mbk_means_cluster_centers)
print(mbk_means_labels)

def ax_fill(ax, data, labels, centers, title, plot_centers=True, n_clusters=3):
    for k, col in zip(range(n_clusters), colors):
        my_members = labels == k
        ax.plot(data[my_members][0], data[my_members][1], 'o', markerfacecolor=col, markersize=6, markeredgecolor=col)
        if plot_centers:
            cluster_center = centers[k]
            ax.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col, markeredgecolor='k', markersize=9)
    ax.set_title(title)

ax = fig.add_subplot(1, 3, 1)
ax_fill(ax,data, k_means_labels, k_means_cluster_centers, 'KMeans')
ax = fig.add_subplot(1, 3, 2)
ax_fill(ax, data, mbk_means_labels, mbk_means_cluster_centers, 'MiniBatchKMeans')
different = (mbk_means_labels == 4)
ax = fig.add_subplot(1, 3, 3)
for k in range(n_clusters):
    different += ((k_means_labels == k) != (mbk_means_labels == k))
identic = np.logical_not(different)
ax.plot(data[identic][0], data[identic][1], 'o', markerfacecolor='#20B2AA', markersize=6, markeredgecolor='#20B2AA')
ax.plot(data[different][0], data[different][1], 'o', markerfacecolor='#FF0000', markersize=6, markeredgecolor='#FF0000')
ax.set_title('Difference')
plt.show()

colors = ['#4EACC5', '#FF9C34', '#4E9A06', '#FF00FF', '#FFFF00']

hier = AgglomerativeClustering(n_clusters=5, linkage='average')
hier = hier.fit(no_labeled_data)
hier_labels = hier.labels_
print(hier_labels)
f, ax = plt.subplots(1, 3)
for i in range(5):
    my_members = hier_labels == i
    for j in range(3):
        ax[j].plot(no_labeled_data[my_members][j], no_labeled_data[my_members][j+1], 'w', markerfacecolor=colors[i], marker='o', markersize=4)

plt.show()

linked = linkage(no_labeled_data)
dendrogram(linked, truncate_mode='level', p=6)
plt.show()

data1 = np.zeros([250, 2])
for i in range(250):
    r = random.uniform(1, 3)
    a = random.uniform(0, 2 * math.pi)
    data1[i, 0] = r * math.sin(a)
    data1[i, 1] = r * math.cos(a)

data2 = np.zeros([500,2])
for i in range(500):
    r = random.uniform(5, 9)
    a = random.uniform(0, 2 * math.pi)
    data2[i, 0] = r * math.sin(a)
    data2[i, 1] = r * math.cos(a)

data = np.vstack((data1, data2))

print(data)

hier = AgglomerativeClustering(n_clusters=2, linkage='ward')
hier = hier.fit(data)
hier_labels = hier.labels_
print(hier_labels)

my_members = hier_labels == 0
plt.plot(data[my_members, 0], data[my_members, 1], 'w', marker='o', markersize=4, color='red',linestyle='None')
my_members = hier_labels == 1
plt.plot(data[my_members, 0], data[my_members, 1], 'w', marker='o', markersize=4, color='blue',linestyle='None')
plt.show()

hier = AgglomerativeClustering(n_clusters=2, linkage='complete')
hier = hier.fit(data)
hier_labels = hier.labels_
print(hier_labels)

my_members = hier_labels == 0
plt.plot(data[my_members, 0], data[my_members, 1], 'w', marker='o', markersize=4, color='red',linestyle='None')
my_members = hier_labels == 1
plt.plot(data[my_members, 0], data[my_members, 1], 'w', marker='o', markersize=4, color='blue',linestyle='None')
plt.show()

hier = AgglomerativeClustering(n_clusters=2, linkage='average')
hier = hier.fit(data)
hier_labels = hier.labels_
print(hier_labels)

my_members = hier_labels == 0
plt.plot(data[my_members, 0], data[my_members, 1], 'w', marker='o', markersize=4, color='red',linestyle='None')
my_members = hier_labels == 1
plt.plot(data[my_members, 0], data[my_members, 1], 'w', marker='o', markersize=4, color='blue',linestyle='None')
plt.show()

hier = AgglomerativeClustering(n_clusters=2, linkage='single')
hier = hier.fit(data)
hier_labels = hier.labels_
print(hier_labels)

my_members = hier_labels == 0
plt.plot(data[my_members, 0], data[my_members, 1], 'w', marker='o', markersize=4, color='red',linestyle='None')
my_members = hier_labels == 1
plt.plot(data[my_members, 0], data[my_members, 1], 'w', marker='o', markersize=4, color='blue',linestyle='None')
plt.show()
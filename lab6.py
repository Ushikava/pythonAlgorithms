import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.cluster import OPTICS
from sklearn.cluster import cluster_optics_dbscan
from sklearn import preprocessing
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import matplotlib.gridspec as gridspec

data = pd.read_csv('CC GENERAL.csv').iloc[:, 1:].dropna()

k_means = KMeans(init='k-means++', n_clusters=3, n_init=15)
no_labeled_data = data[1:]

data = np.array(data, dtype='float')
min_max_scaler = preprocessing.StandardScaler()
scaled_data = min_max_scaler.fit_transform(data)
print(scaled_data)

clustering = DBSCAN().fit(scaled_data)
print("Метки кластеров: ", set(clustering.labels_))
print("Количество кластеров: ", len(set(clustering.labels_)) - 1)
print("Процент неудач: ", list(clustering.labels_).count(-1) / len(list(clustering.labels_)))

eps_range = np.arange(0.01, 5.0, 0.1)
clust_n = []
clust_failure = []

for eps in eps_range:
    clustering = DBSCAN(eps=eps).fit(scaled_data)
    clust_n.append(len(set(clustering.labels_)) - 1)
    clust_failure.append(100 * list(clustering.labels_).count(-1) / len(list(clustering.labels_)))
plt.xlabel("eps")
plt.plot(eps_range, clust_n, label="Clusters")
plt.plot(eps_range, clust_failure, label="Unclassified data")
plt.legend()
plt.show()

samples = np.arange(5, 100, 1)
clust_n = []
clust_failure = []

for sample in samples:
    clustering = DBSCAN(min_samples=sample).fit(scaled_data)
    clust_n.append(len(set(clustering.labels_)) - 1)
    clust_failure.append(100 * list(clustering.labels_).count(-1) / len(list(clustering.labels_)))
plt.xlabel("min samples")
plt.plot(samples, clust_n, label="Clusters")
plt.plot(samples, clust_failure, label="Unclassified data")
plt.legend()
plt.show()

clustering = DBSCAN(eps=2, min_samples=3).fit(scaled_data)
print("Кластеры: ", (len(set(clustering.labels_)) - 1))
print("Процент некластеризованных данных: ", (100 * list(clustering.labels_).count(-1) / len(list(clustering.labels_))))

samples = np.arange(1, 4, 1)
eps_range = np.arange(1.5, 2.5, 0.1)
info = {}
for sample in samples:
    for eps in eps_range:
        clustering = DBSCAN(eps=eps ,min_samples=sample, n_jobs=-1).fit(scaled_data)
        labels_set = set(clustering.labels_)
        info[(sample, eps)]= [len(labels_set) - 1, list(clustering.labels_).count(-1) / len(list(clustering.labels_))]
for key, value in info.items():
    if value[0] >= 5 and value[0] <= 7 and value[1] <= 0.12:
         print(key, value)
clustering = DBSCAN(eps=2, min_samples=3, n_jobs=-1).fit(scaled_data)
pca = PCA(n_components=2)
reduced_data = pca.fit_transform(data)
pca.explained_variance_ratio_
core_samples_mask = np.zeros_like(clustering.labels_, dtype=bool)
core_samples_mask[clustering.core_sample_indices_] = True
labels = clustering.labels_
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
n_noise_ = list(labels).count(-1)
unique_labels = set(labels)
unique_labels.remove(-1)
unique_labels = [-1, *list(unique_labels)]
colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]
plt.figure(figsize=(12, 8))
for k, col in zip(unique_labels, colors):
    if k == -1:
        col = [0, 0, 0, 1]
    class_member_mask = (labels == k)
    xy = reduced_data[class_member_mask & core_samples_mask]
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col), markeredgecolor='k', markersize=9)
    xy = reduced_data[class_member_mask & ~core_samples_mask]
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col), markeredgecolor='k', markersize=5)
plt.show()

clustering = OPTICS(max_eps=2, min_samples=3, cluster_method='dbscan').fit(scaled_data)

print("Метки кластеров: ", set(clustering.labels_))
print("Количество кластеров: ", len(set(clustering.labels_)) - 1)
print("Процент неудач: ", 10 * list(clustering.labels_).count(-1) / len(list(clustering.labels_)))

pca_data = PCA(n_components=2).fit_transform(data)

def plot_optics(clust):
    labels_050 = cluster_optics_dbscan(reachability=clust.reachability_, core_distances=clust.core_distances_, ordering=clust.ordering_, eps=0.5)
    labels_200 = cluster_optics_dbscan(reachability=clust.reachability_, core_distances=clust.core_distances_, ordering=clust.ordering_, eps=2)
    space = np.arange(len(scaled_data))
    reachability = clust.reachability_[clust.ordering_]
    labels = clust.labels_[clust.ordering_]
    plt.figure(figsize=(10, 7))
    G = gridspec.GridSpec(2, 3)
    ax1 = plt.subplot(G[0, :])
    ax2 = plt.subplot(G[1, 0])
    ax3 = plt.subplot(G[1, 1])
    ax4 = plt.subplot(G[1, 2])

    # Reachability plot
    colors = ['g.', 'r.', 'b.', 'y.', 'c.']
    for klass, color in zip(range(0, 5), colors):
        Xk = space[labels == klass]
        Rk = reachability[labels == klass]
        ax1.plot(Xk, Rk, color, alpha=0.3)
    ax1.plot(space[labels == -1], reachability[labels == -1], 'k.', alpha=0.3)
    ax1.plot(space, np.full_like(space, 2., dtype=float), 'k-', alpha=0.5)
    ax1.plot(space, np.full_like(space, 0.5, dtype=float), 'k-.', alpha=0.5)
    ax1.set_ylabel('Reachability (epsilon distance)')
    ax1.set_title('Reachability Plot')

    # OPTICS
    colors = ['g.', 'r.', 'b.', 'y.', 'c.']
    ax2.plot(pca_data[clust.labels_ == -1, 0], pca_data[clust.labels_ == -1, 1], 'k+', alpha=0.1)
    for klass, color in zip(range(0, 5), colors):
        Xk = pca_data[clust.labels_ == klass]
        ax2.plot(Xk[:, 0], Xk[:, 1], color, alpha=0.3)
    ax2.set_title('Automatic Clustering\nOPTICS')

    # DBSCAN at 0.5
    colors = ['g', 'greenyellow', 'olive', 'r', 'b', 'c']
    ax3.plot(pca_data[labels_050 == -1, 0], pca_data[labels_050 == -1, 1], 'k+', alpha=0.1)
    for klass, color in zip(range(0, 6), colors):
        Xk = pca_data[labels_050 == klass]
        ax3.plot(Xk[:, 0], Xk[:, 1], color, alpha=0.3, marker='.')
    ax3.set_title('Clustering at 0.5 epsilon cut\nDBSCAN')

    # DBSCAN at 2.
    colors = ['g.', 'm.', 'y.', 'c.']
    for klass, color in zip(range(0, 4), colors):
        Xk = pca_data[labels_200 == klass]
        ax4.plot(Xk[:, 0], Xk[:, 1], color, alpha=0.3)
    ax4.plot(pca_data[labels_200 == -1, 0], pca_data[labels_200 == -1, 1], 'k+', alpha=0.1)
    ax4.set_title('Clustering at 2.0 epsilon cut\nDBSCAN')

    plt.tight_layout()
    plt.show()

plot_optics(clustering)

metrics = ['euclidean', 'l1', 'manhattan']

for metric in metrics:
    clustering = OPTICS(max_eps=2, min_samples=3, cluster_method="dbscan", metric=metric).fit(scaled_data)
    plot_optics(clustering)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.decomposition import KernelPCA
from sklearn.decomposition import SparsePCA
from sklearn.decomposition import FactorAnalysis

if __name__ == '__main__':
    df = pd.read_csv('glass.csv')
    var_names = list(df.columns)
    labels = df.to_numpy('int')[:, -1]
    data = df.to_numpy('float')[:, :-1]
    #data = preprocessing.minmax_scale(data)
    fig, axs = plt.subplots(2, 4)

    for i in range(data.shape[1] - 1):
       axs[i // 4, i % 4].scatter(data[:, i], data[:, (i + 1)], c=labels)
       axs[i // 4, i % 4].set_xlabel(var_names[i])
       axs[i // 4, i % 4].set_ylabel(var_names[i + 1])
       scatter = axs[i // 4, i % 4].scatter(data[:, i], data[:, (i + 1)], c=labels)
       legend1 = axs[i // 4, i % 4].legend(*scatter.legend_elements(), loc="upper right", title="Classes")
       axs[i // 4, i % 4].add_artist(legend1)

    plt.show()
    pca = PCA(n_components=2)
    pca_data_auto = pca.fit(preprocessing.minmax_scale(data)).transform(preprocessing.minmax_scale(data))
    print("Дисперсия")
    print(pca.explained_variance_ratio_)
    print("Собственные числа")
    print(pca.singular_values_)

    #plt.scatter(pca_data_auto[:, 0], pca_data_auto[:, 1], c=labels)
    #plt.show()

    fig, axs = plt.subplots(2, 2)

    axs[0, 0].scatter(pca_data_auto[:, 0], pca_data_auto[:, 1], c=labels)
    axs[0, 0].set_title('auto')

    pca = PCA(svd_solver='full', n_components=2)
    pca_data_full = pca.fit(data).transform(data)

    axs[0, 1].scatter(pca_data_full[:, 0], pca_data_full[:, 1], c=labels)
    axs[0, 1].set_title('full')

    pca = PCA(svd_solver='arpack', n_components=2)
    pca_data_arpack = pca.fit(data).transform(data)

    axs[1, 0].scatter(pca_data_arpack[:, 0], pca_data_arpack[:, 1], c=labels)
    axs[1, 0].set_title('arpack')

    pca = PCA(svd_solver='randomized', n_components=2)
    pca_data_rnd = pca.fit(data).transform(data)

    axs[1, 1].scatter(pca_data_rnd[:, 0], pca_data_rnd[:, 1], c=labels)
    axs[1, 1].set_title('randomized')

    plt.show()

    # KernelPCA

    fig, axs = plt.subplots(2, 3)

    kpca = KernelPCA(n_components=2, kernel="linear")
    kpca_data_linear = kpca.fit(data).transform(data)

    axs[0, 0].scatter(kpca_data_linear[:, 0], kpca_data_linear[:, 1], c=labels)
    axs[0, 0].set_title('linear')

    kpca = KernelPCA(n_components=2, kernel='poly')
    kpca_data_poly = kpca.fit(data).transform(data)

    axs[0, 1].scatter(kpca_data_poly[:, 0], kpca_data_poly[:, 1], c=labels)
    axs[0, 1].set_title('poly')

    kpca = KernelPCA(n_components=2, kernel='rbf')
    kpca_data_rbf = kpca.fit(data).transform(data)

    axs[0, 2].scatter(kpca_data_rbf[:, 0], kpca_data_rbf[:, 1], c=labels)
    axs[0, 2].set_title('rbf')

    kpca = KernelPCA(n_components=2, kernel='sigmoid', fit_inverse_transform=True)
    kpca_data_sig = kpca.fit(preprocessing.minmax_scale(data)).transform(preprocessing.minmax_scale(data))

    axs[1, 0].scatter(kpca_data_sig[:, 0], kpca_data_sig[:, 1], c=labels)
    axs[1, 0].set_title('sigmoid')

    kpca = KernelPCA(n_components=2, kernel='cosine')
    kpca_data_cos = kpca.fit(data).transform(data)

    axs[1, 1].scatter(kpca_data_cos[:, 0], kpca_data_cos[:, 1], c=labels)
    axs[1, 1].set_title('cosine')

    # SparsePCA

    plt.show()

    spca = SparsePCA(n_components=2)
    spca_data = spca.fit(preprocessing.minmax_scale(data)).transform(preprocessing.minmax_scale(data))
    plt.scatter(spca_data[:, 0], spca_data[:, 1], c=labels)

    plt.show()

    # FactorAnalysis

    fanalysis = FactorAnalysis(n_components=2)
    fanalysis_data = fanalysis.fit(preprocessing.minmax_scale(data)).transform(preprocessing.minmax_scale(data))
    plt.scatter(fanalysis_data[:, 0], fanalysis_data[:, 1], c=labels)

    plt.show()


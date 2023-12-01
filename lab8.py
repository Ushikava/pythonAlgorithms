import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import matplotlib.pyplot as plt
from sklearn import svm

data = pd.read_csv('iris.data', header=None)
print(data)

X = data.iloc[:,:4].to_numpy()
labels = data.iloc[:,4].to_numpy()

le = preprocessing.LabelEncoder()
Y = le.fit_transform(labels)
print(Y)

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.5, random_state=0)

clf = LinearDiscriminantAnalysis()
y_pred = clf.fit(X_train, y_train).predict(X_test)
print(y_pred)
print("Некорректные наблюдения: ", (y_test != y_pred).sum())

print("Результат функции score(): ", clf.score(X_test, y_test))

def method_graph(method, title=""):
    test_sizes = np.arange(0.05, 0.95, 0.05)
    wrong_results = []
    accuracies = []
    for test_size in test_sizes:
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=202305)
        y_pred = method.fit(X_train, y_train).predict(X_test)
        wrong_results.append((y_test != y_pred).sum())
        accuracies.append(method.score(X_test, y_test))
    fig, axs = plt.subplots(1, 2, figsize=(8, 4))
    axs[0].plot(test_sizes, wrong_results)
    axs[1].plot(test_sizes, accuracies)
    axs[0].set_xlabel(title)
    axs[1].set_xlabel(title)
    plt.tight_layout()
    plt.show()

method_graph(clf, 'LinearDiscriminantAnalysis')

transform_data = clf.transform(X_train)
plt.figure()
colors = ['r', 'g', 'b']
lw = 2
for color, i in zip(colors, [0, 1, 2]):
    plt.scatter(transform_data[y_train == i, 0], transform_data[y_train == i, 1], color=color, alpha=.8, lw=lw)
plt.show()

clf = LinearDiscriminantAnalysis(solver='eigen', shrinkage='auto')
method_graph(clf, 'LinearDiscriminantAnalysis')

clf = LinearDiscriminantAnalysis(solver='lsqr', shrinkage=0.5)
method_graph(clf, 'LinearDiscriminantAnalysis')

method_graph(LinearDiscriminantAnalysis(priors=[0.15, 0.7, 0.15]), '[0.15, 0.7, 0.15]')

clf = svm.SVC()
y_pred = clf.fit(X_train, y_train).predict(X_test)
print(y_pred)
print("Некорректные наблюдения: ", (y_test != y_pred).sum())
print("Результат функции score(): ", clf.score(X, Y))

print("support_vectors_: ", clf.support_vectors_)
print("support_: ", clf.support_)
print("n_support_: ", clf.n_support_)

method_graph(clf, 'SVC')
method_graph(svm.SVC(kernel='linear', degree=4, max_iter=400), 'SVC')
method_graph(svm.SVC(kernel='sigmoid', degree=2, max_iter=100), 'SVC')
method_graph(svm.NuSVC(), 'NuSVC')
method_graph(svm.LinearSVC(), 'LinearSVC')

import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import ComplementNB
from sklearn.naive_bayes import BernoulliNB
import matplotlib.pyplot as plt
from sklearn import tree

data = pd.read_csv('iris.data', header=None)
print(data)

X = data.iloc[:,:4].to_numpy()
labels = data.iloc[:,4].to_numpy()

le = preprocessing.LabelEncoder()
Y = le.fit_transform(labels)

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.5)

gnb = GaussianNB()

y_pred = gnb.fit(X_train, y_train).predict(X_test)
print(y_pred)
print("Некорректные наблюдения: ", (y_test != y_pred).sum())

print("Результат функции score(): ", gnb.score(X_test, y_test))

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

# gnb = GaussianNB()
# method_graph(gnb, 'GaussianNB')
# mnb = MultinomialNB()
# method_graph(mnb, 'MultinomialNB')
# cnb = ComplementNB()
# method_graph(cnb, 'ComplementNB')
# bnb = BernoulliNB()
# method_graph(bnb, 'BernoulliNB')

clf = tree.DecisionTreeClassifier()
y_pred = clf.fit(X_train, y_train).predict(X_test)
print(y_pred)
print("Некорректные наблюдения: ", (y_test != y_pred).sum())

print("Результат функции score(): ", clf.score(X_test, y_test))

print("get_n_leaves: ", clf.get_n_leaves())
print("get_depth: ", clf.get_depth())

plt.subplots(1, 1, figsize=(10, 10))
tree.plot_tree(clf, filled=True)
plt.show()
method_graph(clf, 'DecisionTreeClassifier')

clf = tree.DecisionTreeClassifier(criterion="entropy", splitter="random", max_depth=2, min_samples_split=0.5, min_samples_leaf=2)
y_pred = clf.fit(X_train, y_train).predict(X_test)
tree.plot_tree(clf, filled=True)
plt.show()
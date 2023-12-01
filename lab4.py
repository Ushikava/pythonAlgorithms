import mlxtend
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import fpgrowth
from mlxtend.frequent_patterns import fpmax
from mlxtend.frequent_patterns import association_rules
import networkx as nx


if __name__ == '__main__':
    all_data = pd.read_csv('groceries - groceries.csv')
    print(all_data)

    np_data = all_data.to_numpy()
    np_data = [[elem for elem in row[1:] if isinstance(elem, str)] for row in np_data]
    #print('\n', np_data)

    unique_items = set()
    for row in np_data:
        for elem in row:
            unique_items.add(elem)

    print('\n', unique_items, '\n', "Количество товаров:", len(unique_items))

    te = TransactionEncoder()
    te_ary = te.fit(np_data).transform(np_data)
    data = pd.DataFrame(te_ary, columns=te.columns_)
    print(data)

    result = fpgrowth(data, min_support=0.03, use_colnames=True)
    result['length'] = result['itemsets'].apply(lambda x: len(x))
    print(result)

    result_sup1 = result[result['length'] == 1]
    print("Минимальная поддержка при размерности 1 -", min(result_sup1['support']))
    print("Максимальная поддержка при размерности 1 -", max(result_sup1['support']))

    result_sup2 = result[result['length'] == 2]
    print("Минимальная поддержка при размерности 2 -", min(result_sup2['support']))
    print("Максимальная поддержка при размерности 2 -", max(result_sup2['support']))

    result = fpmax(data, min_support=0.03, use_colnames=True)
    result['length'] = result['itemsets'].apply(lambda x: len(x))
    print(result)

    result_sup1 = result[result['length'] == 1]
    print("Минимальная поддержка при размерности 1 -", min(result_sup1['support']))
    print("Максимальная поддержка при размерности 1 -", max(result_sup1['support']))

    result_sup2 = result[result['length'] == 2]
    print("Минимальная поддержка при размерности 2 -", min(result_sup2['support']))
    print("Максимальная поддержка при размерности 2 -", max(result_sup2['support']))

    plt.figure()
    count_of_items = data.sum()
    count_of_items.nlargest(10).plot.bar()
    plt.tight_layout()
    plt.show()

    items = ['whole milk', 'yogurt', 'soda', 'tropical fruit', 'shopping bags', 'sausage',
             'whipped/sour cream', 'rolls/buns', 'other vegetables', 'root vegetables',
             'pork', 'bottled water', 'pastry', 'citrus fruit', 'canned beer', 'bottled beer']
    np_data = all_data.to_numpy()
    np_data = [[elem for elem in row[1:] if isinstance(elem, str) and elem in
                items] for row in np_data]
    te = TransactionEncoder()
    te_ary = te.fit(np_data).transform(np_data)
    new_data = pd.DataFrame(te_ary, columns=te.columns_)

    result = fpgrowth(new_data, min_support=0.03, use_colnames=True)
    result['length'] = result['itemsets'].apply(lambda x: len(x))
    print(result)

    result_sup1 = result[result['length'] == 1]
    print("Минимальная поддержка при размерности 1 -", min(result_sup1['support']))
    print("Максимальная поддержка при размерности 1 -", max(result_sup1['support']))

    result_sup2 = result[result['length'] == 2]
    print("Минимальная поддержка при размерности 2 -", min(result_sup2['support']))
    print("Максимальная поддержка при размерности 2 -", max(result_sup2['support']))

    result = fpmax(new_data, min_support=0.03, use_colnames=True)
    result['length'] = result['itemsets'].apply(lambda x: len(x))
    print(result)

    result_sup1 = result[result['length'] == 1]
    print("Минимальная поддержка при размерности 1 -", min(result_sup1['support']))
    print("Максимальная поддержка при размерности 1 -", max(result_sup1['support']))

    result_sup2 = result[result['length'] == 2]
    print("Минимальная поддержка при размерности 2 -", min(result_sup2['support']))
    print("Максимальная поддержка при размерности 2 -", max(result_sup2['support']))

    colors = ['b', 'g', 'r', 'm', 'y']
    for i in range(1, 6):
        arr = []
        for minSup in np.linspace(0.005, 1.0, 50):
            results = fpgrowth(data, min_support=minSup, use_colnames=True, max_len=i)
            results['length'] = results['itemsets'].apply(lambda x: len(x))
            results = results[results['length'] == i]
            arr.append(len(results))
        plt.plot(np.linspace(0.005, 1, 50), arr, colors[i - 1], label=("Length", + i))
    plt.legend()
    plt.show()

    np_data = all_data.to_numpy()
    np_data = [[elem for elem in row[1:] if isinstance(elem, str) and elem in items] for row in np_data]
    np_data = [row for row in np_data if len(row) > 1]
    result = fpgrowth(data, min_support=0.05, use_colnames=True)
    rules = association_rules(result, min_threshold=0.3)
    print(rules)

    metrics = ["support", "confidence", "lift", "leverage", "conviction"]

    for metric in metrics:
        print("\nМетрика:", metric)
        rules = association_rules(result, min_threshold=0.001, metric=metric)
        print("Среднее значение:", rules[metric].mean())
        print("Медиана:", rules[metric].median())
        print("СКО:", rules[metric].std())

    rules = association_rules(result, min_threshold=0.3, metric='confidence')
    print(rules)
    G = nx.DiGraph()

    for index, row in rules.iterrows():
        l = list(row['antecedents'])[0]
        r = list(row['consequents'])[0]
        w = row['support'] * 25
        label = round(row['confidence'], 4)
        G.add_edge(l, r, label=label, weight=w)
    pos = nx.spring_layout(G)
    plt.figure()
    nx.draw_networkx(G, pos, with_labels=True)
    nx.draw_networkx_edges(G, pos, width=list([G[n1][n2]['weight'] for n1, n2 in G.edges]))
    nx.draw_networkx_edge_labels(G, pos, edge_labels=dict([((n1, n2), f'{G[n1][n2]["label"]}') for n1, n2 in G.edges]), font_color='black')
    plt.show()
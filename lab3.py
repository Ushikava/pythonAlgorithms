import matplotlib.pyplot as plt
import numpy
import pandas as pd
import numpy as np
from sklearn import preprocessing
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori
from pprint import pp

if __name__ == '__main__':
    all_data = pd.read_csv('dataset_group.csv', header=None)
    print(all_data)
    # id
    unique_id = list(set(all_data[1]))
    print("Всего id:", len(unique_id))
    # товары
    items = list(set(all_data[2]))
    print("Товаров:", len(items))

    dataset = [[elem for elem in all_data[all_data[1] == id][2] if elem in items] for id in unique_id]
    print(dataset)

    te = TransactionEncoder()
    te_ary = te.fit(dataset).transform(dataset)
    df = pd.DataFrame(te_ary, columns=te.columns_)

    print(df)

    #results = apriori(df, min_support=0.3, use_colnames=True)
    #results['length'] = results['itemsets'].apply(lambda x: len(x))
    #print(results)

    #results = apriori(df, min_support=0.3, use_colnames=True, max_len=1)
    #print(results)

    num_of_items = []
    borders = []
    border = 0.05
    step = 0.01
    total_length = 4

    while border <= 0.45:
        results = apriori(df, min_support=border, use_colnames=True)
        results['length'] = results['itemsets'].apply(lambda x: len(x))
        num_of_items.append(len(results))
        borders.append(border)
        length_results = results[results['length'] == total_length]
        if len(length_results) == 0:
            print('\nLength', total_length, "больше не встречается с шага", border)
            total_length -= 1
        border += step
    num_of_items = np.array(num_of_items)
    print(num_of_items)
    print('\nCount of searches = ', len(num_of_items))

    #plt.plot(num_of_items, borders, marker="o")
    #plt.show()

    results = apriori(df, min_support=0.38, use_colnames=True, max_len=1)
    new_items = [list(elem)[0] for elem in results['itemsets']]
    new_dataset = [[elem for elem in all_data[all_data[1] == id][2] if elem in
                    new_items] for id in unique_id]

    te_ary = te.fit(new_dataset).transform(new_dataset)
    ndf = pd.DataFrame(te_ary, columns=te.columns_)

    #results = apriori(ndf, min_support=0.3, use_colnames=True)
    #results['length'] = results['itemsets'].apply(lambda x: len(x))
    #print(results)

    results = apriori(ndf, min_support=0.15, use_colnames=True)
    results['length'] = results['itemsets'].apply(lambda x: len(x))
    results = results[results['length'] >= 2]

    def check(f):
        return f.issuperset({'yogurt'}) or f.issuperset({'waffles'})

    results = results[results['itemsets'].apply(check)]
    #results = results[results['itemsets'].apply(lambda y: y.issuperset({'yogurt'}) or y.issuperset({'waffles'}))]
    print(results, '\nИтого элементов:', len(results))

    results = apriori(df, min_support=0.38, use_colnames=True, max_len=1)
    print('\nИтого results:', len(results))
    other_results = apriori(df, min_support=0.05, use_colnames=True)
    print('\nИтого other_results:', len(other_results))
    task_result = pd.concat([results, other_results]).drop_duplicates(keep=False)
    print(task_result)

    new_items = [list(elem)[0] for elem in task_result['itemsets']]
    new_dataset = [[elem for elem in all_data[all_data[1] == id][2] if elem in
                    new_items] for id in unique_id]

    te_ary = te.fit(new_dataset).transform(new_dataset)
    ndf = pd.DataFrame(te_ary, columns=te.columns_)
    print(ndf)

    results = apriori(ndf, min_support=0.3, use_colnames=True)
    results['length'] = results['itemsets'].apply(lambda x: len(x))
    print(results)

    # task 12
    te = TransactionEncoder()
    te_ary = te.fit(dataset).transform(dataset)
    my_df = pd.DataFrame(te_ary, columns=te.columns_)

    results = apriori(my_df, min_support=0.05, use_colnames=True)
    results = results[results['itemsets'].apply(lambda x: np.fromiter(map(lambda y: y.startswith('s'), x), dtype=bool).sum() >= 2)]
    print(results)

    # task 13

    te = TransactionEncoder()
    te_ary = te.fit(dataset).transform(dataset)
    my_df = pd.DataFrame(te_ary, columns=te.columns_)

    results = apriori(my_df, min_support=0.1, use_colnames=True)
    other_results = apriori(my_df, min_support=0.25, use_colnames=True)
    task_result = pd.concat([results, other_results]).drop_duplicates(keep=False)
    print(task_result)

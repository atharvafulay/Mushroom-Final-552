import pandas as pd
from sklearn.utils import shuffle
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import KFold
from sklearn import tree
from subprocess import call
import os
import matplotlib.pyplot as plt
import analysis


def load_data():
    mushrooms = list()
    with open('agaricus-lepiota.data') as f:
        for line in f:
            mushrooms.append(line.strip().split(','))

    df = pd.DataFrame(mushrooms)
    # df.columns = cols # will have to do this manually

    df.columns = ['edible-posionous', 'cap-shape', 'cap-surface', 'cap-color', 'bruises', 'odor', 'gill-attachment',
                  'gill-spacing', 'gill-size', 'gill-color', 'stalk-shape', 'stalk-root', 'stalk-surface-above-ring',
                  'stock-shape-below-ring', 'stalk-color-above-ring', 'stalk-color-below-ring', 'veil-type', 'veil-color',
                  'ring-number', 'ring-type', 'spore-print-color', 'population', 'habitat']

    df = df.apply(lambda x: pd.factorize(x)[0])
    df = shuffle(df)
    df.reset_index(inplace=True, drop=True)

    X = df.loc[:, df.columns != 'edible-posionous']
    y = df.loc[: , 'edible-posionous']

    try:
        os.mkdir('images')
    except FileExistsError:
        pass

    return X, y, list(df.columns)


def k_fold_model_analysis(X, y):
    print('\n---------- K-Fold Model Analysis ----------')

    # Using K-fold cross validation
    num_splits = 10
    kf = KFold(n_splits=num_splits)

    rfc_score = 0.0
    dtc_score = 0.0
    md = 4

    for train_ind, test_ind in kf.split(X):
        X_train, X_test = X.loc[train_ind], X.loc[test_ind]
        y_train, y_test = y.loc[train_ind], y.loc[test_ind]

        rfc = RandomForestClassifier(n_jobs=-1, criterion='entropy', max_depth=md, max_features='sqrt')
        dtc = DecisionTreeClassifier(criterion='entropy', max_depth=md, max_features='sqrt')

        rfc.fit(X_train, y_train)
        rfc_score += rfc.score(X_test, y_test)

        dtc.fit(X_train, y_train)
        dtc_score += dtc.score(X_test, y_test)

    avg_rfc = rfc_score / num_splits
    avg_dtc = dtc_score / num_splits

    print(f'Using K-Fold: average RFC accuracy (max_depth={md}, max_features=sqrt)', avg_rfc)
    print(f'Using K-Fold: average DTC accuracy (max_depth={md}, max_features=sqrt)', avg_dtc)


def tradition_train_test(X, y):
    print('\n---------- Generate Various models for visual analysis ----------')
    # traditional train/test datasets

    train_X = X.loc[:5999]
    train_y = y.loc[:5999]

    test_X = X.loc[6000:]
    test_y = y.loc[6000:]

    res_rfc = list()
    res_dtc = list()

    for mf in [None, 'sqrt']:
        for md in range(1, 11):
            for cw in [None, 1]:
                if cw is None:
                    rfc = RandomForestClassifier(n_jobs=-1, criterion='entropy', max_depth=md, max_features=mf)
                    dtc = DecisionTreeClassifier(criterion='entropy', max_depth=md, max_features=mf)
                else:
                    rfc = RandomForestClassifier(n_jobs=-1, criterion='entropy', max_depth=md, max_features=mf,
                                                 class_weight={0: 1, 1: 100})
                    dtc = DecisionTreeClassifier(criterion='entropy', max_depth=md, max_features=mf,
                                                 class_weight={0: 1, 1: 100})

                rfc.fit(train_X, train_y)
                rfc_train_score = rfc.score(train_X, train_y)
                rfc_score = rfc.score(test_X, test_y)

                dtc.fit(train_X, train_y)
                dtc_train_score = dtc.score(train_X, train_y)
                dtc_score = dtc.score(test_X, test_y)

                if cw is None:
                    cw_file = 'None'
                else:
                    cw_file = 'dictionary'

                if mf is None:
                    mf_file = 'None'
                else:
                    mf_file = mf

                res_rfc.append(f'{rfc_train_score}, {rfc_score}, {md}, {mf_file}, {cw_file}\n')
                res_dtc.append(f'{dtc_train_score}, {dtc_score}, {md}, {mf_file}, {cw_file}\n')

    # separating these because it was causing me some issues when I had these within the triple loop above
    with open('rfc_results.txt', 'w') as f1:
        f1.write('#train, test, max_depth, max_features, class_weight\n')

        for line in res_rfc:
            f1.write(line)

    with open('dtc_results.txt', 'w') as f2:
        f2.write('#train, test, max_depth, max_features, class_weight\n')

        for line in res_dtc:
            f2.write(line)

    print('You can now look for "rfc_results.txt" and "dtc_results.txt". analysis.py will make use of these.')


def generate_optimal_tree(dtc, feat_names):
    print('\n---------- Generating Optimal Decision Tree Visual (see images folder) ----------')

    tree.export_graphviz(dtc, 'optimal_tree.dot', feature_names=feat_names[1:])
    try:
        call(['dot', '-Tpng', 'optimal_tree.dot', '-o', 'images/optimal_tree.png', '-Gdpi=60'])
        os.remove("optimal_tree.dot")
    except FileNotFoundError:
        print('Error with the .dot file made. See below for details. The rest of the program will continue...')
        print('     ...This is a known issue with graphviz. You may have to install it using homebrew, '
              '"brew install graphviz", or a solution from the link below.')
        print('     ...See here for more details: https://github.com/WillKoehrsen/Data-Analysis/issues/36'
              '#issuecomment-498710710')
    else:
        print('Check the images folder for "optimal_tree.png".')


def generate_feat_imp_visuals(X, y, columns):

    train_X = X.loc[:5999]
    train_y = y.loc[:5999]
    rfc = RandomForestClassifier(n_jobs=-1, criterion='entropy', class_weight={0: 1, 1: 100})
    rfc.fit(train_X, train_y)

    print('\n---------- Generating Optimal RFC Feature Importances Visual ----------')
    combined = zip(rfc.feature_importances_, columns[1:])
    combined = sorted(list(combined), reverse=True)

    sorted_feature_imps = list()
    sorted_cols = list()

    for i in combined:
        sorted_feature_imps.append(i[0])
        sorted_cols.append(i[1])


    plt.bar(sorted_cols, sorted_feature_imps)
    plt.xticks(rotation=90)
    plt.title('Feature Importances on Optimal RFC')
    plt.savefig('images/feature_importances_optimal_RFC.png', bbox_inches='tight')
    plt.clf()

    print('Check the images folder for "feature_importances_optimal_RFC.png".')


def generate_conf_matrix(X, y):
    print('\n---------- Confusion matrices for optimal DTC and RFC ----------')

    # note: the reason for using sqrt (for comparison purposes) is that with None, both models
    # return identical results
    train_X = X.loc[:5999]
    train_y = y.loc[:5999]
    rfc = RandomForestClassifier(criterion='entropy', max_features='sqrt', class_weight={0: 1, 1: 100})
    rfc.fit(train_X, train_y)

    dtc = DecisionTreeClassifier(criterion='entropy', max_features='sqrt', class_weight={0: 1, 1: 100})
    dtc.fit(train_X, train_y)

    # note: poisonous = 0, edible = 1
    predictions = rfc.predict(X)
    print('RFC (max_features=sqrt, weighted):')
    print(pd.crosstab(y, predictions, rownames=['Actual'], colnames=['Predicted']))

    predictions = dtc.predict(X)
    print('\nDTC (max_features=sqrt, weighted):')
    print(pd.crosstab(y, predictions, rownames=['Actual'], colnames=['Predicted']))

    # same process as above but now with max-depth capped DTC and RFC
    md = 5
    rfc = RandomForestClassifier(criterion='entropy', max_features='sqrt', class_weight={0: 1, 1: 100}, max_depth=md)
    rfc.fit(train_X, train_y)

    dtc = DecisionTreeClassifier(criterion='entropy', max_features='sqrt', class_weight={0: 1, 1: 100}, max_depth=md)
    dtc.fit(train_X, train_y)

    predictions = rfc.predict(X)
    print('\nMax-depth capped RFC (max_features=sqrt, weighted, max_depth=5):')
    print(pd.crosstab(y, predictions, rownames=['Actual'], colnames=['Predicted']))

    predictions = dtc.predict(X)
    print('\nMax-depth capped DTC (max_features=sqrt, weighted, max_depth=5):')
    print(pd.crosstab(y, predictions, rownames=['Actual'], colnames=['Predicted']))


if __name__ == '__main__':
    plt.figure(figsize=(6, 8))

    X, y, feat_names = load_data()

    # optimal random forest classifier
    train_X = X.loc[:5999]
    train_y = y.loc[:5999]
    rfc = RandomForestClassifier(criterion='entropy', max_features=None, class_weight={0: 1, 1: 100})
    rfc.fit(train_X, train_y)

    # optimal decision tree classifier
    train_X = X.loc[:5999]
    train_y = y.loc[:5999]
    dtc = DecisionTreeClassifier(criterion='entropy', max_features=None, class_weight={0: 1, 1: 100})
    dtc.fit(train_X, train_y)

    k_fold_model_analysis(X, y)
    tradition_train_test(X, y)
    generate_optimal_tree(dtc, feat_names)
    generate_feat_imp_visuals(X, y, feat_names)
    generate_conf_matrix(X, y)
    print('\n---------- End data_and_models.py ----------')
    print('\n---------- Calling analysis.py ----------')
    dtdf = analysis.load_dtc_data()
    rfdf = analysis.load_rfc_data()
    analysis.generate_all_visuals(dtdf, rfdf)
    print('\n---------- End analysis.py ----------')
    print('\n---------- End ----------')

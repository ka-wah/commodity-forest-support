import os
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier

"""
This file is used to create Random Forest predictions.

"""

returns = pd.read_csv("data\\input\\binarycmo.csv", index_col=0)
indicators = ["filtering", "momentum", "moving-average", "oscillator", "support-resistance"]
all_coms = ['energy', 'non-energy', 'precious-metals', 'food', 'beverages', 'raw-materials', 'metals-minerals',
            'agriculture']

def classify(bin, features, trees):
    """
    Create Random Forest predictions via expanding window.
    :param bin: encoding
    :param features: number of max features considered at each split
    :param trees: max trees in the ensemble
    """
    for com in all_coms:
        print(f'{bin} {features} {trees} {com}')
        y = returns[com][277:]
        techindis = pd.DataFrame(index=y.index)
        y = y.shift(-1)
        y = y[1:len(y) - 1]

        for dirpath, dirnames, filenames in os.walk(f'technical-indicators\\{bin}'):
            for filename in filenames:
                file_path = os.path.join(dirpath, filename)
                techindi = pd.read_csv(file_path, index_col=0)
                column_name = filename.split('.')[0]
                techindis = pd.concat([techindis, techindi[com][12:]], axis=1)
                techindis.rename(columns={com: column_name}, inplace=True)

        techindis = techindis.iloc[1:]
        testdate = 94
        preds = []
        train_acc = []
        feature_importances = []
        i = 0
        while testdate < 479:
            x = techindis
            y_train, x_train = y.iloc[:testdate], x.iloc[:testdate]
            try:
                y_test, x_test = y.iloc[testdate], x.iloc[testdate]
            except:
                break
            print(y_train, x_train)
            print(y_test, x_test)
            return 0
            model = RandomForestClassifier(n_estimators=trees, max_features=features, random_state=0)

            model.fit(x_train, y_train)

            feature_importances.append(model.feature_importances_)

            x_test_df = pd.DataFrame(x_test).transpose()

            y_p = model.predict(x_test_df)
            y_p_train = model.predict(x_train)

            train_acc.append(accuracy_score(y_train, y_p_train))

            preds.append(y_p)
            testdate += 1

        df_predictions = pd.concat([pd.Series(preds, name='predicted').reset_index(drop=True),y[94:479].reset_index(drop=True)], axis=1)
        df_predictions.columns = ["predicted", "true"]
        acc = accuracy_score(y_pred=preds, y_true=y[94:479])*100

        new_index = pd.date_range(start="1991-1", periods=len(df_predictions), freq='M')
        df_predictions.index = new_index
        df_predictions.to_csv(
            f'data\\output\\RF\\predictions\\{bin}\\{features}\\{trees}\\{com}-A{acc:.2f}-TA{np.mean(train_acc)*100:.2f}.csv')
        feature_importances = pd.DataFrame(feature_importances, columns=x.columns)
        feature_importances.to_csv(f'data\output\RF\\feature-importances\\{bin}\\{features}-{trees}-{com}.csv')

def select_features(bin):
    """
    Select features that have importance greater than specified threshold for the SVM feature selection.
    :param bin: encoding
    """
    dic = {}
    if bin == 'binary':
        thresh = 0.02
    elif bin == 'non-binary':
        thresh = 0.03

    for com in all_coms:
        avgs = pd.DataFrame()
        files = os.listdir(f'data\output\RF\\feature-importances\\{bin}')
        for file in [f for f in files if com in f]:
            feature_importances = pd.read_csv(f'data\output\RF\\feature-importances\\{bin}\\{file}', index_col=0)
            avgs = pd.concat([avgs, feature_importances.mean()], axis=1)
        avgs = avgs.mean(axis=1)
        avgs.to_csv(f'data\output\measures\\RF\\featimps\\{bin}-{com}')
        dic[com] = avgs[avgs > thresh].index.tolist()

    file_path = f"data\output\measures\\RF\\featimps\selected-features-{bin}.txt"

    with open(file_path, 'w') as file:
        file.write('{')
        for key, value in dic.items():
            file.write(f"'{key}': {value}\n")
        file.write('}')

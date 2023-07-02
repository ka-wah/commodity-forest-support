import ast
import os
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC

"""
This file is used to create SVM predictions.

"""

returns = pd.read_csv("data\\input\\binarycmo.csv", index_col=0)
indicators = ["filtering", "momentum", "moving-average", "oscillator", "support-resistance"]
all_coms = ['energy', 'non-energy', 'precious-metals', 'food', 'beverages', 'raw-materials', 'metals-minerals',
            'agriculture']

with open('data/output/measures/RF/featimps/selected-features-binary.txt', 'r') as file:
    file_contents = file.read()
binfeats = ast.literal_eval(file_contents)

with open('data/output/measures/RF/featimps/selected-features-non-binary.txt', 'r') as file:
    file_contents = file.read()
nonbinfeats = ast.literal_eval(file_contents)

selected_feats = {'binary': binfeats, 'non-binary': nonbinfeats}


def classify(bin, c, gamma):
    """
    Create SVM predictions via expanding window.
    :param bin: encoding
    :param c: penalty parameter
    :param gamma: constant
    """
    for com in all_coms:
        print(com)
        y = returns[com][277:]
        techindis = pd.DataFrame(index=y.index)
        y = y.shift(-1)
        y = y[1:len(y) - 1]

        for dirpath, dirnames, filenames in os.walk(f'technical-indicators\\{bin}'):
            for filename in filenames:
                column_name = filename.split('.')[0]
                if column_name in selected_feats[bin][com]:
                    file_path = os.path.join(dirpath, filename)
                    techindi = pd.read_csv(file_path, index_col=0)
                    techindis = pd.concat([techindis, techindi[com][12:]], axis=1)
                    techindis.rename(columns={com: column_name}, inplace=True)

        techindis = techindis.iloc[1:]
        testdate = 94
        preds = []
        train_acc = []
        i = 0
        while testdate < 479:
            x = techindis
            y_train, x_train = y.iloc[:testdate], x.iloc[:testdate]
            try:
                y_test, x_test = y.iloc[testdate], x.iloc[testdate]
            except:
                break
            model = SVC(kernel='rbf', C=c, gamma=gamma, random_state=0)
            model.fit(x_train, y_train)

            x_test_df = pd.DataFrame(x_test).transpose()

            y_p = model.predict(x_test_df)
            y_p_train = model.predict(x_train)

            train_acc.append(accuracy_score(y_train, y_p_train))

            preds.append(y_p)
            testdate += 1

        df_predictions = pd.concat(
            [pd.Series(preds, name='predicted').reset_index(drop=True), y[94:479].reset_index(drop=True)], axis=1)
        df_predictions.columns = ["predicted", "true"]
        acc = accuracy_score(y_pred=preds, y_true=y[94:479]) * 100

        new_index = pd.date_range(start="1991-1", periods=len(df_predictions), freq='M')
        df_predictions.index = new_index
        df_predictions.to_csv(
            f'data\\output\\SVM\\predictions\\{bin}\\{c}\\{gamma}\\{com}-A{acc:.4f}-TA{np.mean(train_acc) * 100:.4f}.csv')

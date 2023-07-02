import pandas as pd
import os
import statsmodels.api as sm

from sklearn.metrics import accuracy_score

"""
This file creates the OLS predictions.

"""

rules = ["filtering", "momentum", "moving-average", "oscillator", "support-resistance"]
all_coms = ['energy', 'non-energy', 'precious-metals', 'food', 'beverages', 'raw-materials', 'metals-minerals',
            'agriculture']
full_name = {'energy': 'energy', 'non': 'non-energy', 'agriculture': 'agriculture', 'beverages': 'beverages',
             'food': 'food', 'raw': 'raw-materials', 'metals': 'metals-minerals', 'precious': 'precious-metals'}


def regress(binary, rule):
    """
    Create OLS regression, per encoding and technical rule.

    :param binary: the encoding
    :param rule: indicator rule
    """
    returns = pd.read_csv("data\input\logcmo.csv", index_col=0).shift(-1)
    returns = returns.loc['1982M01':'2023M01']
    new_index = returns.index

    coms = {}
    true = {}
    avg_preds = {}

    for dirpath, dirnames, filenames in os.walk(f'non-standard-ti\\technical-indicators\\{binary}\\{rule}'):
        for com in all_coms:
            coms[com] = pd.DataFrame(columns=filenames, index=new_index)
            avg_preds[com] = pd.DataFrame(columns=['Avg Pred', 'True'])
            true[com] = returns[com].loc['1991M01':]
        for filename in filenames:
            file_path = os.path.join(dirpath, filename)
            techindi = pd.read_csv(file_path, index_col=0)

            print(filename)

            for commodity in returns.columns:
                y = returns[commodity]
                x = techindi[commodity]

                datecount = 120 - 12 - 1 + 4 - 3

                while datecount < len(new_index):
                    x_train, y_train = x.loc[:new_index[datecount - 1]], y.loc[:new_index[datecount - 1]]
                    x_valid, y_valid = x[new_index[datecount]], y[new_index[datecount]]

                    x_train = sm.add_constant(x_train)

                    x_train = x_train.dropna()
                    y_train = y_train.loc[x_train.index]

                    model = sm.OLS(y_train, x_train)
                    results = model.fit()

                    y_pred = results.predict((1, x_valid))

                    coms[commodity].loc[new_index[datecount], filename] = y_pred[0]
                    datecount += 1
        for com in all_coms:
            coms[com].dropna(inplace=True)
            avg = coms[com].mean(axis=1)
            coms[com]['average'] = avg
            true = returns[com].loc[avg.index]
            acc = accuracy_score(avg.apply(lambda x: 1 if x > 0 else 0), true.apply(lambda x: 1 if x > 0 else 0)) * 100
            # mspe = 100* (1- (mean_squared_error(true[com], avg))/mean_squared_error(true[com], his_avg))
            coms[com].to_csv(f'data/output/OLS/nonbinarynew/{com}-{rule}-A{acc:.2f}.csv')


def equally_weighted(bin):
    """
    Create equally weighted predictions.

    :param bin: encoding
    """
    tabr2 = pd.DataFrame(columns=['energy', 'non', 'agriculture', 'beverages', 'food', 'raw',
                                  'metals', 'precious'])
    taba = pd.DataFrame(columns=tabr2.columns)

    dic = {}

    for dirpath, dirnames, filenames in os.walk(f'data//output//OLS//{bin}new'):
        for filename in filenames:
            file_path = os.path.join(dirpath, filename)
            techindi = pd.read_csv(file_path, index_col=0)
            com = filename.split('-')[0]
            ti = filename.split('-')[-2]

    allcoms = pd.DataFrame()
    for com in dic:
        allcoms[com] = dic[com].mean(axis=1)
    allcoms = allcoms.applymap(lambda x: 0 if x < 0 else 1)
    allcoms.to_csv(f'data/output/OLS/ew{bin}.csv')

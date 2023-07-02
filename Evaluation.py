import os
from collections import defaultdict

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.signal import savgol_filter
from sklearn.metrics import accuracy_score

"""
This file contains functions to evaluate several aspects of the obtained predictions.

"""

techni = ['momentum', 'filtering', 'average', 'oscillator', 'resistance']
coms = ['energy', 'non', 'agriculture', 'beverages', 'food', 'raw', 'metals', 'precious']
full_name = {'energy': 'energy', 'non': 'non-energy', 'agriculture': 'agriculture', 'beverages': 'beverages',
             'food': 'food', 'raw': 'raw-materials', 'metals': 'metals-minerals', 'precious': 'precious-metals'}


def accuracy(table):
    """
    Finds accuracy in cases where sklearn does not apply
    :param table:
    :return:
    """
    predictions = table['Avg Pred']
    true_values = table['True']
    return (sum((predictions > 0) == (true_values > 0)) / len(predictions)) * 100


def eval_acc(directory):
    """
    Evaluates accuracy of a whole prediction file

    :param directory: predictions to be evaluated
    """
    for root, dirs, files in os.walk(f'data\\output\\{directory}'):
        for file in files:
            print(file)
            df = pd.read_csv(f'data\\output\\{directory}\\{file}', index_col=0)
            df['predicted'] = df['predicted'].apply(lambda x: int(x.strip('[]')))
            print(accuracy_score(df['predicted'], df['true']))


def acc_from_name(filename):
    """
    Obtains accuracy from filename.
    :param filename: name of file to obtain accuracy from
    :return: accuracy and name
    """
    filename = filename.strip('.csv')
    words = filename.split('-')
    return words[-2][1:], words[-1][2:], words[0]


def parameter_from_name(directory_path):
    """
    Obtains parameters from filename
    :param directory_path: name of file to obtain parameters frmo
    :return: parameters
    """
    words = directory_path.split('/')
    print(words)
    return words[-5], words[-3], words[-2], words[-1]


def gather_acc(directory):
    """
    Gather accuracies of a model directory
    :param directory: model to obtain accuracies from
    """
    acc_dic = {}
    for root, dirs, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            directory_path = os.path.dirname(file_path)
            params = parameter_from_name(directory_path)
            accs = acc_from_name(file)
            new_row = pd.Series({'Training': accs[1], 'Hold-out': accs[0]}, name=accs[2])
            if params in acc_dic:
                acc_dic[params] = pd.concat([acc_dic[params], new_row], axis=1)
            else:
                acc_dic[params] = pd.DataFrame(new_row)
    for x in acc_dic:
        acc_dic[x].to_csv(f'data/output/measures/{x[0]}/{x[1]}/{x[2]}-{x[3]}.csv')


def time_accuracy():
    """
    Find accuracy over time, by means of a rolling window.
    """
    returns = pd.read_csv('data/input/binarycmo.csv', index_col=0).shift(-1).loc['1991M01':'2023M01']
    d = {}
    for root, dirs, files in os.walk('data/output/measures/overall/aggregated-preds'):
        for com in list(full_name.values()):
            d[com] = pd.DataFrame()
        for file in files:
            file_path = os.path.join(root, file)
            df = pd.read_csv(file_path, index_col=0)
            if 'OLS' in file:
                df = df.applymap(lambda x: 1 if x > 0 else 0)
            df.index = returns.index

            for com in df.columns:
                accuracies = []
                window_size = 12 * 2
                for i in range(len(df)):
                    if i >= window_size - 1:
                        window = df.iloc[i - window_size + 1:i + 1]
                        truewindow = returns.iloc[i - window_size + 1:i + 1]
                        accuracy = (window[com] == truewindow[com]).mean() * 100
                        accuracies.append(accuracy)
                    else:
                        accuracies.append(
                            None)
                d[com][file] = pd.Series(accuracies, index=returns.index)
    for com in d.keys():
        d[com].to_csv(f'data/output/measures/overall/timeacc24/{com}.csv')


def aggregate_timeacc(direc):
    """
    Aggregate all time accuracies from a directory.
    :param direc: directory to aggregate time accuracies from.
    """
    d = {}
    for com in coms:
        d[com] = pd.DataFrame()
    for root, dirs, files in os.walk(f'data/output/{direc}'):
        for file in files:
            file_path = os.path.join(root, file)
            df = pd.read_csv(file_path, index_col=0)
            if 'OLS' in direc:
                df.index = pd.date_range(start="1991-1", periods=len(df), freq='M')
            d[file.split('-')[0]] = pd.concat([d[file.split('-')[0]], df['timeaccuracy_60']], axis=1)
    for com in coms:
        d[com] = d[com].mean(axis=1)
    agg = pd.concat([d[com] for com in coms], axis=1)
    agg.columns = [com for com in coms]
    name = direc.split("/")
    name = name[0] + name[-1]
    agg.to_csv(f'data/output/measures/overall/timeaccuracy/60{name}.csv')


def timeacc_plot(com):
    """
    Plot time accuracy for each commodity and model.
    :param com: commodity to plot time accuracy
    """
    dataframe = pd.read_csv("data\input\cmo.csv", index_col=0)
    if com == 'allcoms':
        p = dataframe.mean(axis=1).loc['1992M12':'2023M01']
        p.to_csv('data/input/poep.csv')
        return 0
    else:
        p = dataframe[com].loc['1992M12':'2023M01']
    predacc = pd.read_csv(f'data/output/measures/overall/accuracy/time-accuracies/timeacc24/{com}.csv',
                          index_col=0).dropna()

    dates = pd.date_range('1992-12', '2023-02', freq='M')
    predacc.index = dates
    p.index = dates
    fig, ax1 = plt.subplots()

    for col in predacc.columns:
        if 'RF' in col:
            acc_smooth = savgol_filter(predacc[col], window_length=7, polyorder=2)  # Apply smoothing
            if 'non-binary' in col:
                lab = 'Non-Binary'
            else:
                lab = 'Binary'
            ax1.plot(dates, acc_smooth, label=lab)

    ax1.set_ylabel('24-Month Accuracy %')
    ax1.axhline(y=50, color='grey', linestyle='--')

    ax1.set_xlim([dates[0], dates[-1]])
    ax1.set_xticks(['1993', '1998', '2003', '2008', '2013', '2018', '2023'])
    ax1.set_ylim([40, 75])

    ax2 = ax1.twinx()

    ax2.plot(dates, savgol_filter(p, window_length=7, polyorder=2), color='black')
    ax2.set_ylabel('Price', color='black')
    ax2.tick_params(axis='y', labelcolor='black')

    ax1.legend(loc='upper left')

    plt.show()


def timeacc_avg():
    """
    Obtains the average time accuracy over commodities.
    """
    files = os.listdir('data/output/measures/overall/accuracy/time-accuracies/timeacc24')
    d = {'OLS-binary.csv': pd.DataFrame(), 'OLS-non-binary.csv': pd.DataFrame(), 'RF-binary.csv': pd.DataFrame(),
         'RF-non-binary.csv': pd.DataFrame(), 'SVM-binary.csv': pd.DataFrame(), 'SVM-non-binary.csv': pd.DataFrame()}
    for file in files:
        avgacc = pd.read_csv(f'data/output/measures/overall/accuracy/time-accuracies/timeacc24/{file}', index_col=0)
        for com in avgacc.columns:
            d[com][file] = avgacc[com]
    all = pd.DataFrame()
    for model in d.keys():
        all[model] = d[model].mean(axis=1)
    all.to_csv('data/output/measures/overall/timeacc24/allcoms.csv')


def avg_accs(directory):
    """
    Obtains average accuracy over a directory.
    :param directory: directory to obtain average accuracy over.
    """
    acc_dic = {}
    files = os.listdir(f'data\\output\\measures\\{directory}')
    print(files)
    for file in files:
        df = pd.read_csv(f'data\\output\\measures\\{directory}\\{file}', index_col=0)
        acc_dic[file] = df.mean(axis=1)[1]
    dicdf = pd.DataFrame(acc_dic, index=['A'])
    print(dicdf)
    print(acc_dic)
    dicdf.to_csv(f'data\\output\\measures\\overall\\{directory}.csv', index=False)


def avg_com(directory):
    """
    Obtains average accuracy of each commodity over a directory.
    :param directory: directory to obtain average accuracy over
    """
    acc_dic = {}
    for com in coms:
        acc_dic[com] = []
    files = os.listdir(f'data\\output\\measures\\{directory}')
    for file in files:
        df = pd.read_csv(f'data\\output\\measures\\{directory}\\{file}', index_col=0)
        print(df)
        for com in coms:
            acc_dic[com].append(df.loc['Training', com])
    for com in coms:
        acc_dic[com] = np.average(acc_dic[com])
    accdf = pd.DataFrame(acc_dic, index=['A'])
    accdf.to_csv(f'data\\output\\measures\\overall\\{directory}-coms.csv', index=False)


def avg_feat(directory):
    """
    Obtains average feature importances over a model.
    :param directory: directory of model
    """
    imp_dic = defaultdict(lambda: [])
    files = os.listdir(f'data\\RF\\feature-importances\\{directory}')
    for file in files:
        subfiles = os.listdir(f'data\\RF\\feature-importances\\{directory}\\{file}')
        for subfile in subfiles:
            df = pd.read_csv(f'data\\RF\\feature-importances\\{directory}\\{file}\\{subfile}', index_col=0)
            for ti, ti_series in df.items():
                imp_dic[ti].extend(ti_series.values)
    for ti in imp_dic:
        imp_dic[ti] = np.average(imp_dic[ti])
    impdf = pd.DataFrame(imp_dic, index=['feat_imp'])
    impdf.to_csv(f'data\\output\\measures\\RF\\featimps\\{directory}.csv', index=False)


def normalize_ti(bin):
    """
    Normalizes the values of technical indicators.
    :param bin: encoding
    """
    for dirpath, dirnames, filenames in os.walk(f'non-standard-ti\\technical-indicators\\{bin}'):
        for filename in filenames:
            df = pd.read_csv(f'{dirpath}\\{filename}', index_col=0)
            for col in df.columns:
                std_dev = df[col].std()
                df[col] = df[col] / std_dev
            df.to_csv(f'technical-indicators\\{bin}\\{filename}')


def sharpe(direc):
    """
    Calculates the Sharpe ratio using a simple long-only strategy and using the 3-month T-bill.
    :param direc: model to be considered
    :return: Sharpe ratio
    """
    prices = pd.read_csv('data/input/cmo.csv', index_col=0).loc['1991M01': '2023M01']
    predictions = pd.read_csv(direc, index_col=0)
    if 'OLS' in direc:
        predictions = predictions.applymap(lambda x: 1 if x > 0 else 0)
    predictions.index = prices.index

    risk_free_rate = pd.read_csv('data/input/TB3MS.csv', index_col=0)
    risk_free_rate = risk_free_rate / 100
    risk_free_rate.index = prices.index
    returns = pd.DataFrame()

    for com in full_name.values():
        returns[com] = predictions[com] * prices[com].pct_change().shift(-1)
    returns = returns.dropna()

    returns['total_return'] = returns.sum(axis=1)
    excess = returns['total_return'] - risk_free_rate['TB3MS']
    returns['excess_returns'] = excess.dropna()

    return 'Sharpe Ratio:', np.sqrt(12) * (returns["excess_returns"].mean() / returns["excess_returns"].std())


def accuracies():
    """
    Produce Excel file with out-of-sample and training accuracies.
    """
    files = os.listdir('data/output/measures/RF/non-binary')
    oos = pd.DataFrame()
    training = pd.DataFrame()
    for file in files:
        first = file.strip('.csv').split('-')
        acs = pd.read_csv(f'data/output/measures/RF/non-binary/{file}', index_col=0)
        oos.loc[first[0], first[1]] = acs.loc['Hold-out'].mean()
        training.loc[first[0], first[1]] = acs.loc['Training'].mean()
    oos.to_excel('data/output/measures/overall/RFnonbinaryoosaccs2.xlsx')
    training.to_excel('data/output/measures/overall/RFnonbinarytrainingaccs2.xlsx')

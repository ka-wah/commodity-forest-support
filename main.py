import time
import Random_Forest as rf
import Support_Vector as svm
import Least_Squares as ols
import Binary_Indicators as bin
import Nonbinary_Indicators as nbin
import Evaluation as ev

"""
In this main file, you can run the experiment.

"""

begin = time.time()

bins = ['binary', 'non-binary']
rules = ["filtering", "momentum", "moving-average", "oscillator", "support-resistance"]


def create_ti():
    """
    Creates the technical indicators to be used for the predictions.

    """
    bin.momentum()
    bin.filtering()
    bin.moving_average()
    bin.oscillator()
    bin.support_resistance()
    nbin.momentum()
    nbin.filtering()
    nbin.moving_average()
    nbin.oscillator()
    nbin.support_resistance()
    ev.normalize_ti('binary')
    ev.normalize_ti('non-binary')


def run_ols():
    """
    Creates all OLS predictions.
    """
    for bin in bins:
        for rule in rules:
            ols.regress(bin, rule)


def run_rf():
    """
    Creates all Random Forest predictions.
    """
    i = 0
    for feat in [6, 12, 18]:
        for tres in [100, 250, 500]:
            start = time.time()
            rf.classify('binary', feat, tres)
            elaps = time.time() - start
            i += 1
            print(f'{100 * i / 18}%')
            print(f'remaining time: {(18 - i) * elaps / 60} mins')

            start = time.time()
            rf.classify('non-binary', feat, tres)
            elaps = time.time() - start
            i += 1
            print(f'{100 * i / 18}%')
            print(f'remaining time: {(18 - i) * elaps / 60} mins')
    rf.select_features('binary')
    rf.select_features('non-binary')


def run_svm():
    """
    Creates all SVM predictions.
    """
    i = 0
    for c in [1, 10, 100]:
        for gamma in [2, 2.5, 3, 3.5, 4]:
            start = time.time()
            svm.classify('non-binary', c, gamma)
            elaps = time.time() - start
            i += 1
            print(f'{100 * i / 30}%')
            print(f'remaining time: {(30 - i) * elaps / 60} mins')

            start = time.time()
            svm.classify('binary', c, gamma)
            elaps = time.time() - start
            i += 1
            print(f'{100 * i / 30}%')
            print(f'remaining time: {(30 - i) * elaps / 60} mins')

create_ti()
run_ols()
run_rf()
run_svm()

print('elapsed time: ', time.time() - begin, 'seconds')

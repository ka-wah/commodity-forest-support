import os

import numpy as np
import pandas as pd

from Data_Preprocess import get_csv

cmo = get_csv()
cmo = cmo[265:]
lookbacks = [1, 3, 6, 9, 12]

"""
This file is used to generate binary technical indicators.

"""

def momentum():
    for k in lookbacks:
        df = pd.DataFrame(index=cmo.index, columns=cmo.columns)
        for commodity in cmo.columns:
            prices = cmo[commodity]
            for date in range(k, len(prices)):
                current_price = prices[date]
                past_price = prices[date - k]
                df.loc[prices.index[date], commodity] = 1 if current_price >= past_price else 0
        output_file = os.path.join("technical-indicators\\binary\\momentum", f'mom{k}.csv')
        df.to_csv(output_file)


def filtering():
    thresholds = [5, 10]
    for k in lookbacks:
        for n in thresholds:
            df_buy = pd.DataFrame(index=cmo.index, columns=cmo.columns)
            df_sell = pd.DataFrame(index=cmo.index, columns=cmo.columns)
            for commodity in cmo.columns:
                prices = cmo[commodity]
                for date in range(k, len(prices)):
                    current_price = prices[date]
                    minimum = min(prices[date - k: date])
                    maximum = max(prices[date - k: date])
                    buy = 1 if current_price >= (1 + n / 100) * minimum else 0
                    sell = 1 if current_price <= (1 - n / 100) * maximum else 0
                    df_buy.loc[prices.index[date], commodity] = buy
                    df_sell.loc[prices.index[date], commodity] = sell
            output_file_buy = os.path.join("technical-indicators\\binary\\filtering", f'flt{k}_{n}_buy.csv')
            output_file_sell = os.path.join("technical-indicators\\binary\\filtering", f'flt{k}_{n}_sell.csv')
            df_buy.to_csv(output_file_buy)
            df_sell.to_csv(output_file_sell)


def moving_average():
    for s in lookbacks:
        for l in lookbacks:
            if s < l:
                df = pd.DataFrame(index=cmo.index, columns=cmo.columns)
                for commodity in cmo.columns:
                    prices = cmo[commodity]
                    for date in range(l, len(prices)):  # klopt dit?? of is het range(s
                        mas = np.average(prices[date - s + 1:date + 1])
                        mal = np.average(prices[date - l + 1:date + 1])
                        df.loc[prices.index[date], commodity] = 1 if mas >= mal else 0
                output_file = os.path.join("technical-indicators\\binary\\moving-average", f'mva{s}_{l}.csv')
                df.to_csv(output_file)


def oscillator():
    thresholds = [5, 10]

    def up(k, prices, date):
        sum = 0
        for i in range(0, k):
            sum += ((prices[date - i] - prices[date - i - 1]) if prices[date - i] - prices[date - i - 1] > 0 else 0)
        return sum

    def down(k, prices, date):
        sum = 0
        for i in range(0, k):
            sum += (abs(prices[date - i] - prices[date - i - 1]) if prices[date - i] - prices[date - i - 1] < 0 else 0)
        return sum

    for k in lookbacks:
        for n in thresholds:
            df_buy = pd.DataFrame(index=cmo.index, columns=cmo.columns)
            df_sell = pd.DataFrame(index=cmo.index, columns=cmo.columns)
            for commodity in cmo.columns:
                prices = cmo[commodity]
                for date in range(k, len(prices)):
                    rsi = 100 * up(k, prices, date) / (up(k, prices, date) + down(k,prices,date))
                    buy = 1 if rsi <= 50 + n else 0
                    sell = 1 if rsi >= 50 + n else 0
                    df_buy.loc[prices.index[date], commodity] = buy
                    df_sell.loc[prices.index[date], commodity] = sell
            output_file_buy = os.path.join("technical-indicators\\binary\\oscillator", f'osc{k}_{n}_buy.csv')
            output_file_sell = os.path.join("technical-indicators\\binary\\oscillator", f'osc{k}_{n}_sell.csv')
            df_buy.to_csv(output_file_buy)
            df_sell.to_csv(output_file_sell)


def support_resistance():
    thresholds = [1, 2, 3, 4, 5]
    for k in lookbacks:
        for n in thresholds:
            df_buy = pd.DataFrame(index=cmo.index, columns=cmo.columns)
            df_sell = pd.DataFrame(index=cmo.index, columns=cmo.columns)
            for commodity in cmo.columns:
                prices = cmo[commodity]
                for date in range(k, len(prices)):
                    current_price = prices[date]
                    minimum = min(prices[date - k: date]) # + 1?
                    maximum = max(prices[date - k: date])
                    buy = 1 if current_price >= (1 + n / 100) * maximum else 0
                    sell = 1 if current_price <= (1 - n / 100) * minimum else 0
                    df_buy.loc[prices.index[date], commodity] = buy
                    df_sell.loc[prices.index[date], commodity] = sell
            output_file_buy = os.path.join("technical-indicators\\binary\\support-resistance", f'sup{k}_{n}_buy.csv')
            output_file_sell = os.path.join("technical-indicators\\binary\\support-resistance", f'sup{k}_{n}_sell.csv')
            df_buy.to_csv(output_file_buy)
            df_sell.to_csv(output_file_sell)

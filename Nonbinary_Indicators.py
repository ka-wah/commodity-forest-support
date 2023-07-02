import os
import pandas as pd
import numpy as np

from Data_Preprocess import get_csv

cmo = get_csv()
cmo = cmo[265:]
lookbacks = [1, 3, 6, 9, 12]

"""
This file is used to generate non-binary technical indicators.

"""

def momentum():
    for k in lookbacks:
        df = pd.DataFrame(index=cmo.index, columns=cmo.columns)
        for commodity in cmo.columns:
            prices = cmo[commodity]
            for date in range(k, len(prices)):
                current_price = prices[date]
                past_price = prices[date - k]
                df.loc[prices.index[date], commodity] = np.log(current_price) - np.log(past_price)
        output_file = os.path.join("technical-indicators/non-binary\\momentum", f'mom{k}.csv')
        df.to_csv(output_file)


def filtering():
    for k in lookbacks:
        df_buy = pd.DataFrame(index=cmo.index, columns=cmo.columns)
        df_sell = pd.DataFrame(index=cmo.index, columns=cmo.columns)
        for commodity in cmo.columns:
            prices = cmo[commodity]
            for date in range(k, len(prices)):
                current_price = prices[date]
                minimum = min(prices[date- k: date])
                maximum = max(prices[date - k: date])
                buy = np.log(current_price/minimum)
                sell = np.log(maximum/current_price)
                df_buy.loc[prices.index[date], commodity] = buy
                df_sell.loc[prices.index[date], commodity] = sell
        output_file_buy = os.path.join("technical-indicators/non-binary\\filtering", f'flt{k}_buy.csv')
        output_file_sell = os.path.join("technical-indicators/non-binary\\filtering", f'flt{k}_sell.csv')
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
                        mas = sum(prices[date - s + 1:date + 1]) / s
                        mal = sum(prices[date - l + 1:date + 1]) / l
                        df.loc[prices.index[date], commodity] = np.log(mas) - np.log(mal)
                output_file = os.path.join("technical-indicators/non-binary\\moving-average", f'mva{s}_{l}.csv')
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
        df_buy = pd.DataFrame(index=cmo.index, columns=cmo.columns)
        for commodity in cmo.columns:
            prices = cmo[commodity]
            for date in range(k, len(prices)):
                rsi = 100 * up(k, prices, date) / (up(k, prices, date) + down(k,prices,date))
                df_buy.loc[prices.index[date], commodity] = rsi
        output_file_buy = os.path.join("technical-indicators\\non-binary\\oscillator", f'osc{k}.csv')
        df_buy.to_csv(output_file_buy)


def support_resistance():
    for k in lookbacks:
        df_buy = pd.DataFrame(index=cmo.index, columns=cmo.columns)
        df_sell = pd.DataFrame(index=cmo.index, columns=cmo.columns)
        for commodity in cmo.columns:
            prices = cmo[commodity]
            for date in range(k, len(prices)):
                current_price = prices[date]
                minimum = min(prices[date - k: date]) # + 1?
                maximum = max(prices[date - k: date])
                buy = np.log(current_price/maximum)
                sell = np.log(minimum/current_price)
                df_buy.loc[prices.index[date], commodity] = buy
                df_sell.loc[prices.index[date], commodity] = sell
        output_file_buy = os.path.join("technical-indicators/non-binary\\support-resistance", f'sup{k}_buy.csv')
        output_file_sell = os.path.join("technical-indicators/non-binary\\support-resistance", f'sup{k}_sell.csv')
        df_buy.to_csv(output_file_buy)
        df_sell.to_csv(output_file_sell)
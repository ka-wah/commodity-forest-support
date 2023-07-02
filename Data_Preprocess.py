import numpy as np
import pandas as pd
import os
from datetime import datetime

"""
This file is used to preprocess the data.

"""

def filenamer():
    directory = os.path.dirname(os.path.abspath(__file__))
    timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
    return directory, timestamp

def xlsx_to_csv(name="output", input_file="data\cmo-monthly.xlsx"):
    data_frame = pd.read_excel(input_file, sheet_name="Monthly Indices", skiprows=9, parse_dates=[0], index_col=0)

    column_names = ["wbcpi", "energy", "non-energy", "agriculture", "beverages", "food", "oils-meals", "grains",
                    "other-food", "raw-materials", "timber", "other-raw", "fertilizers", "metals-minerals",
                    "base-metals", "precious-metals"]

    data_frame.columns = column_names

    columns_delete = ["wbcpi", "oils-meals", "grains", "other-food", "timber", "other-raw", "fertilizers",
                      "base-metals"]

    data_frame.drop(columns=columns_delete, inplace=True)

    output_file = os.path.join(filenamer()[0], f'{name}_{filenamer()[1]}.csv')

    data_frame.columns = [colname for colname in column_names if colname not in columns_delete]
    data_frame.to_csv(output_file)


def get_csv():
    dataframe = pd.read_csv("data\input\cmo.csv", index_col=0)
    return dataframe

def csv_to_log(dataframe=get_csv()):
    log_returns = {}
    for column in dataframe.columns:
        current_price = np.log(dataframe[column])
        previous_price = np.log(dataframe[column].shift(1))
        log_returns[column] = current_price - previous_price
    new = pd.DataFrame(log_returns)

    output_file = os.path.join(filenamer()[0], f'logcmo_{filenamer()[1]}.csv')

    new.to_csv(output_file)
    return new

def csv_to_difference(dataframe=get_csv()):
    log_returns = {}
    for column in dataframe.columns:
        current_price = dataframe[column]
        previous_price = dataframe[column].shift(1)
        log_returns[column] = current_price - previous_price
    new = pd.DataFrame(log_returns)

    output_file = os.path.join('data', f'diffcmo.csv')

    new.to_csv(output_file)
    return new

def csv_to_differencepercent(dataframe=get_csv()):
    log_returns = {}
    for column in dataframe.columns:
        current_price = dataframe[column]
        previous_price = dataframe[column].shift(1)
        log_returns[column] = (current_price - previous_price)/previous_price
    new = pd.DataFrame(log_returns)

    output_file = os.path.join('data', f'diffpercentcmo.csv')

    new.to_csv(output_file)
    return new

def logcmo_to_binary():
    df = pd.read_csv("data\input\logcmo.csv", index_col=0)
    binary_df = pd.DataFrame(index=df.index, columns=df.columns)
    binary_df = df.applymap(lambda x: 1 if x > 0 else 0)
    output_file = os.path.join('data', f'binarycmo.csv')
    binary_df.to_csv(output_file)

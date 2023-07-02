import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

"""
This file is used to create plots, visualizing data.

"""

proper_name = {'energy': 'Energy', 'non-energy': 'Non-Energy', 'agriculture': 'Agriculture', 'beverages': 'Beverage',
               'food': 'Food', 'raw-materials': 'Raw Materials', 'metals-minerals': 'Metals & Minerals',
               'precious-metals': 'Precious Metals'}


def price():
    """
    Create a plot with all commodity prices over time.
    """
    prices = pd.read_csv('data/input/cmo.csv', index_col=0)
    prices = prices.loc['1981M01':'2023M01']
    prices.index = pd.date_range('1981-1', '2023-2', freq='M')

    for col in prices.columns:
        plt.plot(prices.index, prices[col], label=proper_name[col])

    plt.ylabel('Price')
    plt.grid(axis='y')

    begin_date = prices.index[0]
    end_date = prices.index[-1]

    plt.xlim(begin_date, end_date)

    plt.gca().spines['top'].set_visible(True)
    plt.gca().spines['right'].set_visible(True)
    plt.gca().spines['left'].set_visible(True)
    plt.gca().spines['bottom'].set_visible(True)

    plt.legend()

    plt.show()


def featimps():
    """
    Create a heatmap of feature importances.
    """
    avgs = pd.read_csv('data/output/measures/RF/featimps/binary.csv', index_col=0)

    avgs = avgs[['energy', 'nonenergy', 'agriculture', 'beverages', 'food', 'materials', 'minerals', 'metals']]
    avgs.columns = proper_name.values()

    df = avgs[avgs > 0.01].dropna()
    data = df.values
    data = np.clip(data, None, 0.03)

    plt.imshow(data.transpose(), cmap='PiYG', vmin=0.01, vmax=np.max(data))

    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')

    plt.xticks(range(len(df.index)), \
               df.index, rotation=90)
    plt.yticks(range(len(df.columns)),
               df.columns)

    divider = make_axes_locatable(plt.gca())

    cax = divider.append_axes("top", size="5%", pad=0.3)

    cbar = plt.colorbar(cax=cax, orientation="horizontal", ticks=np.linspace(0.01, np.max(data), num=5))

    plt.show()

# import warnings
# import itertools
# import pandas as pd
# import numpy as np
# import statsmodels.api as sm
# import matplotlib.pyplot as plt
# plt.style.use('fivethirtyeight')

# fig = plt.figure(1)
# plt.plot(yvalues,label='Actual Values')
# plt.plot(yhat,label='Forecast Values',linestyle='--')
# plt.title('Simple Moving Averages')
# plt.grid(True)plt.legend(loc='upper center',
# fancybox=True, shadow=True, ncol=2)

# load and plot dataset
from pandas import read_csv
from pandas import datetime
# from matplotlib import pyplot
from plotly import plotly
# load dataset
def parser(x):
    return x
    # return datetime.strptime("190"+x, "%Y-%m")

series = read_csv("./data/realTweets/Twitter_volume_AAPL.csv", header=0, parse_dates=[0], index_col=0, squeeze=True, date_parser=parser)
series2 = read_csv("./data/realTweets/Twitter_volume_UPS.csv", header=0, parse_dates=[0], index_col=0, squeeze=True, date_parser=parser)
# summarize first few rows
print(series.head())
# line plot
# series.plot()


# pyplot.style.use('fivethirtyeight')

# fig = pyplot.figure(1)
# pyplot.plot(series,label='Actual Values')
# pyplot.plot(series2,label='Forecast Values',linestyle='--')
# pyplot.title('Simple Moving Averages')
# pyplot.grid(True)
# pyplot.legend(loc='upper center',fancybox=True, shadow=True, ncol=2)

# pyplot.show()


data = [dict(
        visible = False,
        line=dict(color='00CED1', width=6),
        name = '𝜈 = '+str(step),
        x = np.arange(0,10,0.01),
        y = np.sin(step*np.arange(0,10,0.01))) for step in np.arange(0,5,0.1)]
data[10]['visible'] = True

steps = []
for i in range(len(data)):
    step = dict(
        method = 'restyle',
        args = ['visible', [False] * len(data)],
    )
    step['args'][1][i] = True # Toggle i'th trace to "visible"
    steps.append(step)

sliders = [dict(
    active = 10,
    currentvalue = {"prefix": "Frequency: "},
    pad = {"t": 50},
    steps = steps
)]

layout = dict(sliders=sliders)
fig = dict(data=data, layout=layout)

plotly.iplot(fig, filename='Sine Wave Slider')
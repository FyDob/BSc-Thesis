# Script to explore model performance
# - How many models >= 50%?
# - How many models < 50%?
# - SRNN performance (regardless of experiment, corpus)
# - LSTM performance (regardless of experiment, corpus)
# - GRU performance (regardless of experiment, corpus)
# - high performance (regardless of experiment, network)
# - base performance (regardless of experiment, network)
# - low performance (regardless of experiment, network)

import os
import pandas as pd

infile = os.path.join('..', 'results', 'results.csv')
results = pd.read_csv(infile)

# print(results.head())
# print(results.describe())

# LRD = results.loc[results.experiment == 'LRD']
# print(LRD.describe())

# ND = results.loc[results.experiment == 'ND']
# print(ND.describe())

SRNN = results.loc[results.network == 'SRNN']#.loc[results.experiment == 'LRD']
print("SRNN")
print(SRNN.describe())

LSTM = results.loc[results.network == 'LSTM']#.loc[results.experiment == 'LRD']
print("LSTM")
print(LSTM.describe())

GRU = results.loc[results.network == 'GRU']#.loc[results.experiment == 'LRD']
print("GRU")
print(GRU.describe())

high = results.loc[results.corpus == 'high']
print("high")
print(high.describe())

base = results.loc[results.corpus == 'base']
print("base")
print(base.describe())

low = results.loc[results.corpus == 'low']
print("low")
print(low.describe())
import pandas as pd
import numpy as np

'''
Thresholds:
with 0.3 : 112,131
with 0.2 : 106,137
'''

i = 0
j = 0
data = pd.read_csv('./Data.csv',header=None)
x = data.iloc[1:,1:].values
while j < x[:,0].size:
    for i in range(0,x[0,:].size):
        if x[j][i] < 112 or x[j][i] > 131:
                x[j][i] = 0
    j = j + 1

np.savetxt("nData.csv", x, delimiter=",")

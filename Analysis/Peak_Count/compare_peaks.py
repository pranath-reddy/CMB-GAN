import h5py
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from peakdetect import peakdetect

data1 = pd.read_csv('./matches128.csv',header=None)
y1 = data1.iloc[:,:].values

data2 = pd.read_csv('./Test128.csv',header=None)
y2 = data2.iloc[:,:].values

peaks1 = []
peaks2 = []

for i in range(100):
    peaksdet1 = peakdetect(y1[i],lookahead=100)
    peaksdet2 = peakdetect(y2[i],lookahead=100)
    peaks1.append(np.shape(peaksdet1[0])[0])
    peaks2.append(np.shape(peaksdet2[0])[0])

'''
plt.plot(peaks1,label="CAMB")
plt.plot(peaks2,label="GAN")
plt.legend()
plt.show()
'''
peakstot1 = []
peakstot2 = []
for i in range(100):
    p1 = 0
    p2 = 0
    for j in range(i+1):
        p1 = p1 + peaks1[j]
        p2 = p2 + peaks2[j]
    peakstot1.append(p1)
    peakstot2.append(p2)

plt.plot(peakstot1,label="CAMB")
plt.plot(peakstot2,label="GAN")
plt.legend()
plt.show()
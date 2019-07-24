import h5py
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data1 = pd.read_csv('./matches128.csv',header=None)
y1 = data1.iloc[:,:].values

data2 = pd.read_csv('./Test128.csv',header=None)
y2 = data2.iloc[:,:].values

std1 = []
std2 = []

for i in range(100):
    std1.append(np.std(y1[i]))
    std2.append(np.std(y2[i]))

plt.plot(std1,label="CAMB")
plt.plot(std2,label="GAN")
plt.legend()
plt.show()
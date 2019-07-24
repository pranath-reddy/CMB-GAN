import h5py
import pickle
import numpy as np
import pandas as pd

with h5py.File('Data.h5', 'r') as hf:
    x = hf['data'][:]

with open('preds128.pkl', 'rb') as f:
    preds = pickle.load(f)

x = x.tolist()
camb = []
for i in range(100):
    camb.append(x[preds[i]*12])

np.savetxt("camb128.csv", camb, delimiter=",")

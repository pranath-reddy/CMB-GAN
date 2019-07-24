import h5py
import pickle
import numpy as np
import pandas as pd

labels = pd.read_csv('./Test128.csv',header=None)
x = labels.iloc[:,:].values

with open('preds128.pkl', 'rb') as f:
    preds = pickle.load(f)

x = x.tolist()
camb = []
for i in range(100):
    camb.append(x[preds[i]])

np.savetxt("matches128.csv", camb, delimiter=",")

print(np.shape(camb))


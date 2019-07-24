import h5py
import numpy as np

with h5py.File('Data.h5', 'r') as hf:
    x = hf['data'][:]

'''
for m in range(60000):
    for n in range(4096):
        x[m][n] = x[m][n] / 255
'''
x = x / 255
with h5py.File('Data2.h5', 'w') as hf:
    hf.create_dataset("data",  data=x)
    



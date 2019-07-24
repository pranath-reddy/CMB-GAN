import h5py
import numpy as np

with h5py.File('Data2.h5', 'r') as hf:
    x = hf['data'][:]

print(x)
print(x.shape)
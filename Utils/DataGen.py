import h5py
import scipy.misc
import numpy as np

with h5py.File('Data.h5', 'r') as hf:
    x = hf['data'][:]

for i in range(60000):
    out = "image" + str(i) + ".png"
    scipy.misc.imsave(out, np.array(x[i]).reshape(64,64))



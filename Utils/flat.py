import cv2
import numpy as np
import os

mypath = './Test128'
files = [os.path.join(mypath, f) for f in os.listdir(mypath) if f.endswith(".jpg")]

numOfImages = len(files)

final = np.zeros(shape=(numOfImages,4096))

i=0

for images in files:
    image = cv2.imread(images,0) 
    image = cv2.resize(image,(64,64))
    data = np.array(image)
    flattened = data.flatten()
    final[i] = flattened
    i = i+1

s = np.shape(final)
print(str(s))
np.savetxt("Test128.csv", final, delimiter=",")
print(final)

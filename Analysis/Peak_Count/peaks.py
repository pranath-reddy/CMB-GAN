import numpy as np
import cv2
from peakdetect import peakdetect
image = cv2.imread('camb2.png')
gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
holder = []
i = 0
j = 0

while j < gray[:,0].size:
    for i in range(0,gray[0,:].size):
        if gray[j][i] != 255:
            holder.append(gray[j][i])
    j = j + 1

holder = np.array(holder)
peaks = peakdetect(holder,lookahead=100)
print(peaks)
print(np.shape(peaks[0])[0])
import cv2
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import numpy as np

image = cv2.imread('test.png')
orgray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

j = 0
while j < gray[:,0].size:
    for i in range(0,gray[0,:].size):
        if gray[j][i] != 255:
            if gray[j][i] < 112 or gray[j][i] > 131:
                    gray[j][i] = 0
    j = j + 1

cv2.imwrite("original.jpg",orgray)
cv2.imwrite("normal.jpg",gray)

import cv2
import matplotlib.pyplot as plt
import numpy as np
 
image = cv2.imread('Test2.jpg')
gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)


holder = []
i = 0
j = 0

while j < gray[:,0].size:
    for i in range(0,gray[0,:].size):
        if gray[j][i] != 255:
            holder.append(gray[j][i])
    j = j + 1

holder = np.asarray(holder)
holder_avg = np.average(holder)
holder_std = np.std(holder)
holder2 = np.random.normal(loc=holder_avg,scale=holder_std,size=holder.shape)

plt.hist(holder, bins='auto', label="Actual values")
plt.hist(holder2, bins='auto',alpha = 0.5, lw=3, label="Gaussian Fit")
plt.title("Distribution of pixel intensity")
plt.xlabel("Intensity")
plt.ylabel("Population")
plt.legend()
plt.show()




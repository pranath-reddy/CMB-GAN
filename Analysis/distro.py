import cv2
import numpy as np
from scipy.stats import norm
 
image = cv2.imread('test.png')
gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)


holder = []
i = 0
j = 0
k = 0

while j < gray[:,0].size:
    for i in range(0,gray[0,:].size):
        if gray[j][i] != 255:
            holder.append(gray[j][i])
    j = j + 1

holder = np.asarray(holder)
holder_avg = np.average(holder)
holder_std = np.std(holder)
holder2 = np.random.normal(loc=holder_avg,scale=holder_std,size=holder.shape)


fixed = []
threshold = 0.2 
size = holder2.shape[0]
while k < size-1:
    if norm.cdf(holder2[k],holder_avg,holder_std) > threshold and norm.cdf(holder2[k],holder_avg,holder_std) < 1-threshold:
        fixed.append(holder2[k])
    k = k + 1

print(np.amin(fixed))
print(np.amax(fixed))













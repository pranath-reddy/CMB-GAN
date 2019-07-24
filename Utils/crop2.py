'''
Authors : Amit Mishra, Pranath Reddy

Crops the generated maps after running visualize.py 
'''

print("  _________ _____________  (_)____")
print("/ ___/ __ \/ ___/ __ `__ \/ / ___/")
print("/ /__/ /_/ (__  ) / / / / / / /__  ")
print("\___/\____/____/_/ /_/ /_/_/\___/ ")

print("A set of deep learning experiments on Cosmic Microwave Background Radiation Data")
print("CROPS SKY MAP IMAGES ALONG EQUATOR INTO 64X64PX IMAGES (Outputs in PNG format.)\n")

import cv2
import os

if not os.path.exists("./cropped_files"):
    os.makedirs("./cropped_files")

mypath = './image_files'
files = [os.path.join(mypath, f) for f in os.listdir(mypath) if f.endswith(".png")]
i = 1

for images in files:

    img = cv2.imread(images)
    y=150
    x=376

    print("Starting to crop image : "+str(i)+"/"+str(len(files)))
    
    for j in range(1):

        croppedImgName = os.path.basename(images) + "-" + str(j) + ".png"
        crop_img = img[y:y+256, x:x+256]
        os.chdir('./cropped_files')
        cv2.imwrite(croppedImgName,crop_img)
        x=x+256
        os.chdir('../')
        print("Generated cropped image : "+str(j+1)+"/5")

    i += 1        

print("\nFinished")

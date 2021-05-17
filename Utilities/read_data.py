import os
import numpy as np
import cv2
import matplotlib.pyplot as plt

os.chdir("..")
data = np.load("data/lfw_grayscale_1000_20_055_flow.npy")


for j in range(4):
    for i in range(9):
        plt.subplot(330 + 1 + i)
        img = data[i + 9*j].reshape(50, 50)
        enhanced_img = cv2.resize(img, (100,100), cv2.INTER_AREA)
        plt.imshow(enhanced_img, cmap=plt.get_cmap('gray'))
    plt.show()
import os
import numpy as np
import cv2

os.chdir("..")

directory_in_str = os.path.abspath(os.curdir) + "\\animals\\dogs"

directory = os.fsencode(directory_in_str)
images_np = list()
for file in os.listdir(directory):
    filename = os.fsdecode(file)
    img = cv2.imread("animals/dogs/" + filename, 0) # grayscale
    # img = cv2.imread("animals/dogs/" + filename)
    # rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)   # BGR -> RGB
    cv2.imshow('image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    images_np.append(img)

images_np = np.array(images_np)
os.chdir("..")
np.save('dogs_images_grayscale.npy', images_np)

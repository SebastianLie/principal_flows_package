import os
import numpy as np
import cv2

os.chdir("..")
directory_in_str = os.path.abspath(os.curdir) + "\\animals\\dogs"

directory = os.fsencode(directory_in_str)
images = list()
for file in os.listdir(directory):
    filename = os.fsdecode(file)
    img = cv2.imread("animals/dogs/" + filename, 0) # grayscale
    # img = cv2.imread("animals/dogs/" + filename)
    # rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)   # BGR -> RGB
    images.append(np.array(img))

'''
cv2.imshow('image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
'''
images_np = np.vstack(images)
print(type(images_np[0]))
print(images_np[0].shape)
print(images_np.shape)
np.save(os.path.abspath(os.curdir)+'\\dogs_images_grayscale.npy', images_np)

imagesloaded = np.load(os.path.abspath(os.curdir)+'\\dogs_images_grayscale.npy')


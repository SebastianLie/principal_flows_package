import os
import numpy as np
import cv2

os.chdir("..")
directory_in_str = os.path.abspath(os.curdir) + "\\cartoonset10k"

directory = os.fsencode(directory_in_str)
images = list()
i = 1
for file in os.listdir(directory):
    filename = os.fsdecode(file)
    if filename.endswith("png"):
        img = cv2.imread("cartoonset10k/" + filename, 0)
        sub_img = cv2.resize(img, (50, 50), interpolation = cv2.INTER_AREA)
        if i == 1:
            cv2.imshow('image', sub_img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            i += 1
            print(sub_img.shape)
        # img = cv2.imread("animals/dogs/" + filename, 0)
        # rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)   # BGR -> RGB
        
        images.append(sub_img)

images_np = np.array(images)
print(images_np[0].shape)
print(images_np.shape)
np.save(os.path.abspath(os.curdir)+'\\data\\cartoon_faces_grayscale_50.npy', images_np)

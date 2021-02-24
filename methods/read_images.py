import os
import numpy as np
import cv2

directory_in_str = os.path.abspath(os.curdir) + "\\cartoonset10k"

directory = os.fsencode(directory_in_str)
images = list()
i = 1
for file in os.listdir(directory):
    filename = os.fsdecode(file)
    if filename.endswith("png"):
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)   # BGR -> RGB        
        if i == 1:
            cv2.imshow('image', img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            i += 1
            print(img.shape)
        # img = cv2.imread("animals/dogs/" + filename, 0)
        # rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)   # BGR -> RGB
        
        images.append(img)

images_np = np.array(images)
print(images_np[0].shape)
print(images_np.shape)
np.save(os.path.abspath(os.curdir)+'\\cartoon_faces_color.npy', images_np)

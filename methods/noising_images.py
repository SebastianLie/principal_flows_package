import os
import shutil
import numpy as np
import cv2
from skimage.measure import block_reduce


# Generates 50 images based on image in /bw_images
# Then saves all noised and original image in noised_data

# set drive to root, above /methods
os.chdir("..")

# make new folder to put new photos in
new_folder = os.path.abspath(os.curdir) + "\\noisy_images"
if not os.path.isdir(new_folder):
    os.makedirs(new_folder)
else:
    # because os.rmdir() only works if the directory is empty
    # shutil.rmtree(new_folder) will remove directory + contents
    shutil.rmtree(new_folder) 
    os.makedirs(new_folder)


# read in image in original folder, write original to new folder
original_img = cv2.imread('bw_images/old_photo_3.jpg', 0)
# downsample
# block_reduce(image, block_size=(3, 3), func=np.mean)
'''
cv2.imshow("img", original_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
'''
# cv2.imwrite('noisy_images/new_image_0.png', original_img)

# function to randomly add some kind of noise
# set seed first
np.random.seed(361)

def add_noise(image):
    '''
    Handles 4 different kinds of noise, and chooses one randomly.
    '''
    noise_type = np.random.randint(1,3, size=1)[0]
    # print(noise_type)
    noisy = image
    if noise_type == 1:
        # gaussian noise 
        noise_intensity = np.random.randint(100, 400, size=1)[0]
        row, col = image.shape
        gauss = np.random.normal(0, noise_intensity, (row, col))
        gauss = gauss.reshape(row, col)
        noisy = image + gauss

    elif noise_type == 2:
        # blurring
        blur_intensity = np.random.randint(10, 50, size=1)[0]
        noisy = cv2.blur(image, (blur_intensity, blur_intensity))
   
    elif noise_type == 3:
        # speckle noise
        row, col = image.shape
        speckle = np.random.randn(row, col)
        speckle = speckle.reshape(row, col)        
        noisy = image + image * speckle
        

    elif noise_type == 4:
        # poisson noise
        noise_mask = np.random.poisson(image)
        noisy = image + noise_mask
 
    new_img = noisy
    return new_img

# loop to create, then write, images to new folder
for i in range(1, 51):
    img_name = "noisy_images/new_image_" + str(i) + ".png"
    noised_img = add_noise(original_img)
    '''
    cv2.imshow("img", noised_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    '''
    cv2.imwrite(img_name, noised_img)


import numpy as np
import cv2
import matplotlib.pyplot as plt

# fix path for sys
import os
import sys

# very important to load methods
os.chdir('.') # go to root dir
sys.path.append(os.getcwd() + '\\Methods')

from common_methods_sphere import put_on_sphere
from principal_flow_main import choose_h_gaussian, principal_flow, choose_h_binary
from principal_boundary_flows import principal_boundary
from centroid_finder import sphere_centroid_finder_vecs

SAMPLES = 2000

os.chdir('..') # navigate to folder above root to get to data
data = np.load('data\\processed_data\\lfw_created\\lfw_grayscale_75.npy') # shape = (13233, 75, 75)

cv2.imshow('image', data[0])
cv2.waitKey(0)
cv2.destroyAllWindows()

# Reshaping Image Array
m = data.shape[1]
n = data.shape[2]

image_vector_size = m*n
train_X = data.reshape(data.shape[0], image_vector_size)

# Sample from train_X #
np.random.seed(666)
train_samples = np.random.choice(train_X.shape[0], size=SAMPLES)
sampled_X = train_X[train_samples]

# Sampled Images #

for i in range(9):
    plt.subplot(330 + 1 + i)
    plt.imshow(sampled_X[i].reshape(m, n), cmap=plt.get_cmap('gray'))
plt.show()


sampled_X_on_sphere = put_on_sphere(sampled_X)
# print(sampled_X_on_sphere[0])

# Find centroid of data #
final_p = sphere_centroid_finder_vecs(sampled_X_on_sphere, sampled_X.shape[1], 0.05, 0.01, max_iter=100)
# print(final_p)
final_p_img = final_p.reshape(m, n)
plt.imshow(final_p_img, cmap=plt.get_cmap('gray'))
plt.show()

# Find principal flow and display first 27 images of flow obtained #
h = choose_h_binary(sampled_X_on_sphere, final_p, 25) # needs to be very high!
radius = choose_h_binary(sampled_X_on_sphere, final_p, 30)
curve = principal_flow(sampled_X_on_sphere, sampled_X.shape[1], 0.055, h, \
    start_point=final_p, kernel_type="binary", max_iter=20)

# idea: upsample images
np.save(os.path.abspath(os.curdir)+'\\data\\lfw_grayscale_75_1000_25_055_flow.npy', curve)
#curve = np.load(os.path.abspath(os.curdir)+'\\data\\lfw_grayscale_75_1000_25_055_flow.npy')
print("curve")
for j in range(4):
    for i in range(9):
        #plt.subplot(330 + 1 + i)
        img = curve[i + 9*j].reshape(m, n)
        enhanced_img = cv2.resize(img, (150,150), cv2.INTER_AREA)
        plt.imshow(enhanced_img, cmap=plt.get_cmap('gray'))
        plt.savefig("lfw_pics/{}.".format(i + 9*j))
    #plt.show()

'''
upper, curve, lower = principal_boundary(sampled_X_on_sphere, sampled_X.shape[1], 0.02, h, radius, \
    start_point=final_p, kernel_type="binary", max_iter=30)

print("upper")
for j in range(6):
    for i in range(9):
        plt.subplot(330 + 1 + i)
        plt.imshow(upper[i + 9*j].reshape(m, n), cmap=plt.get_cmap('gray'))
    plt.show()

print("curve")
for j in range(6):
    for i in range(9):
        plt.subplot(330 + 1 + i)
        plt.imshow(curve[i + 9*j].reshape(m, n), cmap=plt.get_cmap('gray'))
    plt.show()
print("lower")
for j in range(6):
    for i in range(9):
        plt.subplot(330 + 1 + i)
        plt.imshow(lower[i + 9*j].reshape(m, n), cmap=plt.get_cmap('gray'))
    plt.show()
'''

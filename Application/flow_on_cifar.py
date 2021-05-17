import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import cifar10

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

'''
airplane										
automobile										
bird										
cat										
deer										
dog										
frog										
horse										
ship										
truck
'''


# Program to run principal flow on MNIST data.
# Choose the digit,and the number of samples for the data
# in the constants below.

# Constants #

DIGIT = 3
SAMPLES = 2000

# Data #

'''
(train_X, train_y), (test_X, test_y) = cifar10.load_data()
train_y = train_y.reshape(train_y.shape[0])
m = train_X.shape[1]
n = train_X.shape[2]

print(type(train_X))
print(train_X.shape)
print(train_y.shape)


# Sample from train_X #

train_filter = np.where(train_y == DIGIT)

train_X, train_y = train_X[train_filter], train_y[train_filter]
print(train_X.shape)

train_samples = np.random.choice(train_X.shape[0], size=SAMPLES)
sampled_X = train_X[train_samples]
print(sampled_X.shape)
for i in range(9):
    plt.subplot(330 + 1 + i)
    plt.imshow(sampled_X[i])
plt.show()

# Reshape array to easily separate RGB #
# sampled_X.shape = (...., m, n, 3) -> (3,....,m, n)
sampled_X = sampled_X.reshape(3, sampled_X.shape[0], sampled_X.shape[1], sampled_X.shape[2])

rgb = []
for i in range(3):
    X = sampled_X[i]
    print(X.shape)

    X = X.reshape(SAMPLES, m*n)
    sampled_X_on_sphere = put_on_sphere(X)
    # print(sampled_X_on_sphere[0])

    # Find centroid of data #
    final_p = sphere_centroid_finder_vecs(sampled_X_on_sphere, X.shape[1], 0.05, 0.01)
    # print(final_p)
    print(final_p.shape)
    # Find principal flow and display first 27 images of flow obtained #
    h = choose_h_gaussian(sampled_X_on_sphere, final_p, 90) # needs to be very high!
    curve = principal_flow(sampled_X_on_sphere, X.shape[1], 0.02, h, \
        flow_num=1, start_point=final_p, kernel_type="gaussian", max_iter=20)
    rgb.append(curve)

rgb = np.array(rgb)
print(rgb.shape)
rgb = rgb.reshape(rgb.shape[1], m, n, 3)

for j in range(4):
    for i in range(9):
        plt.subplot(330 + 1 + i)
        plt.imshow(rgb[i + 9*j])
    plt.show()

'''

(train_X, train_y), (test_X, test_y) = cifar10.load_data()
train_y = train_y.reshape(train_y.shape[0])
test_y = test_y.reshape(test_y.shape[0])
m = train_X.shape[1]
n = train_X.shape[2]
print(type(train_X))
print(train_X.shape)
print(train_y[0])

def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.299, 0.587, 0.144])

train_X = rgb2gray(train_X)
test_X = rgb2gray(test_X)
print(train_X.shape)

train_filter = np.where(train_y == DIGIT)
test_filter = np.where(test_y == DIGIT)

train_X, train_y = train_X[train_filter], train_y[train_filter]
test_X, test_y = test_X[test_filter], test_y[test_filter]

# Reshaping Image Array

# train_X.shape = (...., 32, 32) -> (....,1024)
image_vector_size = train_X.shape[1]*train_X.shape[2]
train_X = train_X.reshape(train_X.shape[0], image_vector_size)

# Sample from train_X #
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
final_p = sphere_centroid_finder_vecs(sampled_X_on_sphere, sampled_X.shape[1], 0.05, 0.01)
# print(final_p)

final_p_img = final_p.reshape(m, n)
plt.imshow(final_p_img, cmap=plt.get_cmap('gray'))
plt.show()

h = choose_h_gaussian(sampled_X_on_sphere, final_p, 85) # needs to be very high!
radius = choose_h_binary(sampled_X_on_sphere, final_p, 40)
upper, curve, lower = principal_boundary(sampled_X_on_sphere, sampled_X.shape[1], 0.02, h, radius, \
    start_point=final_p, kernel_type="gaussian", max_iter=40)

print("upper")
for j in range(5):
    for i in range(9):
        plt.subplot(330 + 1 + i)
        plt.imshow(upper[i + 9*j].reshape(m, n), cmap=plt.get_cmap('gray'))
    plt.show()

print("curve")
for j in range(5):
    for i in range(9):
        plt.subplot(330 + 1 + i)
        plt.imshow(curve[i + 9*j].reshape(m, n), cmap=plt.get_cmap('gray'))
    plt.show()
print("lower")
for j in range(5):
    for i in range(9):
        plt.subplot(330 + 1 + i)
        plt.imshow(lower[i + 9*j].reshape(m, n), cmap=plt.get_cmap('gray'))
    plt.show()

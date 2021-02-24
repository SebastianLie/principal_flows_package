import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist
from common_methods_sphere import put_on_sphere
from principal_flow import choose_h_gaussian, principal_flow,choose_h_binary
from centroid_finder import sphere_centroid_finder_vecs
from principal_boundary_flows import principal_boundary

# Program to run principal flow on MNIST data.
# Choose the digit,and the number of samples for the data
# in the constants below.

# Constants #

DIGIT = 3
SAMPLES = 2000

# Data #

(train_X, train_y), (test_X, test_y) = mnist.load_data()
print(type(train_X))
print(train_X.shape)
print(train_y[0])

train_filter = np.where(train_y == DIGIT)
test_filter = np.where(test_y == DIGIT)

train_X, train_y = train_X[train_filter], train_y[train_filter]
test_X, test_y = test_X[test_filter], test_y[test_filter]

# Reshaping Image Array

# train_X.shape = (...., 28, 28) -> (....,784)
image_vector_size = train_X.shape[1]*train_X.shape[2]
train_X = train_X.reshape(train_X.shape[0], image_vector_size)

# Sample from train_X #
train_samples = np.random.choice(train_X.shape[0], size=SAMPLES)
sampled_X = train_X[train_samples]

# Sampled Images #
for i in range(9):
    plt.subplot(330 + 1 + i)
    plt.imshow(sampled_X[i].reshape(28, 28), cmap=plt.get_cmap('gray'))
plt.show()

sampled_X_on_sphere = put_on_sphere(sampled_X)
# print(sampled_X_on_sphere[0])

# Find centroid of data #
final_p = sphere_centroid_finder_vecs(sampled_X_on_sphere, sampled_X.shape[1], 0.05, 0.01)
# print(final_p)

final_p_img = final_p.reshape(28, 28)
plt.imshow(final_p_img, cmap=plt.get_cmap('gray'))
plt.show()

# Find principal flow and display first 27 images of flow obtained #
h = choose_h_gaussian(sampled_X_on_sphere, final_p, 75) # needs to be very high!
radius = choose_h_binary(sampled_X_on_sphere, final_p,30)
upper, curve, lower = principal_boundary(sampled_X_on_sphere, sampled_X.shape[1], 0.02, h, radius, \
    start_point=final_p, kernel_type="gaussian", max_iter=40)

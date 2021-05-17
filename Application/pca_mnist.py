import numpy as np
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist
import cv2
from common_methods_sphere import put_on_sphere
from principal_flow import choose_h_gaussian, principal_flow,choose_h_binary
from centroid_finder import sphere_centroid_finder_vecs
from principal_boundary_flows import principal_boundary

# Program to run principal flow on MNIST data.
# Choose the digit,and the number of samples for the data
# in the constants below.

# Constants #

DIGIT = 3
SAMPLES = 1000

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
np.random.seed(888)
train_samples = np.random.choice(train_X.shape[0], size=SAMPLES)
sampled_X = train_X[train_samples]

# Sampled Images #
'''
for i in range(9):
    plt.subplot(330 + 1 + i)
    plt.imshow(sampled_X[i].reshape(28, 28), cmap=plt.get_cmap('gray'))
plt.show()
'''

# sampled_X_on_sphere = put_on_sphere(sampled_X)
# print(sampled_X_on_sphere[0])


pca = PCA(n_components=100)
pca.fit(sampled_X)
principal_direction = np.array(pca.components_[0])
print(principal_direction.shape)
print(principal_direction)

p = np.mean(sampled_X, axis=0)
p_opp = p
curve = []
curve.append(p)
n_iter = 20
STEP = 10

for i in range(n_iter):
    p = p + principal_direction*STEP
    p_opp = p_opp - principal_direction*STEP
    curve.append(p)
    curve.insert(0, p_opp)

for i in range(len(curve)):
    img = curve[i].reshape(28, 28)
    plt.imshow(img, cmap=plt.get_cmap('gray'))
    plt.savefig("mnist_pca_pics/{}.".format(i))
    #plt.show()
    




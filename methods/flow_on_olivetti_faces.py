import numpy as np
import matplotlib.pyplot as plt
import os
from common_methods_sphere import *
from principal_flow import *
from centroid_finder import *

'''
As described on the original website:

There are ten different images of each of 40 distinct subjects. For some subjects, the images were taken at different times, varying the lighting, facial expressions (open / closed eyes, smiling / not smiling) and facial details (glasses / no glasses). All the images were taken against a dark homogeneous background with the subjects in an upright, frontal position (with tolerance for some side movement).
The image is quantized to 256 grey levels and stored as unsigned 8-bit integers; the loader will convert these to floating point values on the interval [0, 1], which are easier to work with for many algorithms.

The “target” for this database is an integer from 0 to 39 indicating the identity of the person pictured; however, with only 10 examples per class, this relatively small dataset is more interesting from an unsupervised or semi-supervised perspective.

The original dataset consisted of 92 x 112, while the version available here consists of 64x64 images.

Credit to AT&T Laboratories Cambridge for images
'''

# Constants #

SAMPLES = 400

PATH = os.path.abspath(os.curdir) + "\\olivetti_faces"
X = np.load('olivetti_faces/olivetti_faces.npy') # shape = (400, 64, 64)
m = X.shape[1]
n = X.shape[2]

image_vector_size = m * n
X = X.reshape(X.shape[0], image_vector_size)

# X.shape = (....,784)
'''
for j in range(40):
    for i in range(9):
        plt.subplot(330 + 1 + i)
        plt.imshow(sampled_X[i+9*j].reshape(m, n), cmap=plt.get_cmap('gray'))
    plt.show()
'''

# sampled_X.shape = (100,784)
sampled_X_on_sphere = put_on_sphere(X)
final_p = sphere_centroid_finder_vecs(sampled_X_on_sphere, sampled_X.shape[1], 0.05, 0.01,max_iter=200)

# Show "centroid image" obtained #
final_p_img = final_p.reshape(m, n)
plt.imshow(final_p_img, cmap=plt.get_cmap('gray'))
plt.show()

h = choose_h_binary(sampled_X_on_sphere, final_p, 40) # needs to be very high!
curve = principal_flow(sampled_X_on_sphere, sampled_X.shape[1], 0.01, h, \
    flow_num=1, start_point=final_p, kernel_type="binary", max_iter=20)
for j in range(4):
    for i in range(9):
        plt.subplot(330 + 1 + i)
        plt.imshow(curve[i + 9*j].reshape(m, n), cmap=plt.get_cmap('gray'))
    plt.show()

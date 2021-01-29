import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm, colors
from mpl_toolkits.mplot3d import Axes3D
import scipy
import math
import os
import cv2
from common_methods_sphere import *
from principal_flow import *
from centroid_finder import *


def test_pairwise_distances(points, p, tol=0.01):
    '''
    Test to see if p is somewhat equally far away from all points
    Cal
    '''
    points = np.array(points)
    p = np.array(p)
    distances = get_pairwise_distances(points, p)
    diff = list(map(lambda x: abs(x - sum(distances)/len(distances), distances)))
    return False if max(diff) > tol else True

def project_point_onto_plane(point, p):
    '''
    TODO CHANGE THIS, IS WRONG
    Test by using 1,0,0 project on 0,0,1

    Plane is tangent plane on the sphere at p.
    Assumes center of sphere = (0, 0, 0)
    Then normal is p since p - 0 = p
    ''' 
    v = point - p
    normal =  -2*p
    dist = np.dot(v, normal)
    projected_point = point - dist*normal

    return projected_point

def log_map_alt(tangent_point, sphere_point):
    '''
    Projects from sphere to tangent plane
    '''
    # formula from http://ani.stat.fsu.edu/~anuj/CVPR_Tutorial/Part2.pdf
    
    p = tangent_point
    q = sphere_point
    theta = angle(p, q)
    return (theta/np.sin(theta)) * (q - p*np.cos(theta)) 


def project_point_onto_sphere(point):
    '''
    Idea is that since the point is already from the 
    origin, the center of the sphere, just need to change the 
    magnitude of the vector so that the point 
    sits on the sphere's edge
    '''
    normal = point - 0  # point - origin is normal to sphere

    return normal/np.linalg.norm(point)


def compute_principal_component(points):
    '''
    Finds principal component of a few points.
    1. Find the centered data
    2. Find the covariance matrix of the centered data
    3. Do the eigenvalue decomposition of the covar mat
    4. sort the eigenvec according to eigenvalue size
    5. Return the eigenvalue, eigenvector tuple that has the largest
        eigenvalue.
    '''
    dimension_means = np.mean(points.T, axis=1)
    centered_data = points - dimension_means
    covar_mat_centered_data = np.cov(centered_data.T)
    eig_values, eig_vectors = scipy.linalg.eig(covar_mat_centered_data)
    #print(scipy.linalg.svd(centered_data.T))
    U, S, Vt = scipy.linalg.svd(centered_data, full_matrices=False)
    # sign flip the svd
    max_abs_cols = np.argmax(np.abs(U), axis=0)
    signs = np.sign(U[max_abs_cols, range(U.shape[1])])
    U *= signs
    Vt *= signs[:, np.newaxis]
    eig_tuples = list(zip(eig_values, eig_vectors.T))
    # reverse = True means sort by descending!
    eig_tuples = sorted(eig_tuples, reverse=True)
    return S[0], Vt[0]


def compute_principal_component_vecs_old(vectors, p):
    '''
    1)      v_j =Log_p(p_j), 1<=j<=n
    2)      V=[v_1’,v_2,…v_n]
    3)      Do eigen on V’V/n-1, find the e_1 (the eigenvector associated with the largest eigen value)
    4)      Update p<-p+\epsilon e_1
    5)      Repeat
    '''
    # vectors currently rows, need to transform to columns!
    X = vectors
    n = len(vectors)
    covar_mat = np.dot(X.T, X)/(n-1)
    eig_values, eig_vectors = np.linalg.eig(covar_mat)
    eig_tuples = list(zip(eig_values, eig_vectors.T))
    eig_tuples = sorted(eig_tuples, reverse=True)
    #print(eig_tuples[0][1])
    sign_vector = np.dot(vectors, eig_tuples[0][1])
    if (sign_vector < 0).sum() >= (sign_vector > 0).sum():
        vec = - eig_tuples[0][1]
    else:
        vec = eig_tuples[0][1]

    return sorted(eig_values,reverse=True), vec

def algorithm_alt(epsilon, tol, debugging=False):
    '''
    0. General some points in advance including p.
    1. choose 1 of the points randomly, call it p
    2. find the tangent plane to the sphere at this point - use z coordinate and make all 
    3. project the remaining points on the sphere onto the tangent plane
    4. calculate principal component
    5. move a step in the PC direction and get new point, p'
    6. project that point back onto the sphere
    7. repeat step 2 onwards

    Use epsilon* V+p to move. Set epsilon to be a small number.

    You don’t move when V is small in the sense its correspond eigenvalue is tiny.

    Note on log maps and exp maps:
    Log maps and exp maps give you the DIRECTION in which to move to get the 
    projected coordinate!!
    Log maps:
    Usage - the log map gives the direction on THE TANGENT PLANE 
    in which the original tangent point should move 
    so that it is the closest point on the tangent plane to the 
    point we want to project.
    Thus, we add the tangent plane point, p and the logmap of p and 
    the point to project.

    Exp maps:

    just needed exp_map(p, tangent_vector),
    where tangent vector = point on plane - p, p is base point for the tangent plane
    since point on plane - p is the vector that points from p, the base point of the tangent
    plane to the point we want to project.
    '''
    # generate points and check that points generated are on the sphere
    points_on_sphere = generate_square()
    points_on_sphere = np.asarray(points_on_sphere.T, dtype=np.float32)
    points_on_sphere = np.array(list(map(spherical_to_cartesian, points_on_sphere)))

    # assert (np.around(list(map(np.linalg.norm, points_on_sphere)), 1) == np.ones(num_points)).all(), "Points generated not on the sphere"

    # choose p, and get the array of points that exclude p.
    p_index = 1
    p = points_on_sphere[p_index]

    # start the loop by the algo in the docstring.
    num_iter = 0
    while True:
        num_iter += 1
        plane_vectors = np.array(list(map(lambda point: log_map_sphere(p, point), points_on_sphere)))
        eig_value, principal_direction = compute_principal_component_vecs(plane_vectors, p)
        print(principal_direction)
        # print(plane_vectors)
        print(eig_value)
        p_prime_plane = p + epsilon * principal_direction
        p_prime = exp_map_sphere(p, p_prime_plane - p)
        p = p_prime
        if eig_value < tol:
            break
        if num_iter > 100:
            break
        if debugging:
            return points_on_sphere.T, p_prime
    return p, num_iter, points_on_sphere.T

def noising_data(points, variance=0.1):
    # approach: add a random centered normal RV to every dimension
    # of each point
    # still needs to be normalised to have norm 1! 
    # data not usable after this step yet
    # assumes points are in (n,p) format
    p = points.shape[1]
    sigma = math.sqrt(variance)
    return np.array([np.array(point) - np.random.normal(0, sigma, p) for point in points])


def testing_flow_all():

    import glob 
    filenames = glob.glob('Sample_Data/*.csv')
    print(filenames)
    for name in filenames:
        # ['data1.csv', 'data10.csv', 'data11.csv', 'data12.csv', 'data13.csv', 'data14.csv', 'data2.csv', 'data3.csv', 'data4.csv', 'data5.csv', 'data7.csv', 'data8.csv', 'data9.csv']
        data = pd.read_csv(name)
        data_np = data.to_numpy()
        data_np = (data_np.T[1:]).T
        random.seed(999)

        final_p = sphere_centroid_finder_vecs(data_np, 3, 0.05, 0.01)

        phi = np.linspace(0, np.pi, 20)
        theta = np.linspace(0, 2 * np.pi, 40)
        x = np.outer(np.sin(theta), np.cos(phi))
        y = np.outer(np.sin(theta), np.sin(phi))
        z = np.outer(np.cos(theta), np.ones_like(phi))

        fig, ax = plt.subplots(1, 1, subplot_kw={'projection':'3d'})
        ax.plot_wireframe(x, y, z, color='k', rstride=1, cstride=1,alpha =0.3) # alpha affects transparency of the plot

        xx, yy, zz = data_np
        ax.scatter(xx, yy, zz, color="k", s=50)
        ax.scatter(final_p[0], final_p[1], final_p[2], color="r", s=50)

        plt.show()


def testing_flow_single():
    data = pd.read_csv('Sample_Data/data13.csv')
    data_np = data.to_numpy()
    
    data_np = (data_np.T[1:]).T
    random.seed(999)
    
    final_p = sphere_centroid_finder_vecs(data_np, 3, 0.05, 0.01)
    #print(final_p)
    '''
    # what's next? try make h a normalising constant
    # make fraction in np.exp smaller to get bigger weight
    # so bigger weight for smaller points - denominator bigger than numerator which is dist from centroid to point.
    distances = get_pairwise_distances(data_np.T, final_p)
    plane_vectors = np.array(list(map(lambda point: log_map_sphere(final_p, point), data_np.T)))
    dist_and_weight = [(d, new_gaussian(point, final_p, 3)) for d, point in zip(distances, plane_vectors)]
    dist_and_exp = [(d, -np.linalg.norm(point - final_p)**2) for d, point in zip(distances, plane_vectors)]
    exp_parts = [(-np.linalg.norm(x - final_p)**2) for x in plane_vectors]
    #print(distances)
    #print(gaussian_kernel(0.1,data_np.T,final_p))
    #print(dist_and_weight)
    print("")
    print(sorted(dist_and_exp))
    #print(sorted(dist_and_weight))
    
    print(final_p)
    #print(choose_h(data_np.T, final_p))
    print(choose_h_gaussian(data_np.T, final_p, 5))
    '''
    print(choose_h_gaussian(data_np.T, final_p, 100))
    print(choose_h_gaussian(data_np.T, final_p, 1))
    h = choose_h_gaussian(data_np.T, final_p, 90) # needs to be very high!
    curve = principal_flow(data_np, 3, 0.02, h, flow_num=1, start_point=final_p, kernel_type="gaussian")
    #print(curve)
    x_curve, y_curve, z_curve = curve.T
   
    phi = np.linspace(0, np.pi, 20)
    theta = np.linspace(0, 2 * np.pi, 40)
    x = np.outer(np.sin(theta), np.cos(phi))
    y = np.outer(np.sin(theta), np.sin(phi))
    z = np.outer(np.cos(theta), np.ones_like(phi))

    fig, ax = plt.subplots(1, 1, subplot_kw={'projection':'3d'})
    ax.plot_wireframe(x, y, z, color='k', rstride=1, cstride=1,alpha =0.3) # alpha affects transparency of the plot

    xx, yy, zz = data_np
    ax.scatter(xx, yy, zz, color="k", s=50)
    ax.scatter(x_curve, y_curve, z_curve, color="r", s=50)

    plt.show()

def testing_flow_noisy():
    data = pd.read_csv('Sample_Data/data13.csv')
    data_np = data.to_numpy()
    data_np = data_np.T[1:]
    noisy_data = noising_data(data_np, 0.01)
    noisy_data = put_on_sphere(noisy_data)
    random.seed(999)
    final_p = sphere_centroid_finder_vecs(noisy_data, 3, 0.05, 0.01)
    #print(final_p)
    h = choose_h_gaussian(noisy_data, final_p, 75) # needs to be very high! # needs to be very high!
    curve = principal_flow(noisy_data, 3, 0.02, h, flow_num=1, start_point=final_p, kernel_type="gaussian", max_iter=30)
    #print(curve)
    x_curve, y_curve, z_curve = curve.T

    phi = np.linspace(0, np.pi, 20)
    theta = np.linspace(0, 2 * np.pi, 40)
    x = np.outer(np.sin(theta), np.cos(phi))
    y = np.outer(np.sin(theta), np.sin(phi))
    z = np.outer(np.cos(theta), np.ones_like(phi))

    fig, ax = plt.subplots(1, 1, subplot_kw={'projection':'3d'})
    ax.plot_wireframe(x, y, z, color='k', rstride=1, cstride=1,alpha =0.3) # alpha affects transparency of the plot

    xx, yy, zz = noisy_data.T
    ax.scatter(xx, yy, zz, color="k", s=50)
    ax.scatter(x_curve, y_curve, z_curve, color="r", s=50)

    plt.show()

def mnist_flow(digit, samples=100): #TODO
    
    from keras.datasets import mnist
    (train_X, train_y), (test_X, test_y) = mnist.load_data()
    print(type(train_X))
    print(train_X.shape)

    train_filter = np.where(train_y == digit)
    test_filter = np.where(test_y == digit)

    train_X, train_y = train_X[train_filter], train_y[train_filter]
    test_X, test_y = test_X[test_filter], test_y[test_filter]

    '''
    for i in range(9):  
        plt.subplot(330 + 1 + i)
        plt.imshow(train_X[i], cmap=plt.get_cmap('gray'))
    plt.show()
    '''
    # train_X.shape = (...., 28, 28)
    image_vector_size = train_X.shape[1]*train_X.shape[2]
    train_X = train_X.reshape(train_X.shape[0], image_vector_size)
    test_X = test_X.reshape(test_X.shape[0], image_vector_size)

    # train_X.shape = (....,784)
    train_samples = np.random.choice(train_X.shape[0], size=samples)
    # print(train_samples)
    sampled_X = train_X[train_samples]

    for i in range(9):
        plt.subplot(330 + 1 + i)
        plt.imshow(sampled_X[i].reshape(28,28), cmap=plt.get_cmap('gray'))
    plt.show()
   
    # sampled_X.shape = (100,784)
   
    sampled_X_on_sphere = put_on_sphere(sampled_X)
    # print(sampled_X_on_sphere[0])
    final_p = sphere_centroid_finder_vecs(sampled_X_on_sphere, sampled_X.shape[1], 0.05, 0.01)
    # print(final_p)
   
    final_p_img = final_p.reshape(28, 28)
    plt.imshow(final_p_img, cmap=plt.get_cmap('gray'))
    plt.show()

    h = choose_h_gaussian(sampled_X_on_sphere, final_p, 75) # needs to be very high!
    curve = principal_flow(sampled_X_on_sphere, sampled_X.shape[1], 0.02, h, flow_num=2,start_point=final_p,kernel_type="gaussian",max_iter=20)
    for j in range(3):
        for i in range(9):
            plt.subplot(330 + 1 + i)
            plt.imshow(curve[i + 9*j].reshape(28,28), cmap=plt.get_cmap('gray'))
        plt.show()

    # TODO use principal flow to get mnist flow of one digit

def noisy_image_flow(): #TODO
    directory_in_str = os.path.abspath(os.curdir) + "\\noisy_images"
    directory = os.fsencode(directory_in_str)
    X = list()  # image array
    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        if filename.endswith(".png"): 
            image = cv2.imread('noisy_images/' + filename, 0)
            X.append(image)
    X = np.array(X)
    m, n = image.shape
    image_vector_size = m * n
    for i in range(9):
        plt.subplot(330 + 1 + i)
        plt.imshow(X[i].reshape(m, n), cmap=plt.get_cmap('gray'))
    plt.show()
    num_images = len(X)
    X = X.reshape(num_images, image_vector_size)
    print(X.shape)
    X_on_sphere = put_on_sphere(X)
    final_p = sphere_centroid_finder_vecs(X_on_sphere, X.shape[1], 0.05, 0.01)
    # print(final_p)
   
    final_p_img = final_p.reshape(m, n)
    plt.imshow(final_p_img, cmap=plt.get_cmap('gray'))
    plt.show()

    h = choose_h_gaussian(X_on_sphere, final_p, 75) # needs to be very high!
    curve = principal_flow(X_on_sphere, X.shape[1], 0.02, h, flow_num=1, start_point=final_p, kernel_type="gaussian",max_iter=20)
    for j in range(3):
        for i in range(9):
            plt.subplot(330 + 1 + i)
            plt.imshow(curve[i + 9*j].reshape(m, n), cmap=plt.get_cmap('gray'))
        plt.show()
    # TODO use principal flow to get mnist flow of one digit


print(os.path.abspath(os.curdir))
os.chdir("..")
print(os.path.abspath(os.curdir))
testing_flow_noisy()
#testing_flow_single()
#mnist_flow(3)
#noisy_image_flow()
'''
ps, p_prime = algorithm_alt(0.05, 0.1,debugging=True)
print("break")
pp1, ps1, p_prime1 = algorithm(0.05, 0.1,debugging=True)
#print(pp)
#print(ps)
#print(p_prime)
'''

'''
phi = np.linspace(0, np.pi, 20)
theta = np.linspace(0, 2 * np.pi, 40)
x = np.outer(np.sin(theta), np.cos(phi))
y = np.outer(np.sin(theta), np.sin(phi))
z = np.outer(np.cos(theta), np.ones_like(phi))

fig, ax = plt.subplots(1, 1, subplot_kw={'projection':'3d'})
ax.plot_wireframe(x, y, z, color='k', rstride=1, cstride=1,alpha =0.3)

xx, yy, zz = ps
ax.scatter(xx, yy, zz, color="k", s=50)

xp, yp, zp = p_prime
ax.scatter(xp, yp, zp, color="r", s=50)

plt.show()
'''
'''
phi = np.linspace(0, np.pi, 20)
theta = np.linspace(0, 2 * np.pi, 40)
x = np.outer(np.sin(theta), np.cos(phi))
y = np.outer(np.sin(theta), np.sin(phi))
z = np.outer(np.cos(theta), np.ones_like(phi))

fig, ax = plt.subplots(1, 1, subplot_kw={'projection':'3d'})
ax.plot_wireframe(x, y, z, color='g', rstride=1, cstride=1)

points_on_sphere = generate_dense_square()
points_on_sphere = np.asarray(points_on_sphere.T, dtype=np.float32)
points_on_sphere = np.array(list(map(spherical_to_cartesian, points_on_sphere)))
xx, yy, zz = points_on_sphere.T
ax.scatter(xx, yy, zz, color="k", s=50)

plt.show()
'''
'''
random.seed(999)
final_p, num_iter, points_to_print = algorithm_vecs(0.05, 0.01)
print(num_iter)

phi = np.linspace(0, np.pi, 20)
theta = np.linspace(0, 2 * np.pi, 40)
x = np.outer(np.sin(theta), np.cos(phi))
y = np.outer(np.sin(theta), np.sin(phi))
z = np.outer(np.cos(theta), np.ones_like(phi))

fig, ax = plt.subplots(1, 1, subplot_kw={'projection':'3d'})
ax.plot_wireframe(x, y, z, color='k', rstride=1, cstride=1,alpha =0.3) # alpha affects transparency of the plot

xx, yy, zz = points_to_print
ax.scatter(xx, yy, zz, color="k", s=50)
ax.scatter(xx[0], yy[0], zz[0], color="b", s=50)
ax.scatter(final_p[0], final_p[1], final_p[2], color="r", s=50)

plt.show()
'''

'''
x = np.array([0,1,0])
p = np.array([1,0,0])
log = p + log_map(p, x)
exp = exponential_map(p, log-p)
print(log)
print(exp)
'''

'''
pca = PCA()
points_on_sphere = generate_square()
points_on_sphere = np.asarray(points_on_sphere.T, dtype=np.float32)
points_on_sphere = np.array(list(map(spherical_to_cartesian, points_on_sphere)))
p_index = random.randint(0, 3)
p = points_on_sphere[p_index]

points_on_plane = list(map(lambda point: p + log_map(p, point), points_on_sphere))
points_on_plane = np.asarray(points_on_plane, dtype=np.float32)

pca.fit(points_on_plane)
print(pca.components_)
compute_principal_component(points_on_plane)
#print(principal_direction)
'''

'''
from geomstats.geometry.hypersphere import Hypersphere
sphere = Hypersphere(dim=2)
points_on_sphere = generate_square()
points_on_sphere = np.asarray(points_on_sphere.T, dtype=np.float32)
points_on_sphere = np.array(list(map(spherical_to_cartesian, points_on_sphere)))
p = points_on_sphere[2]

points_on_plane = list(map(lambda point: p + log_map(p, point), points_on_sphere))
points_on_plane = np.asarray(points_on_plane, dtype=np.float32)

# v = np.array([1,0,0])
# print(exponential_map(v,v))
#print(points_on_sphere)
#print(points_on_plane)
test = sphere.metric.exp(points_on_plane[0],base_point=p)
test2 = exponential_map(p, points_on_plane[0]-p)

print(points_on_sphere)
points_sphere = list(map(lambda point: exponential_map(p, point-p), points_on_plane))
print(points_sphere)
#print(p-points_on_plane[0] + exponential_map(p, p-points_on_plane[0]))
'''

'''
############
# Plotting #
############

def plot_projected():

    pp, ps, p_prime = sphere_centroid_finder_points(0.5, 0.1, debugging=True)
    phi = np.linspace(0, np.pi, 20)
    theta = np.linspace(0, 2 * np.pi, 40)
    x = np.outer(np.sin(theta), np.cos(phi))
    y = np.outer(np.sin(theta), np.sin(phi))
    z = np.outer(np.cos(theta), np.ones_like(phi))

    fig, ax = plt.subplots(1, 1, subplot_kw={'projection':'3d'})
    ax.plot_wireframe(x, y, z, color='g', rstride=1, cstride=1)

    xx, yy, zz = ps
    ax.scatter(xx, yy, zz, color="k", s=50)

    xi, yi, zi = pp
    ax.scatter(xi, yi, zi, color="b", s=50)

    xp, yp, zp = p_prime
    ax.scatter(xp, yp, zp, color="r", s=50)

    plt.show()

def plot_results():
    final_p, num_iter, points_to_print = sphere_centroid_finder_points(0.1, 0.01)
    print(num_iter)

    phi = np.linspace(0, np.pi, 20)
    theta = np.linspace(0, 2 * np.pi, 40)
    x = np.outer(np.sin(theta), np.cos(phi))
    y = np.outer(np.sin(theta), np.sin(phi))
    z = np.outer(np.cos(theta), np.ones_like(phi))

    fig, ax = plt.subplots(1, 1, subplot_kw={'projection':'3d'})
    ax.plot_wireframe(x, y, z, color='k', rstride=1, cstride=1,alpha =0.3) # alpha affects transparency of the plot

    xx, yy, zz = points_to_print
    ax.scatter(xx, yy, zz, color="k", s=50)

    ax.scatter(final_p[0], final_p[1], final_p[2], color="r", s=50)

    plt.show()
'''

# plot_projected()
'''
Issues:
Resolved YAY
Resolved using log maps properly, points now projected to tangent plane correctly
1.
Move in principal direction not on plane?
Sort this out
Test with other PCA libraries, test if we arrive at the same answer.

2.
With using exp map.
project p is not on sphere at all.
Figure out how to use exp map to get projection!
'''
'''
random.seed(999)
final_p, num_iter, points_to_print = sphere_centroid_finder_points(0.1, 0.01)
print(num_iter)

phi = np.linspace(0, np.pi, 20)
theta = np.linspace(0, 2 * np.pi, 40)
x = np.outer(np.sin(theta), np.cos(phi))
y = np.outer(np.sin(theta), np.sin(phi))
z = np.outer(np.cos(theta), np.ones_like(phi))

fig, ax = plt.subplots(1, 1, subplot_kw={'projection':'3d'})
ax.plot_wireframe(x, y, z, color='k', rstride=1, cstride=1,alpha =0.3) # alpha affects transparency of the plot

xx, yy, zz = points_to_print
ax.scatter(xx, yy, zz, color="k", s=50)

ax.scatter(final_p[0], final_p[1], final_p[2], color="r", s=50)

plt.show()
'''

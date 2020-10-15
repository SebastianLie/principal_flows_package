import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib import cm, colors
from mpl_toolkits.mplot3d import Axes3D
import scipy
from common_methods_sphere import *

'''
Sphere - PGA
'''

################################
# Principal Geodesic functions #
################################

def compute_principal_component_points_inaccurate(points):
    '''
    Was abit wrong until compared against sklearn's PCA:
    Realised the eig vector matrix should actually be TRANSPOSED
    to find the individual eigen vectors. Original only gives  
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
    eig_values, eig_vectors = np.linalg.eig(covar_mat_centered_data)
    # V IMPT that vector matrix is transposed! Why? Still figuring it out lol
    eig_tuples = list(zip(eig_values, eig_vectors.T))
    # reverse = True means sort by descending!
    eig_tuples = sorted(eig_tuples, reverse=True)
    return eig_tuples[0]


def compute_principal_component_points(points):
    '''
    Finds principal component of a few points.
    1. Find the centered data
    2. Find the covariance matrix of the centered data
    3. Do the eigenvalue decomposition of the covar mat
    4. sort the eigenvec according to eigenvalue size
    5. Return the eigenvalue, eigenvector tuple that has the largest
        eigenvalue.
    Note: 
    Obtaining the wrong sign for the eigenvector was an issue.
    Using eig decomp on the covar matrix always somehow produced the wrong sign.
    Instead, we simply apply SVD to the centered data, which is equivalent, 
    then do a sign transformation.
    Taken from sklearn's PCA implementation.
    '''
    dimension_means = np.mean(points.T, axis=1)
    centered_data = points - dimension_means
    U, S, Vt = linalg.svd(centered_data, full_matrices=False)
    # this chunk flips sign of the svd
    max_abs_cols = np.argmax(np.abs(U), axis=0)
    signs = np.sign(U[max_abs_cols, range(U.shape[1])])
    U *= signs
    Vt *= signs[:, np.newaxis]
    return S[0]**2, Vt[0]

def compute_principal_component_vecs(vectors, p):
    # works now AHHHH yay so glad it does what I want.
    '''
    Idea: 
    When we center the data, we are really just obtaining the vectors
    that point from each data point to the "center" of the data, 
    i.e the vector containing the mean of each dimension. 
    Does this remind you of something? 
    Yes, this is basically the output of the log map
    of p for each data point. Thus the centered data can 
    simply be replaced with the matrix of the vectors obtained 
    from the log map of p on all data points, using p as the pseudo-center.
    Then we call SVD on the vectors which is the centered data, and sign flip.
    1)      v_j =Log_p(p_j), 1<=j<=n
    2)      V=[v_1’,v_2',…v_n']
    3)      Do eigen on V’V/n-1, find the e_1 (the eigenvector associated with the largest eigen value)
    4)      Update p<-p+epsilon e_1
    5)      Repeat
    '''
    # vectors currently rows, need to transform to columns!
    # svd here since X is like the 'centered data', the data minus the estimated mean which is p.
    X = vectors
    U, S, Vt = scipy.linalg.svd(X, full_matrices=False)
    max_abs_cols = np.argmax(np.abs(U), axis=0)
    signs = np.sign(U[max_abs_cols, range(U.shape[1])])
    Vt *= signs[:, np.newaxis]

    return S, Vt[0]

def sphere_centroid_finder_points(epsilon, tol, num_points=4, debugging=False):
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
    '''
    # generate points and check that points generated are on the sphere
    points_on_sphere = generate_square()
    points_on_sphere = np.asarray(points_on_sphere.T, dtype=np.float32)
    points_on_sphere = np.array(list(map(spherical_to_cartesian, points_on_sphere)))

    # assert (np.around(list(map(np.linalg.norm, points_on_sphere)), 1) == np.ones(num_points)).all(), "Points generated not on the sphere"

    # choose p, and get the array of points that exclude p.
    p_index = random.randint(0, num_points-1)
    p = points_on_sphere[p_index]
    # points = points_on_sphere[np.arange(len(points_on_sphere)) != p_index]
    
    # start the loop by the algo in the docstring.
    num_iter = 0
    while True:
        num_iter += 1
        points_on_plane = list(map(lambda point: p + log_map_sphere(p, point), points_on_sphere))
        points_on_plane = np.asarray(points_on_plane, dtype=np.float32)
        points_on_plane_w_p = np.vstack((points_on_plane, p))

        eig_value, principal_direction = compute_principal_component_points(points_on_plane_w_p)
        p_prime_plane = p + epsilon * principal_direction
        p_prime = exp_map_sphere(p, p_prime_plane - p)
        p = p_prime
        if debugging:
            return points_on_plane.T, points_on_sphere.T, p_prime_plane
        if eig_value < tol:
            break
        if num_iter > 100:
            break
    return p, num_iter, points_on_sphere.T

def sphere_centroid_finder_vecs(data, epsilon, tol, debugging=False):
    '''
    Central Algorithm of this file.
    Works!
    Idea: 
    1. Takes in the data, then chooses the first point in the dataset as the 
    pseudo-center.
    2. Calculate the log map of p on these points, to obtain the vectors residing on the plane 
    tangent to the sphere at p.
    3. 
    '''
    # choose p, and get the array of points that exclude p.
    data = np.array(data)
    if data.shape[1] != 3:
        data = data.T
    points_on_sphere = data
    p_index =  0
    p = points_on_sphere[p_index]

    num_iter = 0
    while True:
        num_iter += 1
        plane_vectors = np.array(list(map(lambda point: log_map_sphere(p, point), points_on_sphere)))
        eig_values, principal_direction = compute_principal_component_vecs(plane_vectors, p)
        #print(principal_direction)
        p_prime_plane = p + epsilon * principal_direction
        p_prime = exp_map_sphere(p, p_prime_plane - p)
        # print(np.linalg.norm(p_prime))
        p = p_prime
        if test_eig_diff(eig_values, tol):
            # gap between eigenvalues are v small
            break
        if num_iter > 100:
            break
        if debugging:
            return points_on_sphere.T, p_prime
    return p, num_iter, points_on_sphere.T

def sphere_centroid_finder_no_pca(epsilon, tol, num_points=4,debugging=False): # works!!
    '''
    takes adv of the fact that sum of plane vectors at mean will equal 0.
    '''
    # generate points and check that points generated are on the sphere
    points_on_sphere = generate_square()
    points_on_sphere = np.asarray(points_on_sphere.T, dtype=np.float32)
    points_on_sphere = np.array(list(map(spherical_to_cartesian, points_on_sphere)))

    # assert (np.around(list(map(np.linalg.norm, points_on_sphere)), 1) == np.ones(num_points)).all(), "Points generated not on the sphere"

    # choose p, and get the array of points that exclude p.
    p_index = 2
    p = points_on_sphere[p_index]

    # start the loop by the algo in the docstring.
    num_iter = 0
    while True:
        num_iter += 1
        plane_vectors = np.array(list(map(lambda point: log_map_sphere(p, point), points_on_sphere)))
        principal_direction = np.sum(plane_vectors, axis=0)
        if np.linalg.norm(principal_direction) < tol:
            break
        p_prime_plane = p + epsilon * principal_direction
        p_prime = exp_map_sphere(p, p_prime_plane - p)
        p = p_prime
        if num_iter > 100:
            break
        if debugging:
            return points_on_sphere.T, p_prime
    return p, num_iter, points_on_sphere.T


import numpy as np
import pandas as pd
import scipy
import math
from common_methods_sphere import log_map_sphere, exp_map_sphere
from common_methods_sphere import spherical_to_cartesian, generate_square, test_eig_diff
from common_methods_sphere import get_pairwise_distances, angle
from centroid_finder import compute_principal_component_vecs

##################
# Kernel Methods #
##################

def binary_kernel(h, data, centroid): # works.
    '''
    Assumes data in (.., 3) shape
    '''
    dists = np.array(get_pairwise_distances(data, centroid))
    return data[(dists < h)]

def gaussian(x, centroid, h) -> int:
    mu = centroid
    sigma = h
    variance = sigma**2
    non_exp_part = 1/(sigma*math.sqrt(2*math.pi))
    exp_part = np.exp(-np.linalg.norm(x - mu)**2/2*variance)
    return non_exp_part * exp_part

def gaussian_kernel(h, data, centroid) -> "1d numpy array":
    '''
    Assumes data in (.., 3) shape,
    i.e each row is a point
    Unlike the binary kernel, it simply obtains the weights
    for each point, and does not change the input data.
    '''
    return np.apply_along_axis(gaussian, 1, data, centroid, h)

def multivar_gaussian_kernel(h, data, centroid):
    '''
    Assumes data in (.., 3) shape
    '''
    covar_mat = h * np.identity(3)
    distribution = scipy.stats.multivariate_normal(mean=centroid, cov=covar_mat)
    func = lambda point: point * distribution.pdf(point)
    vectorised = np.vectorize(func)
    return vectorised(data)

##################
# Main Algorithm #
##################

def compute_principal_component_vecs_weighted(vectors, p, weights):
    X = vectors
    n = len(vectors)
    covar_mat = np.zeros((X.shape[1],X.shape[1]))
    for i in range(n):
        covar_mat += np.multiply(np.outer(X[i,:], X[i,:]), weights[i])
    covar_mat /= sum(weights)

    eig_values, eig_vectors = np.linalg.eig(covar_mat)
    eig_tuples = list(zip(eig_values, eig_vectors.T))
    eig_tuples = sorted(eig_tuples, reverse=True)
    return eig_values, eig_tuples[0][1]


def principal_flow_weighted(centroid, data, epsilon, tol=1e-2):
    data = np.array(data)
    if data.shape[1] != 3:
        data = data.T
    h = 0.5
    points_on_sphere = data
    p = centroid
    p_opp = p
    curve = np.array(centroid)
    num_iter = 0
    while True:
        num_iter += 1
        if num_iter == 1:
            w = gaussian_kernel(h, points_on_sphere, p)
            plane_vectors = np.array(list(map(lambda point: log_map_sphere(p, point), points_on_sphere)))
            eig_values, principal_direction = compute_principal_component_vecs_weighted(plane_vectors, p, w)
            principal_direction_opp = - principal_direction
        
        else:
            # calculate for one direction, then the other 
            w = gaussian_kernel(h, points_on_sphere, p)
            plane_vectors = np.array(list(map(lambda point: log_map_sphere(p, point), points_on_sphere)))
            past_direction = principal_direction
            eig_values, principal_direction = compute_principal_component_vecs_weighted(plane_vectors, p, w)
            # make sure same direction
            if angle(past_direction, principal_direction) > math.pi/2:
                principal_direction = -principal_direction
            
            w_opp = gaussian_kernel(h, points_on_sphere, p_opp)
            plane_vectors_opp = np.array(list(map(lambda point: log_map_sphere(p_opp, point), points_on_sphere)))
            past_direction_opp = principal_direction_opp
            eig_values, principal_direction_opp = compute_principal_component_vecs_weighted(plane_vectors_opp, p_opp, w_opp)
            # make sure same direction
            if angle(past_direction_opp, principal_direction_opp) > math.pi/2:
                principal_direction_opp = -principal_direction_opp

        # update 1 direction
        p_prime_plane = p + epsilon * principal_direction
        p_prime = exp_map_sphere(p, p_prime_plane - p)
        p = p_prime

        # then the other
        p_prime_plane_opp = p_opp + epsilon * principal_direction_opp
        p_prime_opp = exp_map_sphere(p_opp, p_prime_plane_opp - p_opp)
        p_opp = p_prime_opp

        # now add to the curve
        curve = np.concatenate((curve, p))
        curve = np.concatenate((p_opp, curve))

        if test_eig_diff(eig_values, tol):
            # gap between eigenvalues are v small
            break
        if num_iter > 10:
            break
    curve = np.reshape(curve, (-1,3))
    return curve.T

def principal_flow(centroid, data, epsilon, tol=1e-2):
    data = np.array(data)
    if data.shape[1] != 3:
        data = data.T
    h = 0.5
    points_on_sphere = data
    p = centroid
    p_opp = p
    curve = np.array(centroid)
    num_iter = 0
    while True:
        num_iter += 1
        if num_iter == 1:
            filtered_points = binary_kernel(h, points_on_sphere, p)
            plane_vectors = np.array(list(map(lambda point: log_map_sphere(p, point), filtered_points)))
            eig_values, principal_direction = compute_principal_component_vecs(plane_vectors, p)
            principal_direction_opp = - principal_direction
        
        else:
            # calculate for one direction, then the other 
            filtered_points = binary_kernel(h, points_on_sphere, p)
            plane_vectors = np.array(list(map(lambda point: log_map_sphere(p, point), filtered_points)))
            past_direction = principal_direction
            eig_values, principal_direction = compute_principal_component_vecs(plane_vectors, p)
            # make sure same direction
            if angle(past_direction, principal_direction) > math.pi/2:
                principal_direction = -principal_direction
            
            filtered_points_opp = binary_kernel(h, points_on_sphere, p_opp)
            plane_vectors_opp = np.array(list(map(lambda point: log_map_sphere(p_opp, point), filtered_points_opp)))
            past_direction_opp = principal_direction_opp
            eig_values, principal_direction_opp = compute_principal_component_vecs(plane_vectors_opp, p_opp)
            # make sure same direction
            if angle(past_direction_opp, principal_direction_opp) > math.pi/2:
                principal_direction_opp = -principal_direction_opp

        # update 1 direction
        p_prime_plane = p + epsilon * principal_direction
        p_prime = exp_map_sphere(p, p_prime_plane - p)
        p = p_prime

        # then the other
        p_prime_plane_opp = p_opp + epsilon * principal_direction_opp
        p_prime_opp = exp_map_sphere(p_opp, p_prime_plane_opp - p_opp)
        p_opp = p_prime_opp

        # now add to the curve
        curve = np.concatenate((curve, p))
        curve = np.concatenate((p_opp, curve))

        if test_eig_diff(eig_values, tol):
            # gap between eigenvalues are v small
            break
        if num_iter > 10:
            break
    curve = np.reshape(curve, (-1,3))
    return curve.T


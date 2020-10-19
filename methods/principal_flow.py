import numpy as np
import pandas as pd
import scipy
from scipy.stats import norm
import math
from common_methods_sphere import log_map_sphere, exp_map_sphere
from common_methods_sphere import spherical_to_cartesian, generate_square, test_eig_diff
from common_methods_sphere import get_pairwise_distances, angle
from centroid_finder import compute_principal_component_vecs, sphere_centroid_finder_vecs

##################
# Kernel Methods #
##################

def binary_kernel(h, data, centroid):
    """ The 0-1 kernel. Assigns each point a 
    weight of 0 or 1. If the euclidean 
    distance fromt the point to centroid < h, 
    then it has a weight of 1, else 0.
    In other words, includes or excludes points,
    based on whether the point is within 
    the neighbourhood of size h centered
    at centroid.

    Args:
        h (float): [description]
        data (np.array,(n,p)): [description]
        centroid (np.array,(1,p) ): [description]

    Returns:
        1D numpy array: array of weights.
    """    
    dists = np.array(get_pairwise_distances(data, centroid))
    return data[(dists < h)]

def binary_kernel_weights(h, data, centroid):
    """ The 0-1 kernel. Assigns each point a 
    weight of 0 or 1. If the euclidean 
    distance fromt the point to centroid < h, 
    then it has a weight of 1, else 0.
    In other words, includes or excludes points,
    based on whether the point is within 
    the neighbourhood of size h centered
    at centroid.

    Args:
        h (float): [description]
        data (np.array,(n,p)): [description]
        centroid (np.array,(1,p) ): [description]

    Returns:
        1D numpy array: array of weights.
    """    
    dists = np.array(get_pairwise_distances(data, centroid))
    weights = (dists < h)
    return weights.astype(int)

def gaussian(x, centroid, h) -> float:
    """is the gaussian PDF, except we use 
    the euclidean norm of (x-mu) 
    instead of (x-mu)^2. This is to force it 
    to be a 1D gaussian, and to have a scalar output.
    
    Args:
        x (np.array, (1,p)): point to calculate weight for.
        centroid (np.array): center of gaussian.
        h ([type]): is the standard deviation of gaussian.

    Returns:
        float: weight of x
    """    
    mu = centroid
    sigma = h
    variance = sigma**2
    non_exp_part = 1/(sigma*math.sqrt(2*math.pi))
    exp_part = np.exp(-np.linalg.norm(x - mu)**2/2*variance)
    return non_exp_part * exp_part

def gaussian_kernel(h, data, centroid) -> "1d numpy array":
    """Weights each point with a 1D gaussian 
    centered around the centroid that has h standard deviation.

    We use this instead of the multivariate gaussian,
    since the ONLY info we want to use is the distance 
    between the point and the centroid. The multivariate 
    gaussian takes the position of the points into 
    consideration, through extra info like the precision matrix 
    (inverse cov matrix). This is unfavourable, as we should weight 
    the points by closeness to the centroid only.

    Args:
        h (float): scale
        data (np.array, (n,p)): [description]
        centroid (starting point for principal flow): [description]
    Returns:
        1D numpy array: a 1D array of weights.
    """ 
    return np.apply_along_axis(gaussian, 1, data, centroid, h)

def multivar_gaussian_kernel(h, data, centroid):
    """[summary]

    Args:
        h ([type]): [description]
        data ([type]): [description]
        centroid ([type]): [description]

    Returns:
        [type]: [description]
    """    
    covar_mat = h * np.identity(3)
    distribution = scipy.stats.multivariate_normal(mean=centroid, cov=covar_mat)
    func = lambda point: point * distribution.pdf(point)
    vectorised = np.vectorize(func)
    return vectorised(data)

##################
# Main Algorithm #
##################

def compute_principal_component_vecs_weighted(vectors, p, weights):
    """The weighted version of compute_principal_component_vecs:
    It cacluates the largest principal component for the points weighted 
    by some kernel. Follow equation 2 from the principal flow paper,
    namely: 

    Args:
        vectors (np.array, (n,p)): matrix of plane vectors, each row is a vector.
        p (np.array, (p,1)): the point at which the hypersphere is tangent to the plane, 
        where all the vectors are pointing outwards from on the plane.
        weights (np.array, (n,1)): weights calculated previously from some kernel.

    Returns:
        float, np.array (p,1): returns the largest principal component 
        and its associated eigenvalue.
    """    
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


def principal_flow_weighted(data, epsilon, centroid=None, tol=1e-2): 
    # note: non-default arguments must be placed before default
    """[summary]

    Args:
        data ([type]): [description]
        epsilon ([type]): [description]
        centroid ([type], optional): [description]. Defaults to None.
        tol ([type], optional): [description]. Defaults to 1e-2.

    Returns:
        [type]: [description]
    """    
    data = np.array(data)
    if data.shape[1] != 3:
        data = data.T
    if centroid == None:
        p = sphere_centroid_finder_vecs(data, 0.05, 0.01)
    else:
        p = centroid
    h = 0.1
    points_on_sphere = data
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

def principal_flow(data, epsilon, centroid=None, tol=1e-2):
    # note: non-default arguments must be placed before default
    """ Computes the principal flow of the dataset. 
    Idea: This is a "greedy" implmentation of the principal flow 
    algorithm, developed originally by Professor Yao Zhi Gang.

    Defn: The Principal Flow of the dataset is basically an integral curve 
    that is always tangent to the direction of 'maximal variation' at any given point it contains.
    
    We assume the underlying structure of the data is a hypersphere.
    Starting from the centroid of the data set (user defined or calculated below),
    we apply the following procedure: 

    1. Project the data residing on the hypersphere onto the hyperplane 
    tangent to the centroid of the data (p). We use the log map of p 
    for this purpose, obtaining a matrix of vectors on the tangent plane that point from p
    to the projected points, call it the plane vectors.

    2. Compute the largest principal component of the points using the plane vectors, 
    applying weights as necessary via the kernel function provided.

    3. The largest principal component and its opposite direction are the new directions
    that the principal flow moves in. We take a small step in each direction on the plane,
    then project it back to the hypersphere.

    4. Record the points. Repeat until max_iter is reached.

    Args:
        data (np.array, (n,p)): The data set, of shape (n,p), n = number of data points, p = dimension.
        epsilon (float): step size for the principal flow.
        tol ([type], optional): [description]. Defaults to 1e-2.
        centroid ([type]): [description] defaults to None.

    Returns:
        np.array: An array that contains the points of the principal flow.
    """    
    data = np.array(data)
    if data.shape[1] != 3:
        data = data.T
    h = 0.1
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
        if num_iter > 40:
            break
    curve = np.reshape(curve, (-1,3))
    return curve.T


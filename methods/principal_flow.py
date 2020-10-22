import math
import numpy as np
import pandas as pd
import scipy
from scipy.stats import norm
from common_methods_sphere import log_map_sphere, exp_map_sphere
from common_methods_sphere import spherical_to_cartesian, generate_square, test_eig_diff
from common_methods_sphere import get_pairwise_distances, angle
from centroid_finder import compute_principal_component_vecs, sphere_centroid_finder_vecs

##################
# Kernel Methods #
##################

def choose_h(points, p, percent=20):
    """ This function helps to choose a h 
    that accurately reconstructs the data, 
    h tries to let the centroid reach
    around 15% of the data.

    Args:
        points ([np.array,(n,p)]): [data]
        p ([np.array]): [description]
        percent (int, optional): [description]. Defaults to 15.

    Returns:
        [float]: [h value to use.]
    """    
    distances = sorted(get_pairwise_distances(points, p))
    index = math.ceil(len(points)*percent/100)
    print(index)
    return distances[index]

def binary_kernel_old(h, data, centroid):
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
        h (float): scale; smaller h, 
        larger weight on nearer points => smaller neighbourhood
        bigger h, more equal weights => larger neighbourhood 
        data (np.array, (n,p)): data we use to compute weights 
        centroid (np.array, (p,1)): the point at which the plane is tangent
        to the sphere, use as center for the kernel.

    Returns:
        1D numpy array: a 1D array of weights.
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

def new_gaussian(x, centroid, h) -> float:
    """is the gaussian PDF, except we use 
    the euclidean norm of (x-mu) 
    instead of (x-mu)^2. This is to force it 
    to be a 1D gaussian, and to have a scalar output.
    
    Args:
        x (np.array, (1,p)): point to calculate weight for.
        centroid (np.array): center of gaussian.
        h (float): is the standard deviation of gaussian.

    Returns:
        float: weight of x
    """    
    exp_part = np.exp(-np.linalg.norm(x - centroid)**2/2*h**2)
    return exp_part


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
        h (float): scale; smaller h, 
        larger weight on nearer points => smaller neighbourhood
        bigger h, more equal weights => larger neighbourhood 
        data (np.array, (n,p)): data we use to compute weights 
        centroid (np.array, (p,1)): the point at which the plane is tangent
        to the sphere, use as center for the kernel.

    Returns:
        1D numpy array: a 1D array of weights.
    """ 
    plane_data = np.array(list(map(lambda point: log_map_sphere(centroid, point), data)))
    return np.apply_along_axis(new_gaussian, 1, plane_data, centroid, h)

def identity_kernel(h, data, centroid):
    """ identity kernel simply spits out same data again.
    Is simply a default option.

    Args:
        h (float): does nothing here
        data (np.array, (n,p)): data we use to compute weights 
        centroid (np.array, (p,1)): does nothing here.

    Returns:
        1D numpy array: a 1D array of weights.
    """    
    return np.ones(len(data))

def multivar_gaussian_kernel(h, data, centroid):
    """3D gaussian kernel. Uses more info than we want.

    Args:
        h (float): scale; smaller h, 
        larger weight on nearer points => smaller neighbourhood
        bigger h, more equal weights => larger neighbourhood 
        data (np.array, (n,p)): data we use to compute weights 
        centroid (np.array, (p,1)): the point at which the plane is tangent
        to the sphere, use as center for the kernel.

    Returns:
        1D numpy array: a 1D array of weights.
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
    namely: .....

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


def principal_flow(data, epsilon, h, start_point=None, kernel_type="identity", tol=1e-2):
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
    tangent to the centroid of the data (p). We use the log map of p, 
    obtaining a matrix of vectors on the tangent plane that point from p
    to the projected points. These are the plane vectors.

    2. Compute the largest principal component of the points using the plane vectors, 
    applying weights as necessary via the kernel function provided.

    3. The largest principal component is the new directions that the principal flow moves in. 
    We determine it's sign by making sure it is moving in the 
    same direction as the previous principal direction.

    4. We take a small step in each direction on the plane,
    then project it back to the hypersphere.

    5. We do 1-4 for the point on the opposite end of the growing principal flow.

    6. Store both points. 
    
    7. Repeat 1-6 until max_iter is reached.

    Args:
        data (np.array, (n,p)): [The data set, of shape (n,p), n = number of data points, p = dimension.]

        epsilon (float): [step size for the principal flow.]

        h (float): [Scale. Determines how "local" the principal flow is. 
        Smaller scale => smaller neighbourhood, more emphasis on smaller pool of nearer points
        Bigger scale => bigger neighbourhood, emphasis on larger pool of points.]
        
        start_point (np.array, (p,1)): [the centroid, or the place to start the principal flow. 
        Defaults to None.]

        kernel_type (string): [specifies the kernel function. Default is the identity kernel, 
        which applies a weight of 1 to every point.]

        tol (float, optional): [useless for now.]

    Returns:
        np.array: An array that contains the points of the principal flow.
    """
    # handle data
    data = np.array(data)
    if data.shape[1] != 3:
        data = data.T
    points_on_sphere = data
    
    # handle starting point
    if type(start_point) == None:
        p = sphere_centroid_finder_vecs(data, 0.05, 0.01)
    else:
        assert type(start_point) == np.array or \
            type(start_point) == np.ndarray, "Start point must be an np.array or an np.ndarray"
        p = start_point
    flow = np.array(p)

    # handle kernel
    kernel_functions = {"binary": binary_kernel, "gaussian": gaussian_kernel, "identity": identity_kernel}
    assert kernel_type in kernel_functions.keys(), "Kernel must be binary, gaussian or identity."
    kernel = kernel_functions[kernel_type]

    # handle scale of kernel
    # TODO calculate range of h's and give a value for reconstructing data relatively accurately.
    # for now h is input

    p_opp = p
    num_iter = 0
    while True:
        num_iter += 1
        if num_iter == 1:
            weights = gaussian_kernel(h, points_on_sphere, p)
            plane_vectors = np.array(list(map(lambda point: log_map_sphere(p, point), points_on_sphere)))
            _ , principal_direction = compute_principal_component_vecs_weighted(plane_vectors, p, weights)
            principal_direction_opp = - principal_direction
        
        else:
            # calculate for one direction, then the other 
            weights = kernel(h, points_on_sphere, p)
            plane_vectors = np.array(list(map(lambda point: log_map_sphere(p, point), points_on_sphere)))
            past_direction = principal_direction
            _ , principal_direction = compute_principal_component_vecs_weighted(plane_vectors, p, weights)
            # make sure same direction
            if angle(past_direction, principal_direction) > math.pi/2:
                principal_direction = -principal_direction
            
            weights_opp = kernel(h, points_on_sphere, p_opp)
            plane_vectors_opp = np.array(list(map(lambda point: log_map_sphere(p_opp, point), points_on_sphere)))
            past_direction_opp = principal_direction_opp
            eig_values, principal_direction_opp = compute_principal_component_vecs_weighted(plane_vectors_opp, p_opp, weights_opp)
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
        flow = np.concatenate((flow, p))
        flow = np.concatenate((p_opp, flow))

        if num_iter > 40:
            break
    flow = np.reshape(flow, (-1,3))
    return flow


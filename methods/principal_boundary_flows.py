import math
import numpy as np
import matplotlib.pyplot as plt
from principal_flow import choose_h_gaussian, choose_h_binary, \
    compute_principal_component_vecs_weighted, \
    binary_kernel, gaussian_kernel, identity_kernel
from common_methods_sphere import log_map_sphere, exp_map_sphere, angle
from centroid_finder import sphere_centroid_finder_vecs


def principal_boundary(data, dimension, epsilon, h, radius, start_point=None, \
    kernel_type="identity", tol=1e-2, max_iter=40):
    # note: non-default arguments must be placed before default
    """ Computes the principal boundary of the dataset.
    Idea: This is a "greedy" implmentation of the principal boundary
    algorithm, developed originally by Professor Yao Zhi Gang.
    
    It uses the greedy version of the pricipal flow algorithm.

    Args:
        data (np.array, (n,p)): [The data set, of shape (n,p), n = number of data points, p = dimension.]

        dimension (integer): dimension of data

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
    data = np.array(data)
    if data.shape[1] != dimension:
        data = data.T
    
    # handle starting point
    if type(start_point) == None:
        p = sphere_centroid_finder_vecs(data, 3, 0.05, 0.01)
    else:
        # error report: for checking 
        assert type(start_point) is not np.array or \
            type(start_point) is not np.ndarray, "Start point must be an np.array or an np.ndarray"
        p = start_point

    upper_boundary = np.array([])
    flow = np.array(p)
    lower_boundary = np.array([])
    # handle kernel
    kernel_functions = {"binary": binary_kernel, "gaussian": gaussian_kernel, "identity": identity_kernel}
    assert kernel_type in kernel_functions.keys(), "Kernel must be binary, gaussian or identity."
    kernel = kernel_functions[kernel_type]

    p_opp = p
    num_iter = 0
    upper_direction = []
    while True:
        num_iter += 1
        if num_iter == 1:
            weights = kernel(h, data, p)
            plane_vectors = np.array(list(map(lambda point: log_map_sphere(p, point), data)))
            try:
                principal_pair, boundary_pair = compute_principal_component_vecs_weighted(\
                    plane_vectors, p, weights, boundary=True)
            except ValueError:
                print("Flow ends here, the covariance matrix is 0, implying that the flow is far from the data.")
                break

            # update boundary
            first_eigenval = principal_pair[0]
            second_eigenval = boundary_pair[0]
            first_orthogonal = boundary_pair[1]
            sigma_f_p = second_eigenval/first_eigenval * radius  # how much to move for boundary
            upper_boundary_point = p + sigma_f_p * upper_direction
            upper_boundary = np.concatenate((upper_boundary, upper_boundary_point))
            lower_boundary_point = p - sigma_f_p * upper_direction
            lower_boundary = np.concatenate((lower_boundary, lower_boundary_point))

            # for flow
            principal_direction = principal_pair[1]
            principal_direction_opp = - principal_direction

        else:
            # calculate for one direction, then the other 
            weights = kernel(h,data, p)
            plane_vectors = np.array(list(map(lambda point: log_map_sphere(p, point), data)))
            past_direction = principal_direction
            try:
                principal_pair, boundary_pair = compute_principal_component_vecs_weighted(\
                    plane_vectors, p, weights, boundary=True)
            except ValueError:
                print("Flow ends here, the covariance matrix is 0, implying that the flow is far from the data.")
                break
           
            # First we update the main point for the flow:
            # Get principal direction
            principal_direction = principal_pair[1]
            if angle(past_direction, principal_direction) > math.pi/2:
                principal_direction = - principal_direction
            # update point p
            p_prime_plane = p + epsilon * principal_direction
            p_prime = exp_map_sphere(p, p_prime_plane - p)
            p = p_prime

            # obtain boundary for this point - first we obtain intial info
            first_eigenval = principal_pair[0]
            second_eigenval = boundary_pair[0]
            orthogonal_to_flow = boundary_pair[1]
            if angle(orthogonal_to_flow, first_orthogonal) > math.pi/2:
                orthogonal_to_flow = - orthogonal_to_flow
        
            # move in direction orthogonal to flow, a distance of sigma_f_p
            sigma_f_p = second_eigenval/first_eigenval * radius

            # get both sides of the boundary + and - orthogonal_to_flow
            upper_boundary_point = p + sigma_f_p * orthogonal_to_flow
            upper_boundary = np.concatenate((upper_boundary, upper_boundary_point))
            lower_boundary_point = p - sigma_f_p * orthogonal_to_flow
            lower_boundary = np.concatenate((lower_boundary, lower_boundary_point))
           
            weights_opp = kernel(h, data, p_opp)
            plane_vectors_opp = np.array(list(map(lambda point: log_map_sphere(p_opp, point), data)))
            past_direction_opp = principal_direction_opp
            try:
                principal_pair_opp, boundary_pair_opp = compute_principal_component_vecs_weighted(\
                   plane_vectors_opp, p, weights_opp, boundary=True)
            except ValueError:
                print("Flow ends here, the covariance matrix is 0, implying that the flow is far from the data.")
                break

            # make sure same direction
            principal_direction_opp = principal_pair_opp[1]
            if angle(past_direction_opp, principal_direction_opp) > math.pi/2:
                principal_direction_opp = - principal_direction_opp

            # now we do the other direction
            p_prime_plane_opp = p_opp + epsilon * principal_direction_opp
            p_prime_opp = exp_map_sphere(p_opp, p_prime_plane_opp - p_opp)
            p_opp = p_prime_opp
          
            # get info again
            first_eigenval_opp = principal_pair_opp[0]
            second_eigenval_opp = boundary_pair_opp[0]
            orthogonal_to_flow_opp = boundary_pair_opp[1]

            if angle(orthogonal_to_flow_opp, first_orthogonal) > math.pi/2:
                orthogonal_to_flow_opp = - orthogonal_to_flow_opp
          
            sigma_f_p_opp = second_eigenval_opp/first_eigenval_opp * radius
            upper_boundary_point_opp = p_opp + sigma_f_p_opp * orthogonal_to_flow_opp
            upper_boundary = np.concatenate((upper_boundary, upper_boundary_point_opp))
            lower_boundary_point_opp = p_opp - sigma_f_p_opp * orthogonal_to_flow_opp
            lower_boundary = np.concatenate((lower_boundary, lower_boundary_point_opp))

            # now add to the curve
            flow = np.concatenate((flow, p))
            flow = np.concatenate((p_opp, flow))

        if num_iter > max_iter:
            break
    flow = np.reshape(flow, (-1, dimension))
    return upper_boundary, flow, lower_boundary


def calculate_radius(data, p):
    
    return 0